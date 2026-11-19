"""Experiment grid runner (phase 10).

    uv run python -m resqcity.experiments.run [--quick] [--scenario normal] [--policy B0]

Runs the full (scenario x policy x replication) matrix with common random numbers, computes
per-incident metrics, runs paired statistical tests, and writes everything to
``results/experiments/``.

``--quick`` cuts replications to 3 and runs only the ``normal`` scenario for development.

Design notes
~~~~~~~~~~~~
* The expensive part is ``sim.runner.run_replication``. Everything else is seconds.
* The ``RoadNetwork`` is loaded once and reused across all runs for one scenario.
* Common random numbers: ``Week`` is deterministic given (scenario, seed_base, rep), so
  the same week is generated for every policy in that (scenario, rep) cell.
* Results are written incrementally: each (scenario, policy) cell writes its own parquet
  and JSON so a crash does not lose earlier cells.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.experiments.metrics import (
    AggregateMetrics,
    aggregate_metrics,
    compute_incident_metrics,
)
from resqcity.experiments.stats import (
    PairedTestResult,
    holm_bonferroni,
    non_inferiority_test,
    paired_test,
)
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.sim.config import POLICIES, load_sim_config, load_survival_config
from resqcity.sim.runner import Replication, run_replication, to_frame

log = logging.getLogger(__name__)

OUT = PROJECT_ROOT / "results" / "experiments"
SCENARIOS = ("normal", "monsoon", "ganeshotsav")


# ---------------------------------------------------------------------------
# single cell: (scenario, policy, all reps)
# ---------------------------------------------------------------------------


def _run_cell(
    network: RoadNetwork,
    scenario: str,
    policy: str,
    n_reps: int,
    seed_base: int,
) -> tuple[list[Replication], pd.DataFrame]:
    """Run all replications for one (scenario, policy) cell and return raw results."""
    reps: list[Replication] = []
    for rep in range(n_reps):
        log.info("  rep %d/%d ...", rep + 1, n_reps)
        result = run_replication(network, scenario, policy, rep, seed_base)
        reps.append(result)
    frame = to_frame(reps)
    return reps, frame


def _compute_metrics_frame(
    frame: pd.DataFrame,
    survival: load_survival_config | None = None,
) -> pd.DataFrame:
    """Convert raw IncidentRow frame to per-incident metrics frame."""
    rows = []
    for _, row in frame.iterrows():
        m = compute_incident_metrics(row.to_dict(), survival=survival)
        rows.append(asdict(m))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# write one cell's outputs
# ---------------------------------------------------------------------------


def _write_cell(
    out_dir: Path,
    scenario: str,
    policy: str,
    reps: list[Replication],
    metrics_df: pd.DataFrame,
    agg: AggregateMetrics,
    seed_base: int,
) -> None:
    """Write parquet + summary JSON for one (scenario, policy) cell."""
    stem = f"{scenario}_{policy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # parquet of per-incident metrics
    pq_path = out_dir / f"metrics_{stem}.parquet"
    metrics_df.to_parquet(pq_path, index=False)

    # aggregate summary
    summary = {
        "scenario": scenario,
        "policy": policy,
        "replications": len(reps),
        "seed_base": seed_base,
        "n_incidents": agg.n_incidents,
        "primary_mean_s": _safe(agg.primary_mean_s),
        "primary_median_s": _safe(agg.primary_median_s),
        "primary_p90_s": _safe(agg.primary_p90_s),
        "pct_within_8min": _safe(agg.pct_within_8min),
        "pct_within_15min": _safe(agg.pct_within_15min),
        "detection_delay_mean_s": _safe(agg.detection_delay_mean_s),
        "detection_to_dispatch_mean_s": _safe(agg.detection_to_dispatch_mean_s),
        "first_to_last_gap_mean_s": _safe(agg.first_to_last_gap_mean_s),
        "expected_survival_mean": _safe(agg.expected_survival_mean),
        "ward_gini": _safe(agg.ward_gini),
        "false_alarm_incidents": agg.false_alarm_incidents,
        "utilisation": {
            a: _safe(getattr(agg, f"utilisation_{a}")) for a in ("ambulance", "police", "fire")
        },
        "wall_seconds": [r.wall_s for r in reps],
        "evidence_digests": [r.evidence_digest for r in reps],
    }
    json_path = out_dir / f"summary_{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n")
    write_meta(
        json_path,
        configs=[CONFIG_DIR / "sim.yaml", CONFIG_DIR / "survival.yaml"],
        synthetic=True,
        seeds={"seed_base": seed_base},
    )


def _safe(v: float) -> float | None:
    """Replace NaN/inf with None for JSON serialisation."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return v


# ---------------------------------------------------------------------------
# pairwise statistical tests
# ---------------------------------------------------------------------------


def _run_pairwise_tests(
    aggregates: dict[str, AggregateMetrics],
    policy_list: list[str],
) -> list[PairedTestResult]:
    """Run all pairwise paired tests on the primary outcome."""
    results: list[PairedTestResult] = []
    for i, pa in enumerate(policy_list):
        for pb in policy_list[i + 1 :]:
            agg_a = aggregates.get(pa)
            agg_b = aggregates.get(pb)
            if agg_a is None or agg_b is None:
                continue
            vals_a = np.array(agg_a.per_rep_primary_means)
            vals_b = np.array(agg_b.per_rep_primary_means)
            if len(vals_a) < 3 or len(vals_b) < 3:
                log.warning("Skipping %s vs %s: too few reps", pa, pb)
                continue
            r = paired_test(
                vals_a,
                vals_b,
                metric="primary_mean_s",
                policy_a=pa,
                policy_b=pb,
            )
            results.append(r)
    return results


def _write_stats(
    out_dir: Path,
    test_results: list[PairedTestResult],
    equity_result: object | None,
    scenario: str,
) -> None:
    """Write statistical test results to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if test_results:
        rows = [asdict(r) for r in test_results]
        # drop per_rep fields for CSV
        for row in rows:
            row.pop("per_rep_primary_means", None)
            row.pop("per_rep_ginis", None)
        pd.DataFrame(rows).to_csv(out_dir / f"paired_tests_{scenario}.csv", index=False)
    if equity_result is not None:
        pd.DataFrame([asdict(equity_result)]).to_csv(
            out_dir / f"equity_{scenario}.csv", index=False
        )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_experiments(
    scenarios: tuple[str, ...] = SCENARIOS,
    policies: tuple[str, ...] = POLICIES,
    n_reps: int | None = None,
    out_dir: Path = OUT,
) -> dict[str, dict[str, AggregateMetrics]]:
    """Run the full experiment grid.

    Returns a nested dict: ``results[scenario][policy] = AggregateMetrics``.
    """
    base = load_base_config()
    sim_cfg = load_sim_config()
    survival = load_survival_config()
    seed_base = base.random.seed_base
    if n_reps is None:
        n_reps = sim_cfg.run.replications

    all_results: dict[str, dict[str, AggregateMetrics]] = {}

    for scenario in scenarios:
        log.info("=== Scenario: %s ===", scenario)
        geo_cfg = load_geo_config()
        network = RoadNetwork.load(base.paths.resolve("data_processed"), geo_cfg)

        scenario_aggregates: dict[str, AggregateMetrics] = {}
        scenario_reps: dict[str, list[Replication]] = {}

        for policy in policies:
            t0 = time.perf_counter()
            log.info("--- Policy: %s (%d reps) ---", policy, n_reps)

            reps, raw_frame = _run_cell(network, scenario, policy, n_reps, seed_base)
            metrics_df = _compute_metrics_frame(raw_frame, survival=survival)

            utilisations = [r.utilisation for r in reps]
            agg = aggregate_metrics(
                metrics_df,
                scenario,
                policy,
                utilisations=utilisations,
                survival=survival,
            )

            _write_cell(out_dir, scenario, policy, reps, metrics_df, agg, seed_base)
            scenario_aggregates[policy] = agg
            scenario_reps[policy] = reps

            elapsed = time.perf_counter() - t0
            log.info(
                "  %s/%s: %d incidents, primary mean %.1f s, %.1f s wall",
                scenario,
                policy,
                agg.n_incidents,
                agg.primary_mean_s,
                elapsed,
            )

        # pairwise tests for this scenario
        test_results = _run_pairwise_tests(scenario_aggregates, list(policies))
        corrected = holm_bonferroni(test_results)

        # equity non-inferiority (RQ4)
        equity = None
        if "FULL" in scenario_aggregates and "B0" in scenario_aggregates:
            equity = non_inferiority_test(
                np.array(scenario_aggregates["FULL"].per_rep_ginis),
                np.array(scenario_aggregates["B0"].per_rep_ginis),
            )

        _write_stats(out_dir / "stats", corrected, equity, scenario)
        all_results[scenario] = scenario_aggregates

        # log test readings
        for r in corrected:
            log.info("  TEST: %s", r.reading)
        if equity is not None:
            log.info("  EQUITY: %s", equity.reading)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 10 experiment grid.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Development mode: 3 reps, normal scenario only.",
    )
    parser.add_argument("--scenario", default=None, help="Single scenario to run.")
    parser.add_argument("--policy", default=None, help="Single policy to run.")
    parser.add_argument("--reps", type=int, default=None, help="Override replication count.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.quick:
        scenarios = ("normal",)
        n_reps = 3
        policies = POLICIES
    else:
        scenarios = (args.scenario,) if args.scenario else SCENARIOS
        n_reps = args.reps
        policies = (args.policy,) if args.policy else POLICIES

    run_experiments(scenarios=scenarios, policies=policies, n_reps=n_reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
