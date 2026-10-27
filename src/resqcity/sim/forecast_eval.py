"""Fit and score the ward-hour demand forecasters (phase 5, D-030).

    uv run python -m resqcity.sim.forecast_eval

Time-based split, never shuffled: the last ``test_weeks`` of the history are held out, and
LightGBM picks its stopping round on the last ``validation_weeks`` of the TRAINING part only.

**Forecast accuracy here is not a research claim.** The training data comes from the incident
generator, whose intensity is a formula this project wrote, so the ``informed`` model -- handed
the generator's own rain and ward weights -- recovers that formula almost exactly. It is reported
as the circular upper bound it is. The research claim is only how dispatch performance depends on
forecast quality, which phase 10 sweeps using these models as its quality levels.

Metrics (per model, at ward-hour and ward-day): Poisson deviance, MAE (reported but nearly
useless on a panel where 99.6 % of cells are zero -- predicting zero everywhere scores well),
a calibration table over predicted deciles, and the top-k ward hit rate, which is what
pre-positioning actually cares about.
Outputs ``results/forecasting/metrics_synthetic.json`` and ``calibration_synthetic.csv``.

It lives in ``sim`` rather than ``forecasting`` because it uses the declared oracle ablation
and the panel builder, and ``tests/test_architecture.py`` forbids ``forecasting -> sim``.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import meta_path, sha256_file, write_meta
from resqcity.forecasting.config import ForecastingConfig, load_forecasting_config
from resqcity.forecasting.metrics import (
    BOOTSTRAP_SEED,
    bootstrap_ci,
    per_hour_hits,
    score,
    split,
)
from resqcity.forecasting.models import (
    Climatology,
    Forecaster,
    LightGbmPoisson,
    NoisyForecaster,
    PerCellClimatology,
    UniformRate,
)

OUT = PROJECT_ROOT / "results" / "forecasting"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "forecast"
NOISE_LEVELS = (0.5, 1.0)
log = logging.getLogger(__name__)


def fit_all(cfg: ForecastingConfig, parts: dict[str, pd.DataFrame]) -> list[Forecaster]:
    from resqcity.sim.oracle_forecast import OracleForecast  # declared oracle ablation

    train, valid = parts["train"], parts["valid"]
    # the boosters choose their stopping round on `valid`, so the baselines are fitted on
    # train+valid too: otherwise they see 36 % less data than the model they are compared with
    history = pd.concat([train, valid], ignore_index=True)
    fitted: list[Forecaster] = [
        UniformRate.fit(history),
        Climatology.fit(history),
        PerCellClimatology.fit(history),
    ]
    for name in ("realistic", "informed"):
        log.info("fitting lightgbm on the %s feature set (%d rows)", name, len(train))
        model = LightGbmPoisson.fit(cfg, name, train, valid)
        model.save(MODEL_DIR / model.name)
        log.info("  stopped at iteration %d", model.best_iteration)
        fitted.append(model)
    fitted.append(OracleForecast())
    return fitted


def _panel_provenance(path: Path) -> dict[str, object]:
    """Identify the exact panel these metrics were computed from, not just its file name.

    The panel is a large intermediate that is not committed, so a reader of the results file
    needs its hash and the commit it was built at to reproduce the numbers.
    """
    out: dict[str, object] = {"file": path.name, "sha256": sha256_file(path)}
    side = meta_path(path)
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        out |= {
            "built_at_utc": meta.get("generated_at_utc"),
            "built_at_commit": meta.get("git", {}).get("commit"),
            "configs_sha256": meta.get("configs_sha256"),
            "seeds": meta.get("seeds"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from resqcity.sim.panel import panel_path  # path helper only; no ground truth is read here

    cfg = load_forecasting_config()
    base = load_base_config()
    panel = pd.read_parquet(args.panel or panel_path(cfg))
    parts = split(panel, cfg)
    log.info("split: %s", {k: len(v) for k, v in parts.items()})
    models = fit_all(cfg, parts)
    rows = parts["test"]
    metrics, tables = [], []
    for model in models:
        m, t = score(rows, model.predict(rows), model.name)
        metrics.append(m)
        tables += t
        log.info(
            "%-22s deviance %.5f  top-5 %.3f  predicted %.0f vs %.0f observed",
            model.name, m["poisson_deviance"], m["top_5_ward_hit_rate"],
            m["predicted"], m["observed"],
        )  # fmt: skip
    uncertainty = {}
    for model in models:
        pred = model.predict(rows)
        y = rows["count"].to_numpy(float)
        mu = np.maximum(pred, 1e-12)
        unit = 2.0 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))
        by_hour = (
            pd.DataFrame({"ts": rows["ts"].to_numpy(), "dev": unit})
            .groupby("ts", observed=True)["dev"]
            .agg(["sum", "count"])
        )
        hits = per_hour_hits(rows, pred, 5)
        uncertainty[model.name] = {
            "poisson_deviance_ci95": bootstrap_ci(
                by_hour["sum"].to_numpy(), by_hour["count"].to_numpy()
            ),
            "top_5_ward_hit_rate_ci95": bootstrap_ci(
                hits["hits"].to_numpy(float), hits["total"].to_numpy(float)
            ),
        }
    by_name = {m["model"]: m for m in metrics}
    floor, ceiling = by_name["uniform"], by_name["oracle"]
    for row in metrics:
        for key in ("poisson_deviance", "top_5_ward_hit_rate"):
            span = floor[key] - ceiling[key]
            row[f"{key}_share_of_gap_closed"] = (
                (floor[key] - row[key]) / span if span else float("nan")
            )
    uniform_model = next(m for m in models if m.name == "uniform")
    name_tie = per_hour_hits(rows, uniform_model.predict(rows), 5, tie_break="name")
    tie_artefact = {
        "what": "a model predicting the same value everywhere has no real ranking; breaking ties "
        "by ward name made it silently 'choose' the alphabetically first wards",
        "uniform_top_5_name_tiebreak": float(name_tie["hits"].sum() / name_tie["total"].sum()),
        "uniform_top_5_random_tiebreak": by_name["uniform"]["top_5_ward_hit_rate"],
        "analytic_k_over_wards": 5 / rows["ward"].nunique(),
    }
    percell = next(m for m in models if m.name == "climatology_percell")
    percell_pred = percell.predict(rows)
    degenerate = {
        "what": "why a per-cell historical average is not a usable baseline at this sparsity",
        "cells_predicted_exactly_zero": percell.zero_share(rows),
        "test_incidents_landing_on_a_zero": int(rows["count"].to_numpy()[percell_pred == 0].sum()),
        "test_incidents_total": int(rows["count"].sum()),
        "mean_incidents_per_cell_in_history": float(
            (parts["train"]["count"].sum() + parts["valid"]["count"].sum())
            / (rows["ward"].nunique() * 168 * len(cfg.targets.types))
        ),
    }
    base_model = next(m for m in models if m.name == "lightgbm_realistic")
    for sigma in NOISE_LEVELS:
        noisy = NoisyForecaster(base_model, sigma, base.random.forecast_training_seed_base)
        m, t = score(rows, noisy.predict(rows), noisy.name)
        metrics.append(m)
        tables += t
    importances = {m.name: m.importance() for m in models if isinstance(m, LightGbmPoisson)}
    report = {
        "status": "SYNTHETIC. Forecast accuracy is NOT a research claim: the training data comes "
        "from the project's own incident generator, so the 'informed' model recovers its "
        "intensity formula by construction. The claim is only how dispatch performance depends "
        "on forecast quality (phase 10).",
        "panel": _panel_provenance(args.panel or panel_path(cfg)),
        "scenario": cfg.history_scenario,
        "split_rows": {k: len(v) for k, v in parts.items()},
        "test_weeks": cfg.split.test_weeks,
        "feature_sets": cfg.feature_sets,
        "metrics": metrics,
        "uncertainty": {
            "method": "paired bootstrap over test HOURS (the unit of dependence: wards compete "
            "within an hour), 2000 resamples",
            "seed": BOOTSTRAP_SEED,
            "by_model": uncertainty,
        },
        "gap_note": "*_share_of_gap_closed is the share of the uniform->oracle span a model "
        "closes. The realistic model closing most of it IS the circularity of this phase (D-030).",
        "fitting_note": "The baselines are fitted on train+valid; the boosters are fitted on "
        "train and use valid only to choose a stopping round. That residual asymmetry FAVOURS "
        "the baselines by "
        f"{len(parts['valid']) / max(1, len(parts['train'])):.0%} more fitting rows, so it "
        "cannot explain the boosters' lower deviance. The oracle is ground truth, not a fit.",
        "tie_break_artefact": tie_artefact,
        "per_cell_climatology_is_degenerate": degenerate,
        "lightgbm_gain_importance": importances,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_synthetic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / "calibration_synthetic.csv", index=False, lineterminator="\n"
    )
    configs = [CONFIG_DIR / "forecasting.yaml", CONFIG_DIR / "incidents.yaml"]
    seeds = {"forecast_training_seed_base": base.random.forecast_training_seed_base}
    for path in (OUT / "metrics_synthetic.json", OUT / "calibration_synthetic.csv"):
        write_meta(path, configs=configs, synthetic=True, seeds=seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
