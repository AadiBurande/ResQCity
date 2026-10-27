"""Solve time of C1 against instance size (phase 6 acceptance criterion).

    uv run python -m resqcity.optimization.bench_dispatch

The phase requires a realistic live batch -- about 10 incidents and the whole 91-unit fleet --
to be answered within 1 s, or a documented fallback. ETAs are drawn from a fixed seed rather
than the road graph: the question here is solver time against size, and the graph work is
measured separately in the routing benchmark.

Each size is solved TWICE, because an OR review of this phase found the benchmark timing a
reduced model and the acceptance criterion being read off it:

* ``full`` -- the green corridor (C7) and ward coverage (C8) both active, which is what live
  dispatch actually solves. The corridor adds a reified AND per eligible ambulance and the
  coverage term a free/busy indicator per unit per ward, so this is the honest number.
* ``reduced`` -- neither term, i.e. what the benchmark used to report. Kept so the cost of the
  two terms is visible rather than inferred.

The acceptance criterion is judged on ``full``.
Output: ``results/optimization/solver_bench.json``.
"""

import argparse
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.common.schemas import Agency, ServiceCounts, Unit, UnitState
from resqcity.optimization.config import load_optimization_config
from resqcity.optimization.dispatch import DispatchIncident, solve
from resqcity.optimization.routing import EtaTable, green_corridor_etas

OUT = PROJECT_ROOT / "results" / "optimization" / "solver_bench.json"
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 3, 3, 9, 0, tzinfo=IST)
SIZES = [(2, 20), (5, 45), (10, 91), (20, 91), (40, 91)]
SEED = 20261027
MIX = {Agency.AMBULANCE: 30, Agency.FIRE: 22, Agency.POLICE: 39}
WARDS = 58  # PMC electoral wards, as in phase 2
COVERERS_PER_WARD = 6  # synthetic: how many units are within the coverage radius of a ward
log = logging.getLogger(__name__)


def instance(
    n_incidents: int, n_units: int, rng: np.random.Generator
) -> tuple[list[DispatchIncident], list[Unit], EtaTable]:
    """Units in the measured fleet proportions; ETAs lognormal around ~7 min, 5 % unreachable."""
    agencies: list[Agency] = []
    for agency, share in MIX.items():
        agencies += [agency] * max(1, round(n_units * share / sum(MIX.values())))
    agencies = agencies[:n_units] or [Agency.POLICE]
    units = [
        Unit(id=f"U{i:03d}", agency=a, home_station=f"S{i:03d}", node=i + 1,
             state=UnitState.IDLE, eta_fn_ref="default")
        for i, a in enumerate(agencies)
    ]  # fmt: skip
    etas = 420.0 * np.exp(0.6 * rng.standard_normal((len(units), n_incidents)))
    etas[rng.random(etas.shape) < 0.05] = np.inf
    severities = rng.choice(["minor", "serious", "critical"], n_incidents, p=[0.5, 0.35, 0.15])
    incidents = []
    for i in range(n_incidents):
        needs = {
            "police": 1,
            "ambulance": int(rng.random() < 0.6),
            "fire": int(rng.random() < 0.25),
        }
        incidents.append(
            DispatchIncident(
                id=f"I{i:03d}",
                node=1000 + i,
                requirements=ServiceCounts(**needs),
                severity_class=str(severities[i]),
                ward_id=f"W{i % 58 + 1:02d}",
            )
        )
    table = EtaTable(list(range(len(units))), list(range(n_incidents)), etas)
    return incidents, units, table


def coverage(
    n_units: int, rng: np.random.Generator
) -> tuple[dict[str, list[int]], dict[str, float]]:
    """Which units can cover each ward, and the ward's risk weight (C8 inputs).

    SYNTHETIC structure, not the real coverage map: each ward draws a fixed number of units
    within its radius. The point is to load the solver with the same NUMBER of indicator
    variables the live model carries, not to be geographically faithful.
    """
    units = {}
    for w in range(WARDS):
        size = min(n_units, COVERERS_PER_WARD)
        units[f"W{w + 1:02d}"] = sorted(rng.choice(n_units, size=size, replace=False).tolist())
    value = {w: float(v) for w, v in zip(units, rng.gamma(2.0, 0.5, WARDS), strict=True)}
    return units, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_optimization_config()
    rng = np.random.default_rng(SEED)
    rows = []
    for n_i, n_u in SIZES:
        measured: dict[str, dict[str, object]] = {}
        # the same instances for both variants, so the comparison is paired
        batch = [instance(n_i, n_u, rng) for _ in range(args.repeats)]
        covers = [coverage(n_u, rng) for _ in range(args.repeats)]
        for variant in ("full", "reduced"):
            times, statuses = [], []
            for (incidents, units, table), (ward_units, ward_value) in zip(
                batch, covers, strict=True
            ):
                extra: dict[str, object] = {}
                if variant == "full":
                    agencies = [u.agency for u in units]
                    extra = {
                        "assisted_etas": green_corridor_etas(cfg, table, agencies),
                        "ward_units": ward_units,
                        "ward_value": ward_value,
                    }
                result = solve(incidents, units, table, cfg, NOW, **extra)
                times.append(result.solve_s)
                statuses.append(result.status)
            measured[variant] = {
                "median_solve_s": float(np.median(times)),
                "max_solve_s": float(max(times)),
                "statuses": sorted(set(statuses)),
                "within_limit": bool(max(times) <= cfg.dispatch.solver.time_limit_s),
            }
        rows.append({"incidents": n_i, "units": n_u, "repeats": args.repeats, **measured})
        log.info("%2d incidents x %2d units: full median %.3f s max %.3f s | reduced median "
                 "%.3f s max %.3f s | %s",
                 n_i, n_u, measured["full"]["median_solve_s"], measured["full"]["max_solve_s"],
                 measured["reduced"]["median_solve_s"], measured["reduced"]["max_solve_s"],
                 ",".join(measured["full"]["statuses"]))  # fmt: skip
    live = next(r for r in rows if (r["incidents"], r["units"]) == (10, 91))
    worst = max(rows, key=lambda r: r["full"]["max_solve_s"])
    report = {
        "status": "synthetic ETAs and a synthetic ward-coverage map; measures solver time "
        "against instance size, not route quality",
        "variants": {
            "full": "green corridor (C7) and ward coverage (C8) active -- what live dispatch "
            "solves, and what the acceptance criterion is judged on",
            "reduced": "neither term; what this benchmark reported before the OR review",
        },
        "time_limit_s": cfg.dispatch.solver.time_limit_s,
        "live_batch_within_limit": live["full"]["within_limit"],
        "live_batch_median_s": live["full"]["median_solve_s"],
        "live_batch_max_s": live["full"]["max_solve_s"],
        "worst_case": {
            "incidents": worst["incidents"],
            "units": worst["units"],
            "max_solve_s": worst["full"]["max_solve_s"],
            "within_limit": worst["full"]["within_limit"],
        },
        "sizes": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(
        OUT, configs=[CONFIG_DIR / "optimization.yaml"], synthetic=True, seeds={"etas": SEED}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
