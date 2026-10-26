"""Solve time of C1 against instance size (phase 6 acceptance criterion).

    uv run python -m resqcity.optimization.bench_dispatch

The phase requires a realistic live batch -- about 10 incidents and the whole 91-unit fleet --
to be answered within 1 s, or a documented fallback. ETAs are drawn from a fixed seed rather
than the road graph: the question here is solver time against size, and the graph work is
measured separately in the routing benchmark.
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
from resqcity.optimization.routing import EtaTable

OUT = PROJECT_ROOT / "results" / "optimization" / "solver_bench.json"
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 3, 3, 9, 0, tzinfo=IST)
SIZES = [(2, 20), (5, 45), (10, 91), (20, 91), (40, 91)]
SEED = 20261027
MIX = {Agency.AMBULANCE: 30, Agency.FIRE: 22, Agency.POLICE: 39}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_optimization_config()
    rng = np.random.default_rng(SEED)
    rows = []
    for n_i, n_u in SIZES:
        times, statuses = [], []
        for _ in range(args.repeats):
            incidents, units, table = instance(n_i, n_u, rng)
            result = solve(incidents, units, table, cfg, NOW)
            times.append(result.solve_s)
            statuses.append(result.status)
        rows.append({
            "incidents": n_i, "units": n_u, "repeats": args.repeats,
            "median_solve_s": float(np.median(times)), "max_solve_s": float(max(times)),
            "statuses": sorted(set(statuses)),
            "within_limit": bool(max(times) <= cfg.dispatch.solver.time_limit_s),
        })  # fmt: skip
        log.info("%2d incidents x %2d units: median %.3f s, max %.3f s, %s",
                 n_i, n_u, rows[-1]["median_solve_s"], rows[-1]["max_solve_s"],
                 ",".join(rows[-1]["statuses"]))  # fmt: skip
    live = next(r for r in rows if (r["incidents"], r["units"]) == (10, 91))
    report = {
        "status": "synthetic ETAs; measures solver time against instance size, not route quality",
        "time_limit_s": cfg.dispatch.solver.time_limit_s,
        "live_batch_within_limit": live["within_limit"],
        "live_batch_median_s": live["median_solve_s"],
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
