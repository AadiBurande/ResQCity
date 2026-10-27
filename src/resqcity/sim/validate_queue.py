"""Validate the simulator's queueing against Erlang-C (phase 7 slice 7).

    uv run python -m resqcity.sim.validate_queue

This is the one check that the engine's queueing is **right** rather than merely plausible.
Everything else in phase 7 is verified against its own inputs; here the simulator is stripped to
a system with a known closed-form answer and the two are compared.

The reduction is exact, not approximate. The background workload of
``sim/background_load.py`` already *is* an M/M/c queue: Poisson arrivals per agency, exponential
service, ``c`` identical servers, one FIFO queue, no abandonment, no pre-emption. Running it on
its own -- with no incidents, so no travel and no dispatch -- leaves precisely the system
``optimization.placement.queue_metrics`` solves analytically. So any disagreement beyond
sampling error is a bug in the simulator (or in the Erlang code), not a modelling difference.

Three quantities are compared:

* **utilisation** rho = lambda S / c, which the simulator must reproduce because no work is lost
  (jobs queue rather than being dropped);
* **P(wait)**, the share of jobs that find every server busy -- Erlang-C's C(c, a);
* **E[W_q]**, the mean time a job spends waiting.

What this does NOT validate is the incident response: travel, on-scene time, the second wave and
the policies have no closed form. It validates the engine's clock, its queue discipline and its
busy-time accounting, which everything else is built on.

Output: ``results/sim/queue_validation.json``. Tolerances come from
``configs/sim.yaml`` -> ``validation.erlang_check``.
"""

import argparse
import json
import logging
import time
from datetime import datetime

import numpy as np

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.rng import stream
from resqcity.common.schemas import Agency, Unit, UnitState
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import load_optimization_config
from resqcity.optimization.placement import queue_metrics
from resqcity.sim.background_load import BackgroundLoad
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.engine import Engine
from resqcity.sim.units import Fleet

OUT = PROJECT_ROOT / "results" / "sim" / "queue_validation.json"
log = logging.getLogger(__name__)


def _fleet(agency: Agency, servers: int, node: int) -> Fleet:
    return Fleet.from_units([
        Unit(
            id=f"{agency.value[:1].upper()}{i:03d}",
            agency=agency,
            home_station=f"S{i:03d}",
            node=node,
            state=UnitState.IDLE,
            eta_fn_ref="default",
        )
        for i in range(servers)
    ])  # fmt: skip


def _with_rate(cfg: SimConfig, agency: Agency, rho: float) -> SimConfig:
    """Target utilisation ``rho`` for ``agency`` and zero for the others, so only one queue runs."""
    raw = cfg.model_dump(mode="json")
    raw["background"]["target_utilisation"]["value"] = {
        a.value: (rho if a == agency else 0.0) for a in Agency
    }
    return SimConfig.model_validate(raw)


def run_case(
    network: RoadNetwork, agency: Agency, servers: int, rho: float, horizon_days: int, seed: int
) -> dict[str, object]:
    """One M/M/c case: simulate it, solve it, and report both."""
    sim_cfg, opt_cfg = load_sim_config(), load_optimization_config()
    cfg = _with_rate(sim_cfg, agency, rho)
    node = int(network.nodes["node_id"].iloc[0])
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        # one persistent stream per name: a fresh generator per call would replay its first draw
        if name not in cache:
            cache[name] = stream(seed, 0, name)
        return cache[name]

    engine = Engine(
        network, None, _fleet(agency, servers, node), cfg, opt_cfg, load_scenario_start(), rng
    )
    load = BackgroundLoad(engine, cfg, opt_cfg)
    load.start()
    horizon_s = horizon_days * 86400.0
    started = time.perf_counter()
    engine.run(until_s=horizon_s)
    stats = load.stats[agency]
    simulated = {
        "utilisation": engine.fleet.utilisation(agency, 0.0, horizon_s),
        "p_wait": stats.p_wait,
        "expected_wait_s": stats.mean_wait_s,
        "jobs_arrived": stats.arrived,
        "jobs_started": stats.started,
        "queued_at_end": len(load.queues[agency]),
    }
    analytic = queue_metrics(load.rates[agency], load.service_mean_s(agency), servers)
    return {
        "agency": agency.value,
        "servers": servers,
        "target_utilisation": rho,
        "horizon_days": horizon_days,
        "arrival_per_day": load.rates[agency] * 86400.0,
        "service_mean_s": load.service_mean_s(agency),
        "simulated": simulated,
        "analytic": analytic,
        "difference": {
            "utilisation": simulated["utilisation"] - analytic["utilisation"],
            "p_wait": simulated["p_wait"] - analytic["p_wait"],
            "expected_wait_s": simulated["expected_wait_s"] - analytic["expected_wait_s"],
        },
        "wall_s": time.perf_counter() - started,
    }


def load_scenario_start() -> datetime:
    """Any aware start instant will do: nothing in this reduction depends on the calendar."""
    from resqcity.geo.scenario import load_scenario

    return load_scenario("normal").start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=None, help="default: validation.horizon_days")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    sim_cfg = load_sim_config()
    check = sim_cfg.validation.erlang_check
    days = args.days or check.horizon_days
    seed = base.random.seed_base

    cases = []
    # The configured case, plus a light and a heavy load. One case could agree by luck; the
    # interesting one is the heavy load, where the queue is actually busy and P(wait) is large.
    for rho in (0.2, 0.5, 0.85):
        cases.append(run_case(network, check.agency, check.units, rho, days, seed))
    worst = {
        key: max(abs(c["difference"][key]) for c in cases)
        for key in ("utilisation", "p_wait", "expected_wait_s")
    }
    passed = (
        worst["utilisation"] <= check.tolerance_utilisation
        and worst["expected_wait_s"] <= check.tolerance_wait_s
    )
    report = {
        "status": "SYNTHETIC. The simulator's background workload reduced to M/M/c and compared "
        "with the Erlang-C formulas of optimization/placement.py. This validates the engine's "
        "clock, queue discipline and busy-time accounting; it does NOT validate travel, "
        "on-scene time or the policies, which have no closed form.",
        "agency": check.agency.value,
        "servers": check.units,
        "tolerances": {
            "utilisation": check.tolerance_utilisation,
            "expected_wait_s": check.tolerance_wait_s,
        },
        "worst_absolute_difference": worst,
        "passed": bool(passed),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8", newline="\n")
    write_meta(
        OUT,
        configs=[CONFIG_DIR / "sim.yaml", CONFIG_DIR / "optimization.yaml"],
        synthetic=True,
        seeds={"seed_base": seed},
    )
    for case in cases:
        log.info(
            "rho %.2f: simulated util %.4f / P(wait) %.4f / Wq %.1f s  vs  analytic %.4f / "
            "%.4f / %.1f s",
            case["target_utilisation"],
            case["simulated"]["utilisation"],
            case["simulated"]["p_wait"],
            case["simulated"]["expected_wait_s"],
            case["analytic"]["utilisation"],
            case["analytic"]["p_wait"],
            case["analytic"]["expected_wait_s"],
        )
    log.info("worst difference %s -> %s", worst, "PASS" if passed else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
