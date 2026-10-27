"""The simulator's queueing against Erlang-C (phase 7 slice 7).

This is the only place the engine is checked against an answer it did not produce itself. The
background workload reduced to one agency with no incidents IS an M/M/c queue -- Poisson
arrivals, exponential service, c identical servers, one FIFO queue, no abandonment, no
pre-emption -- so `optimization.placement.queue_metrics` solves exactly the same system in closed
form. A disagreement beyond sampling error is a bug, not a modelling difference.

The toy network is enough: a background job never travels, so the graph is only needed to
construct the engine.
"""

from datetime import datetime

import numpy as np
import pytest

from resqcity.common.rng import stream
from resqcity.common.schemas import Agency, Unit, UnitState
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import load_optimization_config
from resqcity.optimization.placement import queue_metrics
from resqcity.sim.background_load import BackgroundLoad
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.engine import Engine
from resqcity.sim.units import Fleet
from tests.conftest import IST, node_id

START = datetime(2026, 2, 3, 8, 0, tzinfo=IST)
CFG = load_sim_config()
OPT = load_optimization_config()
HOME = node_id(0, 0)


def _only(agency: Agency, rho: float) -> SimConfig:
    """Load ``agency`` to ``rho`` and leave the other agencies idle, so one queue runs."""
    raw = CFG.model_dump(mode="json")
    raw["background"]["target_utilisation"]["value"] = {
        a.value: (rho if a == agency else 0.0) for a in Agency
    }
    return SimConfig.model_validate(raw)


def _run(
    network: RoadNetwork, agency: Agency, servers: int, rho: float, days: float, seed: int = 11
) -> tuple[dict, dict]:
    cfg = _only(agency, rho)
    fleet = Fleet.from_units(
        [
            Unit(
                id=f"U{i:03d}",
                agency=agency,
                home_station=f"S{i:03d}",
                node=HOME,
                state=UnitState.IDLE,
                eta_fn_ref="default",
            )
            for i in range(servers)
        ]
    )
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        if name not in cache:
            cache[name] = stream(seed, 0, name)
        return cache[name]

    engine = Engine(network, None, fleet, cfg, OPT, START, rng)
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    horizon = days * 86400.0
    engine.run(until_s=horizon)
    stats = load.stats[agency]
    simulated = {
        "utilisation": engine.fleet.utilisation(agency, 0.0, horizon),
        "p_wait": stats.p_wait,
        "expected_wait_s": stats.mean_wait_s,
        "arrived": stats.arrived,
    }
    analytic = queue_metrics(load.rates[agency], load.service_mean_s(agency), servers)
    return simulated, analytic


def test_utilisation_matches_erlang_c_under_a_light_load(toy_network: RoadNetwork) -> None:
    simulated, analytic = _run(toy_network, Agency.POLICE, 10, 0.3, days=40)
    assert simulated["arrived"] > 500
    assert simulated["utilisation"] == pytest.approx(analytic["utilisation"], abs=0.03)


@pytest.mark.slow
def test_waiting_matches_erlang_c_under_a_heavy_load(toy_network: RoadNetwork) -> None:
    """The case that actually exercises the queue: at rho = 0.85 a fifth of jobs wait, so
    P(wait) and E[W_q] are large enough to disagree visibly if the discipline were wrong."""
    simulated, analytic = _run(toy_network, Agency.POLICE, 10, 0.85, days=60)
    assert analytic["p_wait"] > 0.1, "the analytic case must actually produce a queue"
    assert simulated["utilisation"] == pytest.approx(analytic["utilisation"], abs=0.03)
    assert simulated["p_wait"] == pytest.approx(analytic["p_wait"], abs=0.06)
    assert simulated["expected_wait_s"] == pytest.approx(
        analytic["expected_wait_s"], rel=0.3, abs=15.0
    )


def test_no_work_is_lost_so_utilisation_cannot_fall_short(toy_network: RoadNetwork) -> None:
    """Jobs queue rather than being dropped, so in the long run the servers must be busy exactly
    rho of the time. A shortfall would mean work vanished -- the symptom of the RNG bug that
    made every service time identical (0.859 x the mean, repeated)."""
    for rho in (0.4, 0.7):
        simulated, analytic = _run(toy_network, Agency.POLICE, 8, rho, days=40)
        assert simulated["utilisation"] == pytest.approx(analytic["utilisation"], abs=0.04), rho


def test_an_unstable_queue_is_reported_as_unstable() -> None:
    """rho >= 1 has no steady state; the analytic side must say so rather than return a number."""
    metrics = queue_metrics(arrival_rate_per_s=1.0, service_time_s=10.0, servers=5)
    assert metrics["utilisation"] >= 1.0
    assert metrics["stable"] is False
    assert metrics["expected_wait_s"] == float("inf")
