"""The out-of-scope workload (phase 7 slice 4; D-031 item 3, L30).

The point of this process is that the simulated city stops being capacity-free. The test that
matters is therefore the one that checks the realised utilisation actually lands on the derived
target -- if it does not, every capacity conclusion in phase 10 is wrong.
"""

from datetime import datetime

import numpy as np
import pytest

from resqcity.common.schemas import Agency, Unit, UnitState
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import load_optimization_config
from resqcity.sim.background_load import BackgroundLoad
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.engine import Engine
from resqcity.sim.units import BUSY_STATES, Fleet
from tests.conftest import IST, node_id

START = datetime(2026, 2, 3, 8, 0, tzinfo=IST)
CFG = load_sim_config()
OPT = load_optimization_config()
HOME = node_id(0, 0)


def _units(counts: dict[Agency, int]) -> list[Unit]:
    out = []
    for agency, n in counts.items():
        for i in range(n):
            uid = f"{agency.value[:1].upper()}{i:02d}"
            out.append(
                Unit(
                    id=uid,
                    agency=agency,
                    home_station=f"S-{uid}",
                    node=HOME,
                    state=UnitState.IDLE,
                    eta_fn_ref="default",
                )
            )
    return out


def _engine(network: RoadNetwork, counts: dict[Agency, int], seed: int = 7) -> Engine:
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        # one PERSISTENT stream per name, as the real runner does: handing back a fresh
        # generator each call would replay the same draw for ever
        return cache.setdefault(name, np.random.default_rng([seed, *map(ord, name[:16])]))

    return Engine(network, None, Fleet.from_units(_units(counts)), CFG, OPT, START, rng)


def _with_target(value: float, enabled: bool = True) -> SimConfig:
    raw = CFG.model_dump(mode="json")
    raw["background"]["enabled"] = enabled
    raw["background"]["target_utilisation"]["value"] = dict.fromkeys(
        raw["background"]["target_utilisation"]["value"], value
    )
    return SimConfig.model_validate(raw)


# -- sizing ---------------------------------------------------------------------------------


def test_the_arrival_rate_is_derived_from_the_target_not_configured(
    toy_network: RoadNetwork,
) -> None:
    """lambda = rho * c / S, so the offered load is exactly rho * c erlangs."""
    engine = _engine(toy_network, {Agency.POLICE: 39})
    load = BackgroundLoad(engine, CFG, OPT)
    rho = CFG.background.target_utilisation.value[Agency.POLICE]
    assert load.offered_load(Agency.POLICE) == pytest.approx(rho * 39)
    # and that is the Dial-112 volume D-031 item 3 quotes, which is the whole point of the number
    assert load.rates[Agency.POLICE] * 86400 == pytest.approx(771.0, rel=1e-3)


def test_the_service_time_comes_from_the_erlang_config_not_a_copy(
    toy_network: RoadNetwork,
) -> None:
    """A second copy of this number would let the simulator and the phase-6 cross-check drift."""
    engine = _engine(toy_network, {Agency.FIRE: 22})
    load = BackgroundLoad(engine, CFG, OPT)
    for agency in Agency:
        assert load.service_mean_s(agency) == OPT.erlang.service_time_s.value[agency]


def test_no_units_means_no_jobs(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, {Agency.POLICE: 0})
    load = BackgroundLoad(engine, CFG, OPT)
    assert load.rates[Agency.POLICE] == 0.0


# -- the measurement that the phase turns on ------------------------------------------------


@pytest.mark.parametrize("target", [0.2, 0.5])
def test_realised_utilisation_matches_the_target(toy_network: RoadNetwork, target: float) -> None:
    """Simulate long enough for the M/M/c steady state and compare.

    With 10 servers and a 1800 s mean service time, 60 simulated days is a few thousand jobs,
    so the sampling error on utilisation is well under the 0.03 tolerance.
    """
    cfg = _with_target(target)
    engine = _engine(toy_network, {Agency.POLICE: 10})
    engine.cfg = cfg
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    horizon = 60 * 86400.0
    engine.run(until_s=horizon)
    realised = engine.fleet.utilisation(Agency.POLICE, 0.0, horizon)
    assert realised == pytest.approx(target, abs=0.03)
    assert load.stats[Agency.POLICE].arrived > 1000


def test_switching_the_target_off_reproduces_the_capacity_free_city(
    toy_network: RoadNetwork,
) -> None:
    """The phase-10 ablation must be exactly the old model: no jobs, no busy time."""
    cfg = _with_target(0.0, enabled=False)
    engine = _engine(toy_network, {Agency.POLICE: 5})
    engine.cfg = cfg
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    engine.run(until_s=7 * 86400.0)
    assert load.stats[Agency.POLICE].arrived == 0
    assert engine.fleet.busy_seconds(Agency.POLICE) == pytest.approx(0.0)
    assert engine.on_unit_free == []  # the hook was never even registered


# -- queueing, which slice 7 validates against Erlang-C --------------------------------------


def test_a_job_with_no_free_unit_waits_and_is_started_by_the_next_one_freed(
    toy_network: RoadNetwork,
) -> None:
    """If the queue were dropped instead, waiting time would be zero by construction and there
    would be nothing for the M/M/c comparison to check."""
    cfg = _with_target(0.9)  # heavy load on a single server, so a queue must form
    engine = _engine(toy_network, {Agency.POLICE: 1})
    engine.cfg = cfg
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    engine.run(until_s=30 * 86400.0)
    stats = load.stats[Agency.POLICE]
    assert stats.arrived > stats.started  # a backlog exists
    assert stats.p_wait > 0.5
    assert stats.mean_wait_s > 0.0


def test_a_unit_on_a_background_job_is_busy_and_cannot_be_dispatched(
    toy_network: RoadNetwork,
) -> None:
    engine = _engine(toy_network, {Agency.POLICE: 1})
    fleet = engine.fleet
    fleet.set_state(0, UnitState.ON_OTHER_TASK, 0.0)
    assert UnitState.ON_OTHER_TASK in BUSY_STATES
    assert fleet.units[0].busy
    assert fleet.claimable(Agency.POLICE) == []


def test_a_background_job_is_never_pre_empted(toy_network: RoadNetwork) -> None:
    """configs/sim.yaml says preempt: false, so the only way out is finishing. The transition
    table enforces it rather than leaving it to the caller."""
    engine = _engine(toy_network, {Agency.POLICE: 1})
    engine.fleet.set_state(0, UnitState.ON_OTHER_TASK, 0.0)
    with pytest.raises(ValueError, match="not a legal transition"):
        engine.fleet.set_state(0, UnitState.EN_ROUTE, 10.0)


def test_a_pre_alerted_unit_is_not_pulled_onto_background_work(
    toy_network: RoadNetwork,
) -> None:
    """READY is claimable for an incident but was held back for a specific candidate; letting
    out-of-scope work take it would make confidence tiering look worse than it is."""
    cfg = _with_target(0.5)
    engine = _engine(toy_network, {Agency.POLICE: 1})
    engine.cfg = cfg
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    engine.fleet.set_state(0, UnitState.READY, 0.0, incident_id="I1")
    engine.run(until_s=7 * 86400.0)
    assert load.stats[Agency.POLICE].arrived > 0  # jobs did arrive
    assert load.stats[Agency.POLICE].started == 0  # but none took the pre-alerted unit
    assert engine.fleet.units[0].state is UnitState.READY


def test_the_report_records_what_was_asked_for_and_what_happened(
    toy_network: RoadNetwork,
) -> None:
    cfg = _with_target(0.3)
    engine = _engine(toy_network, {Agency.POLICE: 4})
    engine.cfg = cfg
    load = BackgroundLoad(engine, cfg, OPT)
    load.start()
    engine.run(until_s=14 * 86400.0)
    report = load.report()
    police = report["by_agency"]["police"]
    assert report["enabled"] is True
    assert police["target_utilisation"] == pytest.approx(0.3)
    assert police["offered_load_erlangs"] == pytest.approx(0.3 * 4)
    assert police["jobs_started"] > 0
    assert "erlang.service_time_s" in report["service_time_source"]
