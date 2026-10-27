"""Hospital choice, transport and handover (phase 7 slice 5).

Transport is most of an ambulance's busy time in a real system, so these tests are mostly about
the ambulance being correctly *unavailable*: an ambulance that teleports home after the scene
would make the fleet look far more available than it is, which is the same class of error as the
missing background workload of D-031 item 3.
"""

from datetime import datetime

import numpy as np
import pytest

from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DispatchDecision,
    DispatchTier,
    EventType,
    GeoPoint,
    Hospital,
    Unit,
    UnitState,
)
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import load_optimization_config
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.engine import Engine
from resqcity.sim.hospitals import CAPABILITY_FOR, HospitalNetwork
from resqcity.sim.units import Fleet
from tests.conftest import IST, node_id

START = datetime(2026, 2, 3, 8, 0, tzinfo=IST)
CFG = load_sim_config()
OPT = load_optimization_config()
NEAR, FAR, SCENE = node_id(0, 0), node_id(4, 4), node_id(2, 2)


def _hospital(
    hid: str, node: int, *, trauma: bool = False, burns: bool = False, beds: int = 2
) -> Hospital:
    return Hospital(
        id=hid, name=hid, location=GeoPoint(lat=18.52, lon=73.85), node=node,
        has_trauma=trauma, has_burns=burns, icu_beds=beds, beds_available=beds,
    )  # fmt: skip


def _unit(uid: str, agency: Agency, node: int) -> Unit:
    return Unit(
        id=uid, agency=agency, home_station=f"S-{uid}", node=node,
        state=UnitState.IDLE, eta_fn_ref="default",
    )  # fmt: skip


def _always_transport(handover_s: float = 600.0) -> SimConfig:
    """p_transport = 1 for every type, so the branch is deterministic in a test."""
    raw = CFG.model_dump(mode="json")
    raw["transport"]["p_transport"]["value"] = dict.fromkeys(
        raw["transport"]["p_transport"]["value"], 1.0
    )
    raw["transport"]["handover_median_s"]["value"] = handover_s
    return SimConfig.model_validate(raw)


def _engine(
    network: RoadNetwork,
    units: list[Unit],
    hospitals: list[Hospital],
    cfg: SimConfig | None = None,
) -> Engine:
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        return cache.setdefault(name, np.random.default_rng(len(name)))

    return Engine(
        network,
        None,
        Fleet.from_units(units),
        cfg or CFG,
        OPT,
        START,
        rng,
        hospitals=HospitalNetwork(hospitals),
    )


def _decision(incident_id: str, unit_id: str) -> DispatchDecision:
    return DispatchDecision(
        id=f"DEC-{incident_id}",
        incident_id=incident_id,
        created_ts=START,
        policy="TEST",
        assignments=[Assignment(unit_id=unit_id, eta_s=0.0, role=AssignmentRole.RESPOND)],
        tier=DispatchTier.FULL,
    )


# -- choosing a hospital (no engine needed) ----------------------------------------------------


def test_an_accident_goes_to_a_trauma_hospital_even_if_it_is_further() -> None:
    """Nearest CAPABLE, not nearest: sending a trauma case to the closest hospital that cannot
    treat it would be a worse model than sending it further."""
    net = HospitalNetwork([
        _hospital("near-general", NEAR),
        _hospital("far-trauma", FAR, trauma=True),
    ])  # fmt: skip
    chosen = net.choose(EventType.ACCIDENT, {"near-general": 60.0, "far-trauma": 900.0})
    assert chosen == "far-trauma"
    assert CAPABILITY_FOR[EventType.ACCIDENT] == "has_trauma"


def test_a_type_with_no_capability_requirement_takes_the_nearest() -> None:
    net = HospitalNetwork([_hospital("a", NEAR), _hospital("b", FAR, trauma=True)])
    assert net.choose(EventType.OTHER, {"a": 300.0, "b": 100.0}) == "b"
    assert net.capability_for(EventType.OTHER) is None


def test_a_full_hospital_is_passed_over_for_one_with_a_free_bay() -> None:
    net = HospitalNetwork([_hospital("busy", NEAR, beds=1), _hospital("free", FAR, beds=1)])
    net.take("busy")
    assert not net.has_room("busy")
    assert net.choose(EventType.OTHER, {"busy": 60.0, "free": 900.0}) == "free"


def test_every_hospital_being_full_still_returns_the_nearest() -> None:
    """Queueing at the nearest beats refusing the patient."""
    net = HospitalNetwork([_hospital("a", NEAR, beds=1), _hospital("b", FAR, beds=1)])
    net.take("a")
    net.take("b")
    assert net.choose(EventType.OTHER, {"a": 60.0, "b": 900.0}) == "a"


def test_no_capable_hospital_returns_none_and_is_recorded() -> None:
    net = HospitalNetwork([_hospital("general", NEAR)])
    assert net.choose(EventType.ACCIDENT, {"general": 60.0}) is None
    net.refuse("I1", "no reachable capable hospital")
    assert net.report()["patients_with_no_capable_hospital"] == 1


def test_an_unreachable_hospital_is_not_chosen() -> None:
    net = HospitalNetwork([_hospital("a", NEAR), _hospital("b", FAR)])
    assert net.choose(EventType.OTHER, {"a": float("inf"), "b": 300.0}) == "b"
    assert net.choose(EventType.OTHER, {"a": float("inf"), "b": float("inf")}) is None


def test_a_hospital_recorded_with_no_beds_can_still_take_one_patient() -> None:
    """A zero in OSM is missing data, not a hospital that receives nobody."""
    net = HospitalNetwork([_hospital("a", NEAR, beds=0)])
    assert net.slots("a") == 1
    assert net.has_room("a")


def test_the_report_says_the_capabilities_are_synthetic() -> None:
    net = HospitalNetwork([_hospital("a", NEAR, trauma=True), _hospital("b", FAR, burns=True)])
    report = net.report()
    assert "SYNTHETIC" in report["status"]
    assert report["with_trauma"] == 1 and report["with_burns"] == 1


# -- transport in the engine -------------------------------------------------------------------


def test_transport_keeps_the_ambulance_busy_through_the_handover(
    toy_network: RoadNetwork,
) -> None:
    cfg = _always_transport(handover_s=600.0)
    units = [_unit("A1", Agency.AMBULANCE, NEAR)]
    engine = _engine(toy_network, units, [_hospital("H", FAR, trauma=True)], cfg)
    state = engine.register(
        "I1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False, event_type=EventType.ACCIDENT
    )
    engine.execute([_decision("I1", "A1")])
    engine.run(until_s=10 * 86400.0)
    assert state.transported
    assert state.hospital_arrival_s is not None
    assert state.handover_s is not None and state.handover_s > state.hospital_arrival_s
    assert engine.hospitals.report()["handovers"] == 1
    # the ambulance is home and free again, having been busy for the whole chain
    assert engine.fleet.units[0].state is UnitState.IDLE
    assert engine.fleet.units[0].node == NEAR
    assert engine.fleet.busy_seconds(Agency.AMBULANCE) > 600.0


def test_a_police_unit_never_transports(toy_network: RoadNetwork) -> None:
    cfg = _always_transport()
    units = [_unit("P1", Agency.POLICE, NEAR)]
    engine = _engine(toy_network, units, [_hospital("H", FAR)], cfg)
    state = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", "P1")])
    engine.run(until_s=10 * 86400.0)
    assert not state.transported
    assert engine.hospitals.report()["handovers"] == 0


def test_a_false_alarm_never_transports(toy_network: RoadNetwork) -> None:
    """There is no patient. Transporting one would invent work and flatter nothing but the clock."""
    cfg = _always_transport()
    units = [_unit("A1", Agency.AMBULANCE, NEAR)]
    engine = _engine(toy_network, units, [_hospital("H", FAR, trauma=True)], cfg)
    state = engine.register(
        "F1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=True, event_type=EventType.ACCIDENT
    )
    engine.execute([_decision("F1", "A1")])
    engine.run(until_s=10 * 86400.0)
    assert not state.transported
    assert engine.hospitals.report()["handovers"] == 0


def test_only_one_ambulance_transports_per_incident(toy_network: RoadNetwork) -> None:
    cfg = _always_transport()
    units = [_unit("A1", Agency.AMBULANCE, NEAR), _unit("A2", Agency.AMBULANCE, FAR)]
    engine = _engine(toy_network, units, [_hospital("H", SCENE, trauma=True)], cfg)
    engine.register(
        "I1", SCENE, {Agency.AMBULANCE: 2}, is_false_alarm=False, event_type=EventType.ACCIDENT
    )
    engine.execute([
        DispatchDecision(
            id="DEC-I1", incident_id="I1", created_ts=START, policy="TEST",
            assignments=[
                Assignment(unit_id="A1", eta_s=0.0, role=AssignmentRole.RESPOND),
                Assignment(unit_id="A2", eta_s=0.0, role=AssignmentRole.RESPOND),
            ],
            tier=DispatchTier.FULL,
        )
    ])  # fmt: skip
    engine.run(until_s=10 * 86400.0)
    assert engine.hospitals.report()["handovers"] == 1


def test_an_incident_with_no_capable_hospital_releases_the_ambulance(
    toy_network: RoadNetwork,
) -> None:
    """log_and_release: the case is recorded and the unit goes home, rather than the ambulance
    disappearing from the fleet."""
    cfg = _always_transport()
    units = [_unit("A1", Agency.AMBULANCE, NEAR)]
    engine = _engine(toy_network, units, [_hospital("general-only", FAR)], cfg)
    engine.register(
        "I1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False, event_type=EventType.ACCIDENT
    )
    engine.execute([_decision("I1", "A1")])
    engine.run(until_s=10 * 86400.0)
    assert engine.hospitals.report()["patients_with_no_capable_hospital"] == 1
    assert engine.fleet.units[0].state is UnitState.IDLE
    assert engine.fleet.units[0].node == NEAR


def test_handover_capacity_makes_the_second_ambulance_wait(toy_network: RoadNetwork) -> None:
    """One bay, two patients: the second handover must start after the first finishes, or a
    mass-casualty scenario would absorb unlimited patients at one address."""
    cfg = _always_transport(handover_s=3600.0)
    units = [_unit("A1", Agency.AMBULANCE, NEAR), _unit("A2", Agency.AMBULANCE, NEAR)]
    engine = _engine(toy_network, units, [_hospital("H", SCENE, trauma=True, beds=1)], cfg)
    for iid, uid in (("I1", "A1"), ("I2", "A2")):
        engine.register(
            iid, SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False,
            event_type=EventType.ACCIDENT,
        )  # fmt: skip
        engine.execute([_decision(iid, uid)])
    engine.run(until_s=20 * 86400.0)
    assert engine.hospitals.report()["handovers"] == 2
    records = sorted(engine.hospitals.handovers, key=lambda r: r.arrived_s)
    # the first found the bay free; the second queued behind it
    assert records[0].waited_s == pytest.approx(0.0, abs=31.0)  # one poll interval
    assert records[1].waited_s > 300.0, [r.waited_s for r in records]
    # and the queue wait is separate from the handover itself
    assert records[1].total_s > records[1].waited_s
    assert engine.hospitals.busy["H"] == 0  # the bay was released both times
