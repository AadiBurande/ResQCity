"""The simulator's clock, unit state machine and response lifecycle (phase 7 slice 1).

Travel times are not hard-coded here. They come from the same router the dispatch models use, and
the assertions compare the engine's arrival times against it: what is under test is the engine's
bookkeeping -- who was claimed, when the requirement was met, what counted as busy -- not the
road graph, which `tests/geo/` already covers.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DispatchDecision,
    DispatchTier,
    Unit,
    UnitState,
)
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import load_optimization_config
from resqcity.optimization.routing import Router
from resqcity.sim.config import load_sim_config
from resqcity.sim.engine import Engine
from resqcity.sim.units import BUSY_STATES, Fleet
from tests.conftest import IST, node_id

START = datetime(2026, 2, 3, 8, 0, tzinfo=IST)
CFG = load_sim_config()
OPT = load_optimization_config()
NEAR, FAR, SCENE = node_id(0, 0), node_id(4, 4), node_id(2, 2)
ISOLATED = 999


def _unit(uid: str, agency: Agency, node: int) -> Unit:
    return Unit(
        id=uid, agency=agency, home_station=f"S-{uid}", node=node,
        state=UnitState.IDLE, eta_fn_ref="default",
    )  # fmt: skip


def _engine(network: RoadNetwork, units: list[Unit]) -> Engine:
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        return cache.setdefault(name, np.random.default_rng(len(name)))

    return Engine(network, None, Fleet.from_units(units), CFG, OPT, START, rng)


def _travel(network: RoadNetwork, a: int, b: int) -> float:
    """The router's own answer, so the test does not re-implement travel."""
    table = Router(network, OPT, None).eta_table([a], [b], START)
    return float(table.seconds[0, 0])


def _decision(incident_id: str, pairs: list[tuple[str, AssignmentRole]]) -> DispatchDecision:
    return DispatchDecision(
        id=f"DEC-{incident_id}",
        incident_id=incident_id,
        created_ts=START,
        policy="TEST",
        assignments=[Assignment(unit_id=u, eta_s=0.0, role=r) for u, r in pairs],
        tier=DispatchTier.FULL,
    )


# -- the clock --------------------------------------------------------------------------------


def test_the_clock_maps_simulated_seconds_to_wall_clock(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("A1", Agency.AMBULANCE, NEAR)])
    assert engine.now == START and engine.now_s == 0.0
    engine.env.run(until=3600.0)
    assert engine.now_s == 3600.0
    assert engine.now == START + timedelta(hours=1)
    assert engine.seconds_of(engine.now) == pytest.approx(3600.0)


# -- the state machine (no graph needed) ------------------------------------------------------


def test_an_illegal_transition_raises_rather_than_corrupting_the_log() -> None:
    fleet = Fleet.from_units([_unit("A1", Agency.AMBULANCE, NEAR)])
    fleet.set_state(0, UnitState.EN_ROUTE, 0.0)
    with pytest.raises(ValueError, match="not a legal transition"):
        fleet.set_state(0, UnitState.IDLE, 10.0)  # must go through RETURNING


def test_time_cannot_run_backwards() -> None:
    fleet = Fleet.from_units([_unit("A1", Agency.AMBULANCE, NEAR)])
    fleet.set_state(0, UnitState.EN_ROUTE, 100.0)
    with pytest.raises(ValueError, match="time went backwards"):
        fleet.set_state(0, UnitState.ON_SCENE, 50.0)


def test_busy_periods_accumulate_and_the_warm_up_can_be_excluded() -> None:
    fleet = Fleet.from_units([_unit("P1", Agency.POLICE, NEAR)])
    fleet.set_state(0, UnitState.EN_ROUTE, 0.0)
    fleet.set_state(0, UnitState.ON_SCENE, 100.0)
    fleet.set_state(0, UnitState.RETURNING, 400.0)
    fleet.set_state(0, UnitState.IDLE, 500.0)
    assert fleet.busy_seconds(Agency.POLICE) == pytest.approx(500.0)
    assert fleet.busy_seconds(Agency.POLICE, after_s=200.0) == pytest.approx(300.0)
    assert fleet.utilisation(Agency.POLICE, 0.0, 1000.0) == pytest.approx(0.5)


def test_a_pre_alerted_unit_is_not_counted_as_busy_but_is_still_claimable() -> None:
    """A documented modelling choice (src/resqcity/sim/units.py): the crew is in the vehicle, no
    fuel is burned, and a confirmed incident should still be able to take the vehicle."""
    fleet = Fleet.from_units([_unit("A1", Agency.AMBULANCE, NEAR)])
    fleet.set_state(0, UnitState.READY, 0.0, incident_id="I1")
    assert UnitState.READY not in BUSY_STATES
    assert not fleet.units[0].busy
    assert fleet.claimable(Agency.AMBULANCE) == [0]
    fleet.set_state(0, UnitState.EN_ROUTE, 60.0)
    assert fleet.busy_seconds(Agency.AMBULANCE) == pytest.approx(0.0)


def test_closing_open_periods_is_idempotent() -> None:
    fleet = Fleet.from_units([_unit("P1", Agency.POLICE, NEAR)])
    fleet.set_state(0, UnitState.EN_ROUTE, 0.0)
    fleet.close_open_periods(300.0)
    fleet.close_open_periods(300.0)
    assert fleet.busy_seconds(Agency.POLICE) == pytest.approx(300.0)


# -- the response lifecycle -------------------------------------------------------------------


def test_one_incident_end_to_end(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("A1", Agency.AMBULANCE, NEAR)])
    state = engine.register("I1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("A1", AssignmentRole.RESPOND)])])
    engine.run(until_s=86400.0)
    expected = _travel(toy_network, NEAR, SCENE)
    assert len(state.arrivals) == 1
    assert state.arrivals[0].travel_s == pytest.approx(expected)
    assert state.arrivals[0].arrived_s == pytest.approx(expected)
    assert state.satisfied()
    assert state.last_required_arrival_s() == pytest.approx(expected)
    # and the unit went home rather than staying on scene for ever
    assert engine.fleet.units[0].state is UnitState.IDLE
    assert engine.fleet.units[0].node == NEAR


def test_the_last_required_arrival_waits_for_the_slowest_required_agency(
    toy_network: RoadNetwork,
) -> None:
    units = [_unit("A1", Agency.AMBULANCE, NEAR), _unit("P1", Agency.POLICE, FAR)]
    engine = _engine(toy_network, units)
    state = engine.register(
        "I1", SCENE, {Agency.AMBULANCE: 1, Agency.POLICE: 1}, is_false_alarm=False
    )
    engine.execute(
        [_decision("I1", [("A1", AssignmentRole.RESPOND), ("P1", AssignmentRole.RESPOND)])]
    )
    engine.run(until_s=86400.0)
    slowest = max(_travel(toy_network, NEAR, SCENE), _travel(toy_network, FAR, SCENE))
    assert state.last_required_arrival_s() == pytest.approx(slowest)


def test_an_unmet_requirement_has_no_last_arrival_time(toy_network: RoadNetwork) -> None:
    """A half-served incident must not report a completion -- the same rule the dispatch model's
    C6 constraint enforces (docs/models/dispatch.md)."""
    engine = _engine(toy_network, [_unit("A1", Agency.AMBULANCE, NEAR)])
    state = engine.register(
        "I1", SCENE, {Agency.AMBULANCE: 1, Agency.FIRE: 1}, is_false_alarm=False
    )
    engine.execute([_decision("I1", [("A1", AssignmentRole.RESPOND)])])
    engine.run(until_s=86400.0)
    assert len(state.arrivals) == 1
    assert not state.satisfied()
    assert state.last_required_arrival_s() is None


def test_units_beyond_the_requirement_do_not_set_the_clock(toy_network: RoadNetwork) -> None:
    """Two police sent to an incident needing one: the completion is the FIRST arrival."""
    units = [_unit("P1", Agency.POLICE, NEAR), _unit("P2", Agency.POLICE, FAR)]
    engine = _engine(toy_network, units)
    state = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute(
        [_decision("I1", [("P1", AssignmentRole.RESPOND), ("P2", AssignmentRole.RESPOND)])]
    )
    engine.run(until_s=86400.0)
    assert len(state.arrivals) == 2
    first = min(_travel(toy_network, NEAR, SCENE), _travel(toy_network, FAR, SCENE))
    assert state.last_required_arrival_s() == pytest.approx(first)


def test_a_pre_alerted_unit_never_travels(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("A1", Agency.AMBULANCE, NEAR)])
    state = engine.register("I1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("A1", AssignmentRole.PRE_ALERT)])])
    engine.run(until_s=86400.0)
    assert state.arrivals == []
    assert engine.fleet.units[0].state is UnitState.READY
    assert engine.fleet.units[0].node == NEAR


def test_a_unit_cannot_be_sent_to_two_incidents_at_once(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    first = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    second = engine.register("I2", FAR, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([
        _decision("I1", [("P1", AssignmentRole.RESPOND)]),
        _decision("I2", [("P1", AssignmentRole.RESPOND)]),
    ])  # fmt: skip
    engine.run(until_s=86400.0)
    assert len(first.arrivals) == 1
    assert second.arrivals == []  # the claim was refused, not double-booked


def test_a_false_alarm_still_costs_a_trip(toy_network: RoadNetwork) -> None:
    """The detector's precision cost has to land in the metrics, not in prose (D-029)."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    engine.register("F1", FAR, {Agency.POLICE: 1}, is_false_alarm=True)
    engine.execute([_decision("F1", [("P1", AssignmentRole.VERIFY)])])
    engine.run(until_s=86400.0)
    out_and_back = 2 * _travel(toy_network, NEAR, FAR)
    assert engine.false_alarm_trips == 1
    assert engine.fleet.busy_seconds(Agency.POLICE) > out_and_back
    assert engine.fleet.units[0].state is UnitState.IDLE


def test_an_unreachable_incident_consumes_no_unit_time(toy_network: RoadNetwork) -> None:
    """Node 999 is the one-way spur the network drops when it keeps the largest strongly
    connected component, so no unit can reach it. The unit must stay available, not be left
    EN_ROUTE for ever, and one such incident must not end the replication."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    state = engine.register("I1", ISOLATED, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("P1", AssignmentRole.RESPOND)])])
    engine.run(until_s=86400.0)
    assert not np.isfinite(engine.travel_s(NEAR, ISOLATED))
    assert state.arrivals == []
    assert engine.fleet.units[0].state is UnitState.IDLE
    assert engine.fleet.busy_seconds(Agency.POLICE) == pytest.approx(0.0)
    assert "P1" not in state.dispatched_units  # free to be sent elsewhere


def test_an_unknown_unit_is_ignored_not_crashed_on(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    state = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("NOPE", AssignmentRole.RESPOND)])])
    engine.run(until_s=3600.0)
    assert state.arrivals == []


def test_the_eta_table_is_built_from_current_positions(toy_network: RoadNetwork) -> None:
    units = [_unit("P1", Agency.POLICE, NEAR), _unit("P2", Agency.POLICE, FAR)]
    engine = _engine(toy_network, units)
    table = engine.eta_table([SCENE])
    assert table.seconds.shape == (2, 1)
    assert table.seconds[0, 0] == pytest.approx(_travel(toy_network, NEAR, SCENE))
    assert table.seconds[1, 0] == pytest.approx(_travel(toy_network, FAR, SCENE))


def test_a_handoff_delay_postpones_departure_and_is_charged_as_busy_time(
    toy_network: RoadNetwork,
) -> None:
    """B0's sequential hand-off. The vehicle is committed from the decision, so the wait counts
    as busy time, and the record keeps travel and delay separable: the delay is
    (arrived - dispatched) - travel."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    state = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute(
        [_decision("I1", [("P1", AssignmentRole.RESPOND)])],
        delays={("I1", "P1"): 240.0},
    )
    engine.run(until_s=86400.0)
    travel = _travel(toy_network, NEAR, SCENE)
    arrival = state.arrivals[0]
    assert arrival.travel_s == pytest.approx(travel)
    assert arrival.arrived_s == pytest.approx(240.0 + travel)
    assert arrival.arrived_s - arrival.dispatched_s - arrival.travel_s == pytest.approx(240.0)
    assert engine.fleet.busy_seconds(Agency.POLICE) > 240.0


def test_a_delayed_unit_is_still_claimed_immediately(toy_network: RoadNetwork) -> None:
    """Otherwise another incident could take it during the hand-off and sequential dispatch
    would look cheaper than it is."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    first = engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute(
        [_decision("I1", [("P1", AssignmentRole.RESPOND)])], delays={("I1", "P1"): 600.0}
    )
    assert engine.fleet.units[0].state is UnitState.EN_ROUTE
    second = engine.register("I2", NEAR, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I2", [("P1", AssignmentRole.RESPOND)])])
    engine.run(until_s=86400.0)
    assert len(first.arrivals) == 1
    assert second.arrivals == []


# -- topping up a short response ---------------------------------------------------------------


def test_outstanding_counts_units_already_on_their_way(toy_network: RoadNetwork) -> None:
    """A re-dispatch must top the response up, not send the whole requirement again."""
    units = [_unit("P1", Agency.POLICE, NEAR), _unit("P2", Agency.POLICE, FAR)]
    engine = _engine(toy_network, units)
    state = engine.register("I1", SCENE, {Agency.POLICE: 2}, is_false_alarm=False)
    assert state.outstanding() == {Agency.POLICE: 2}
    engine.execute([_decision("I1", [("P1", AssignmentRole.RESPOND)])])
    assert state.outstanding() == {Agency.POLICE: 1}  # P1 is committed, one still needed
    assert state.unfinished()
    engine.execute([_decision("I1", [("P2", AssignmentRole.RESPOND)])])
    assert state.outstanding() == {Agency.POLICE: 0}
    assert not state.unfinished()
    engine.run(until_s=86400.0)
    assert state.satisfied()
    assert state.last_required_arrival_s() is not None


def test_a_response_that_could_not_be_filled_stays_unfinished(toy_network: RoadNetwork) -> None:
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    state = engine.register("I1", SCENE, {Agency.POLICE: 1, Agency.FIRE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("P1", AssignmentRole.RESPOND)])])
    engine.run(until_s=86400.0)
    assert state.outstanding() == {Agency.POLICE: 0, Agency.FIRE: 1}
    assert state.unfinished()  # so the runner will keep offering it to the dispatcher


def test_a_later_estimate_can_raise_the_requirement_but_never_lower_it(
    toy_network: RoadNetwork,
) -> None:
    """The second wave raises what an incident needs. A weaker later estimate must not shrink a
    requirement the first unit on scene already established."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    state = engine.register("I1", SCENE, {Agency.POLICE: 1, Agency.FIRE: 2}, is_false_alarm=False)
    assert state.required[Agency.FIRE] == 2
    state = engine.register("I1", SCENE, {Agency.POLICE: 1, Agency.FIRE: 0}, is_false_alarm=False)
    assert state.required[Agency.FIRE] == 2  # not lowered


def test_a_refused_ask_is_recorded_and_a_withheld_one_is_not(toy_network: RoadNetwork) -> None:
    """The distinction the retry logic rests on: "wanted a unit and could not get one" must be
    recorded, while "chose to send fewer than the requirement" must not. A confidence tier does
    the latter deliberately, and retrying it would fight the policy."""
    engine = _engine(toy_network, [_unit("P1", Agency.POLICE, NEAR)])
    # asked for two police, only one exists and it is taken by the first ask
    state = engine.register("I1", SCENE, {Agency.POLICE: 2}, is_false_alarm=False)
    engine.execute([
        _decision("I1", [("P1", AssignmentRole.RESPOND)]),
    ])  # fmt: skip
    assert state.was_refused() == {}  # nothing was refused; the policy only asked for one
    other = engine.register("I2", FAR, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I2", [("P1", AssignmentRole.RESPOND)])])
    assert other.was_refused() == {Agency.POLICE: 1}  # P1 was busy, so this ask was refused


def test_a_refusal_clears_once_the_ask_is_met(toy_network: RoadNetwork) -> None:
    """Otherwise one unlucky batch would keep the incident on the retry list for ever."""
    units = [_unit("P1", Agency.POLICE, NEAR), _unit("P2", Agency.POLICE, FAR)]
    engine = _engine(toy_network, units)
    engine.register("I1", SCENE, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("P1", AssignmentRole.RESPOND)])])
    second = engine.register("I2", FAR, {Agency.POLICE: 1}, is_false_alarm=False)
    engine.execute([_decision("I2", [("P1", AssignmentRole.RESPOND)])])  # refused, P1 is busy
    assert second.was_refused() == {Agency.POLICE: 1}
    engine.execute([_decision("I2", [("P2", AssignmentRole.RESPOND)])])  # the retry succeeds
    assert second.was_refused() == {}
    assert not second.unfinished()


def test_pre_alerting_an_already_ready_unit_is_a_no_op(toy_network: RoadNetwork) -> None:
    """A VERIFY candidate whose posterior keeps moving is re-offered to the policy, so the same
    unit gets pre-alerted again. The crew is already in the vehicle; re-alerting must not be
    treated as a state change (it raised ready -> ready before this was fixed)."""
    engine = _engine(toy_network, [_unit("A1", Agency.AMBULANCE, NEAR)])
    engine.register("I1", SCENE, {Agency.AMBULANCE: 1}, is_false_alarm=False)
    engine.execute([_decision("I1", [("A1", AssignmentRole.PRE_ALERT)])])
    assert engine.fleet.units[0].state is UnitState.READY
    first_since = engine.fleet.units[0].since_s
    engine.execute([_decision("I1", [("A1", AssignmentRole.PRE_ALERT)])])
    assert engine.fleet.units[0].state is UnitState.READY
    assert engine.fleet.units[0].since_s == first_since  # the clock was not restarted either
