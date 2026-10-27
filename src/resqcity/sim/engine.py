"""The SimPy engine: the clock, unit movement and the response lifecycle (phase 7, D-032).

What this module does and does not decide:

* It **owns the clock and the fleet**. Time is simulated seconds since the scenario start; the
  engine converts to and from wall-clock datetimes for the schemas, which are timezone-aware.
* It **does not decide who goes where**. A policy returns ``DispatchDecision`` objects and the
  engine executes them, so the same lifecycle runs under B0 and FULL and no policy can quietly
  get a different travel model (slice 3).
* It **does not generate evidence**. Incidents, camera events and calls are produced by
  ``sim/{generator,detection,calls}.py`` before the run and injected at their timestamps, exactly
  as ``sim/fusion_check.py`` already replays them. Pre-generating is what makes common random
  numbers across policies exact rather than approximate: B0 and FULL see the same week.

Travel uses ``RoadNetwork.shortest_times`` through ``optimization.routing``, so the simulator and
the dispatch models cannot disagree about how long a journey takes.
"""

import logging
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import simpy

from resqcity.common.schemas import (
    Agency,
    AssignmentRole,
    DispatchDecision,
    Unit,
    UnitState,
)
from resqcity.geo.network import RoadNetwork
from resqcity.geo.scenario import ScenarioState
from resqcity.optimization.config import OptimizationConfig
from resqcity.optimization.routing import EtaTable, Router
from resqcity.sim.config import SimConfig
from resqcity.sim.units import Fleet

log = logging.getLogger(__name__)

#: Travel time used when the road graph says a journey is impossible (flooded or disconnected).
#: The unit is simply not sent; this constant only guards arithmetic.
UNREACHABLE_S = float("inf")


@dataclass
class ArrivalRecord:
    """One unit reaching one incident -- the raw material of the primary outcome."""

    incident_id: str
    unit_id: str
    agency: Agency
    role: AssignmentRole
    dispatched_s: float
    arrived_s: float
    travel_s: float


@dataclass
class ResponseState:
    """What the engine knows about one incident's response as it unfolds."""

    incident_id: str
    node: int
    created_s: float
    required: dict[Agency, int]
    arrivals: list[ArrivalRecord] = field(default_factory=list)
    dispatched_units: set[str] = field(default_factory=set)
    second_wave_s: float | None = None
    is_false_alarm: bool = False

    def arrived_counts(self) -> dict[Agency, int]:
        counts = dict.fromkeys(Agency, 0)
        for a in self.arrivals:
            if a.role in (AssignmentRole.RESPOND, AssignmentRole.VERIFY):
                counts[a.agency] += 1
        return counts

    def satisfied(self) -> bool:
        got = self.arrived_counts()
        return all(got[a] >= n for a, n in self.required.items() if n > 0)

    def last_required_arrival_s(self) -> float | None:
        """When the last REQUIRED unit arrived, or None while the requirement is unmet.

        Only the units that count towards the requirement are considered: a pre-alerted vehicle
        that never moved, or a fourth police car at a one-car incident, must not set the clock.
        """
        if not self.required or not self.satisfied():
            return None
        last = 0.0
        for agency, need in self.required.items():
            if need <= 0:
                continue
            times = sorted(
                a.arrived_s
                for a in self.arrivals
                if a.agency == agency and a.role in (AssignmentRole.RESPOND, AssignmentRole.VERIFY)
            )
            last = max(last, times[need - 1])
        return last


class Engine:
    """One replication of one policy on one scenario."""

    def __init__(
        self,
        network: RoadNetwork,
        scenario: ScenarioState,
        fleet: Fleet,
        cfg: SimConfig,
        opt: OptimizationConfig,
        start: datetime,
        rng: Callable[[str], np.random.Generator],
    ) -> None:
        self.network, self.scenario, self.fleet, self.cfg = network, scenario, fleet, cfg
        self.opt = opt
        self.start, self.rng = start, rng
        self.env = simpy.Environment()
        self.router = Router(network, opt, scenario)
        self.responses: dict[str, ResponseState] = {}
        self.arrivals: list[ArrivalRecord] = []
        self.background_jobs = 0
        self.false_alarm_trips = 0

    # -- clock --------------------------------------------------------------------------------

    @property
    def now_s(self) -> float:
        return float(self.env.now)

    @property
    def now(self) -> datetime:
        return self.start + timedelta(seconds=self.now_s)

    def seconds_of(self, ts: datetime) -> float:
        return (ts - self.start).total_seconds()

    # -- travel -------------------------------------------------------------------------------

    def eta_table(self, incident_nodes: list[int]) -> EtaTable:
        """ETAs from every unit's current node to each incident node, at the current hour."""
        return self.router.eta_table(self.fleet.nodes, incident_nodes, self.now)

    def travel_s(self, from_node: int, to_node: int) -> float:
        """Travel time between two nodes now, or ``inf`` if there is no route.

        A node the graph does not contain is treated as unreachable rather than raised on: the
        network keeps only the largest strongly connected component, so a node on a one-way spur
        is a real possibility, and one such incident must not end the whole replication.
        """
        if from_node == to_node:
            return 0.0
        try:
            table = self.router.eta_table([from_node], [to_node], self.now)
        except KeyError:
            log.warning("node %s is not in the network; treating it as unreachable", to_node)
            return UNREACHABLE_S
        return float(table.seconds[0, 0])

    # -- the response lifecycle ---------------------------------------------------------------

    def register(
        self, incident_id: str, node: int, required: dict[Agency, int], is_false_alarm: bool
    ) -> ResponseState:
        state = self.responses.get(incident_id)
        if state is None:
            state = ResponseState(
                incident_id=incident_id,
                node=int(node),
                created_s=self.now_s,
                required=dict(required),
                is_false_alarm=is_false_alarm,
            )
            self.responses[incident_id] = state
        else:  # a later decision may raise the requirement (second wave)
            state.required = dict(required)
        return state

    def execute(self, decisions: Iterable[DispatchDecision]) -> None:
        """Start a unit process for every assignment that actually moves a vehicle."""
        for decision in decisions:
            state = self.responses.get(decision.incident_id)
            if state is None:
                continue
            for assignment in decision.assignments:
                if assignment.role == AssignmentRole.PRE_ALERT:
                    index = self._claim(assignment.unit_id, state, pre_alert=True)
                    if index is not None:
                        self.fleet.set_state(
                            index, UnitState.READY, self.now_s, incident_id=state.incident_id
                        )
                    continue
                index = self._claim(assignment.unit_id, state)
                if index is None:
                    continue
                # Travel is resolved and the unit is moved to EN_ROUTE **synchronously**, before
                # the next assignment is looked at. A SimPy process does not start until the
                # environment next runs, so deferring this would let two decisions in the same
                # batch each see the unit idle and both claim it.
                travel = self.travel_s(self.fleet.units[index].node, state.node)
                if not np.isfinite(travel):
                    state.dispatched_units.discard(assignment.unit_id)
                    continue  # nothing was changed, so the unit stays available
                self.fleet.set_state(
                    index, UnitState.EN_ROUTE, self.now_s, incident_id=state.incident_id
                )
                self.env.process(self._respond(index, state, assignment.role, travel))

    def _claim(self, unit_id: str, state: ResponseState, pre_alert: bool = False) -> int | None:
        """Take a unit for this incident, or refuse. One unit serves one incident at a time."""
        try:
            index = self.fleet.index_of(unit_id)
        except KeyError:
            log.warning("policy named unknown unit %s", unit_id)
            return None
        unit = self.fleet.units[index]
        if not unit.claimable:
            return None
        if unit_id in state.dispatched_units:
            return None
        if not pre_alert:
            state.dispatched_units.add(unit_id)
        return index

    def _respond(
        self, index: int, state: ResponseState, role: AssignmentRole, travel: float
    ) -> Generator:
        """Travel to the scene, work, then return home. The unit is already EN_ROUTE."""
        unit = self.fleet.units[index]
        dispatched_s = self.now_s
        yield self.env.timeout(travel)
        self.fleet.set_state(index, UnitState.ON_SCENE, self.now_s, node=state.node)
        record = ArrivalRecord(
            incident_id=state.incident_id,
            unit_id=unit.unit_id,
            agency=unit.agency,
            role=role,
            dispatched_s=dispatched_s,
            arrived_s=self.now_s,
            travel_s=travel,
        )
        state.arrivals.append(record)
        self.arrivals.append(record)
        yield self.env.process(self._on_scene(index, state))
        yield self.env.process(self._go_home(index))

    def _on_scene(self, index: int, state: ResponseState) -> Generator:
        """Scene time. A false alarm costs the short look, not a full response."""
        unit = self.fleet.units[index]
        if state.is_false_alarm:
            self.false_alarm_trips += 1
            median = self.cfg.on_scene.false_alarm_s.value.get(
                unit.agency, self.cfg.on_scene.median_s.value[unit.agency]
            )
        else:
            median = self.cfg.on_scene.median_s.value[unit.agency]
        draw = self.rng(f"on_scene/{state.incident_id}/{unit.unit_id}")
        yield self.env.timeout(float(draw.exponential(median)))

    def _go_home(self, index: int) -> Generator:
        unit = self.fleet.units[index]
        back = self.travel_s(unit.node, unit.home_node)
        self.fleet.set_state(index, UnitState.RETURNING, self.now_s)
        yield self.env.timeout(0.0 if not np.isfinite(back) else back)
        self.fleet.set_state(index, UnitState.IDLE, self.now_s, node=unit.home_node)

    # -- running ------------------------------------------------------------------------------

    def run(self, until_s: float) -> None:
        self.env.run(until=until_s)
        self.fleet.close_open_periods(min(self.now_s, until_s))


def fleet_from_roster(units: Iterable[Unit]) -> Fleet:
    return Fleet.from_units(units)
