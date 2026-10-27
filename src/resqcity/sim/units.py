"""Runtime fleet state for the simulator (phase 7, D-032).

``common.schemas.Unit`` is the wire/record view of a unit. This module holds the *mutable*
runtime state the engine drives: where each unit is, what it is doing, and how long it has been
busy. Keeping them apart means a policy can be handed ``Unit`` records without being able to
mutate the fleet.

Two choices worth stating, because they affect the measured numbers:

* **Legal transitions are enforced.** A bug that teleports a unit or sends an on-scene vehicle to
  a second incident would quietly flatter whichever policy caused it, so every transition is
  checked against ``ALLOWED`` and an illegal one raises.
* **READY (pre-alerted) is committed but still claimable, and is not counted as busy.** C2
  pre-alerts units for a VERIFY incident; the crew is in the vehicle but has not moved. If a
  confirmed incident needs that vehicle it should get it, and the pre-alert should not show up as
  utilisation. The alternative -- treating a pre-alert as a full commitment -- would make
  confidence tiering look expensive for a cost no fuel was burned on. This is a modelling choice,
  not a measurement, and phase 10 reports pre-alert counts beside utilisation so it can be judged.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from resqcity.common.schemas import Agency, Unit, UnitState

#: States in which a unit is doing work that occupies the vehicle. READY is excluded on purpose
#: (see the module docstring); OUT_OF_SERVICE is excluded because it is unavailability, not work.
BUSY_STATES = frozenset({
    UnitState.EN_ROUTE,
    UnitState.ON_SCENE,
    UnitState.TRANSPORTING,
    UnitState.AT_HOSPITAL,
    UnitState.RETURNING,
    UnitState.RELOCATING,
    UnitState.ON_OTHER_TASK,
})  # fmt: skip

#: States from which a unit can be given a new incident.
CLAIMABLE = frozenset({UnitState.IDLE, UnitState.READY})

ALLOWED: dict[UnitState, frozenset[UnitState]] = {
    UnitState.IDLE: frozenset(
        {
            UnitState.READY,
            UnitState.EN_ROUTE,
            UnitState.RELOCATING,
            UnitState.ON_OTHER_TASK,
            UnitState.OUT_OF_SERVICE,
        }
    ),
    UnitState.READY: frozenset({UnitState.EN_ROUTE, UnitState.IDLE}),
    # a unit can be recalled while travelling (the incident turned out to be a false alarm), in
    # which case it returns rather than arriving
    UnitState.EN_ROUTE: frozenset({UnitState.ON_SCENE, UnitState.RETURNING}),
    UnitState.ON_SCENE: frozenset({UnitState.TRANSPORTING, UnitState.RETURNING}),
    UnitState.TRANSPORTING: frozenset({UnitState.AT_HOSPITAL}),
    UnitState.AT_HOSPITAL: frozenset({UnitState.RETURNING}),
    UnitState.RETURNING: frozenset({UnitState.IDLE}),
    UnitState.RELOCATING: frozenset({UnitState.IDLE, UnitState.EN_ROUTE}),
    # a background job is never pre-empted for an incident (configs/sim.yaml background.preempt),
    # so the only way out is finishing it
    UnitState.ON_OTHER_TASK: frozenset({UnitState.IDLE}),
    UnitState.OUT_OF_SERVICE: frozenset({UnitState.IDLE}),
}


@dataclass
class BusyPeriod:
    """One stretch of work, for the utilisation log."""

    unit_id: str
    agency: Agency
    state: UnitState
    start_s: float
    end_s: float
    incident_id: str | None


@dataclass
class UnitRuntime:
    unit_id: str
    agency: Agency
    home_node: int
    node: int
    state: UnitState = UnitState.IDLE
    incident_id: str | None = None
    since_s: float = 0.0

    @property
    def busy(self) -> bool:
        return self.state in BUSY_STATES

    @property
    def claimable(self) -> bool:
        return self.state in CLAIMABLE


@dataclass
class Fleet:
    """Every unit's runtime state, indexed the way the ETA tables are (by position)."""

    units: list[UnitRuntime]
    periods: list[BusyPeriod] = field(default_factory=list)

    @classmethod
    def from_units(cls, units: Iterable[Unit]) -> "Fleet":
        return cls([
            UnitRuntime(
                unit_id=u.id, agency=u.agency, home_node=int(u.node), node=int(u.node),
                state=UnitState.IDLE,
            )
            for u in units
        ])  # fmt: skip

    # -- queries ------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.units)

    @property
    def agencies(self) -> list[Agency]:
        return [u.agency for u in self.units]

    @property
    def nodes(self) -> list[int]:
        """Current node per unit, in index order -- what an ETA table is built from."""
        return [u.node for u in self.units]

    def index_of(self, unit_id: str) -> int:
        for i, u in enumerate(self.units):
            if u.unit_id == unit_id:
                return i
        raise KeyError(unit_id)

    def claimable(self, agency: Agency | None = None) -> list[int]:
        return [
            i
            for i, u in enumerate(self.units)
            if u.claimable and (agency is None or u.agency == agency)
        ]

    def busy_count(self, agency: Agency | None = None) -> int:
        return sum(u.busy and (agency is None or u.agency == agency) for u in self.units)

    def count(self, agency: Agency) -> int:
        return sum(u.agency == agency for u in self.units)

    # -- transitions --------------------------------------------------------------------------

    def set_state(
        self,
        index: int,
        state: UnitState,
        now_s: float,
        *,
        incident_id: str | None = None,
        node: int | None = None,
    ) -> None:
        """Move one unit to a new state, closing its busy period if it was working.

        Raises on an illegal transition: silently allowing one would corrupt the utilisation and
        arrival numbers the whole study rests on.
        """
        unit = self.units[index]
        if state not in ALLOWED[unit.state]:
            raise ValueError(
                f"{unit.unit_id}: {unit.state.value} -> {state.value} is not a legal transition"
            )
        if now_s < unit.since_s:
            raise ValueError(f"{unit.unit_id}: time went backwards ({now_s} < {unit.since_s})")
        if unit.busy:
            self.periods.append(
                BusyPeriod(unit.unit_id, unit.agency, unit.state, unit.since_s, now_s,
                           unit.incident_id)
            )  # fmt: skip
        unit.state = state
        unit.since_s = now_s
        if node is not None:
            unit.node = int(node)
        if state is UnitState.IDLE:
            unit.incident_id = None
        elif incident_id is not None:
            unit.incident_id = incident_id

    def close_open_periods(self, now_s: float) -> None:
        """Close the busy period of every working unit, so utilisation is not truncated at the
        end of the run. Idempotent: a second call adds zero-length periods for nothing, because
        each unit's clock is moved to ``now_s``."""
        for unit in self.units:
            if unit.busy and now_s > unit.since_s:
                self.periods.append(
                    BusyPeriod(unit.unit_id, unit.agency, unit.state, unit.since_s, now_s,
                               unit.incident_id)
                )  # fmt: skip
                unit.since_s = now_s

    # -- utilisation --------------------------------------------------------------------------

    def busy_seconds(self, agency: Agency | None = None, after_s: float = 0.0) -> float:
        """Busy seconds recorded after ``after_s`` (so the warm-up can be excluded)."""
        total = 0.0
        for p in self.periods:
            if agency is not None and p.agency != agency:
                continue
            total += max(0.0, p.end_s - max(p.start_s, after_s))
        return total

    def utilisation(self, agency: Agency, from_s: float, to_s: float) -> float:
        """Share of unit-seconds spent busy over a window. 0 when the window is empty."""
        span = (to_s - from_s) * self.count(agency)
        if span <= 0:
            return 0.0
        return self.busy_seconds(agency, after_s=from_s) / span
