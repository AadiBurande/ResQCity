"""Hospital choice, transport and handover (phase 7 slice 5, D-032).

An ambulance that transports a patient is unavailable for a long time: travel to the hospital,
then a handover before it can take another case. That is most of an ambulance's busy time in a
real system, so leaving it out would make the ambulance fleet look far more available than it is
-- the same class of error as the missing background workload of D-031 item 3.

Three rules, each a decision rather than a measurement:

1. **Nearest CAPABLE hospital, not nearest hospital.** A trauma case goes to a hospital with
   ``has_trauma``, a burns case to one with ``has_burns``; anything else goes to the nearest.
   Capability comes from ``data/processed/facilities.json`` and is **synthetic** (the
   ``capabilities_synthetic`` flag on every record says so), so this routing is structurally
   right and numerically unverified.
2. **Capacity is a counted resource.** Each hospital holds ``icu_beds`` concurrent handovers;
   beyond that an arriving ambulance queues. Without a cap, a mass-casualty scenario would
   absorb unlimited patients at one address.
3. **No capable hospital reachable is logged, never silently dropped** (``configs/sim.yaml`` ->
   ``transport.no_capable_hospital: log_and_release``). A city that cannot place a patient is a
   finding, and an ambulance that vanished would corrupt the fleet accounting.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from resqcity.common.schemas import EventType, Hospital

log = logging.getLogger(__name__)

#: Which incident types need which capability. Accidents are the trauma pathway; a fire or smoke
#: casualty is the burns pathway. Crowd and other have no capability requirement.
CAPABILITY_FOR: dict[EventType, str] = {
    EventType.ACCIDENT: "has_trauma",
    EventType.FIRE_SMOKE: "has_burns",
}


@dataclass
class HandoverRecord:
    hospital_id: str
    incident_id: str
    unit_id: str
    arrived_s: float
    released_s: float
    #: Time spent waiting for a free handover bay. NOT the handover duration, which is
    #: released_s - arrived_s - waited_s; conflating the two would report a queue where there
    #: was none.
    waited_s: float

    @property
    def total_s(self) -> float:
        return self.released_s - self.arrived_s


@dataclass
class HospitalNetwork:
    """Hospitals with their capability flags and a concurrent-handover count."""

    hospitals: list[Hospital]
    #: hospital id -> handovers in progress
    busy: dict[str, int] = field(default_factory=dict)
    handovers: list[HandoverRecord] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)  # (incident, reason)

    def __post_init__(self) -> None:
        self.by_id = {h.id: h for h in self.hospitals}
        self.busy = dict.fromkeys(self.by_id, 0)

    def capability_for(self, event_type: EventType) -> str | None:
        return CAPABILITY_FOR.get(event_type)

    def capable(self, event_type: EventType) -> list[Hospital]:
        """Hospitals that can take this type. Falls back to all of them when none is required."""
        needed = self.capability_for(event_type)
        if needed is None:
            return list(self.hospitals)
        return [h for h in self.hospitals if getattr(h, needed)]

    def slots(self, hospital_id: str) -> int:
        """Concurrent handovers this hospital can hold. ICU beds stand in for handover bays.

        A hospital recorded with zero beds still takes one patient at a time: a zero here is
        missing OSM data, not a hospital that cannot receive anyone.
        """
        return max(1, int(self.by_id[hospital_id].icu_beds))

    def has_room(self, hospital_id: str) -> bool:
        return self.busy.get(hospital_id, 0) < self.slots(hospital_id)

    def choose(
        self,
        event_type: EventType,
        travel_s: "Sequence[float] | dict[str, float]",
        prefer_free: bool = True,
    ) -> str | None:
        """Nearest capable hospital by travel time, preferring one with a free bay.

        ``travel_s`` maps hospital id -> seconds (or is a sequence aligned with ``hospitals``).
        Returns ``None`` when no capable hospital is reachable, which the caller must log.
        """
        times = (
            travel_s
            if isinstance(travel_s, dict)
            else {h.id: travel_s[i] for i, h in enumerate(self.hospitals)}
        )
        options = [
            h.id for h in self.capable(event_type) if h.id in times and times[h.id] < float("inf")
        ]
        if not options:
            return None
        if prefer_free and (free := [hid for hid in options if self.has_room(hid)]):
            options = free
        return min(options, key=lambda hid: (times[hid], hid))

    # -- occupancy ----------------------------------------------------------------------------

    def take(self, hospital_id: str) -> None:
        self.busy[hospital_id] = self.busy.get(hospital_id, 0) + 1

    def release(self, hospital_id: str) -> None:
        self.busy[hospital_id] = max(0, self.busy.get(hospital_id, 0) - 1)

    def record(self, record: HandoverRecord) -> None:
        self.handovers.append(record)

    def refuse(self, incident_id: str, reason: str) -> None:
        """No capable hospital was reachable. Logged, not dropped."""
        self.refused.append((incident_id, reason))
        log.info("no capable hospital for %s: %s", incident_id, reason)

    def report(self) -> dict[str, object]:
        waits = [r.waited_s for r in self.handovers]
        return {
            "status": "hospital CAPABILITIES AND BEDS ARE SYNTHETIC (facilities.json records "
            "capabilities_synthetic=true); only the locations are real OSM data",
            "hospitals": len(self.hospitals),
            "with_trauma": sum(h.has_trauma for h in self.hospitals),
            "with_burns": sum(h.has_burns for h in self.hospitals),
            "handovers": len(self.handovers),
            "mean_bay_wait_s": sum(waits) / len(waits) if waits else 0.0,
            "max_bay_wait_s": max(waits, default=0.0),
            "mean_time_at_hospital_s": (
                sum(r.total_s for r in self.handovers) / len(self.handovers)
                if self.handovers
                else 0.0
            ),
            "patients_with_no_capable_hospital": len(self.refused),
        }
