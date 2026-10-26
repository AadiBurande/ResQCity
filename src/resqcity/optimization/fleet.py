"""Build the unit roster: how many units each agency has and which station each starts from.

SYNTHETIC roster on REAL station locations (phase 6, D-031). The counts in ``configs/fleet.yaml``
are press-derived and UNVERIFIED; this module only decides where those units stand.

Placement rule (``demand_weighted``): stations are ranked by the forecast incident demand of
their ward, every chosen station gets at least one unit, and the remaining units are handed out
by largest remainder, capped per station. Demand is passed in rather than read, so this module
does not import the generator -- ``tests/test_architecture.py`` forbids ``optimization -> sim``.
"""

from collections import Counter

from resqcity.common.schemas import Agency, Station, Unit, UnitState
from resqcity.optimization.config import FleetConfig

ETA_PROFILE = "default"


def _rank_stations(
    stations: list[Station], demand_by_ward: dict[str, float], limit: int | None
) -> list[Station]:
    """Busiest wards first; ties by station id so the roster is reproducible."""
    ranked = sorted(stations, key=lambda s: (-demand_by_ward.get(s.ward_id, 0.0), s.id))
    return ranked if limit is None else ranked[:limit]


def _allocate(n_units: int, weights: list[float], cap: int) -> list[int]:
    """Largest-remainder allocation of ``n_units`` over weighted slots, at least 1 each, capped.

    Every slot offered to the allocator is one we intend to staff, so each gets a unit before
    any slot gets a second. Whatever is left is shared out in proportion to demand.
    """
    slots = len(weights)
    if slots == 0 or n_units <= 0:
        return [0] * slots
    if n_units <= slots:
        return [1] * n_units + [0] * (slots - n_units)
    total = sum(weights) or float(slots)
    spare = n_units - slots
    shares = [spare * (w / total) for w in weights]
    counts = [1 + int(s) for s in shares]
    remainder = sorted(range(slots), key=lambda i: (-(shares[i] - int(shares[i])), i))
    short = n_units - sum(counts)
    for i in remainder[:short]:
        counts[i] += 1
    # respect the cap, pushing any overflow onto stations that still have room
    overflow = 0
    for i, c in enumerate(counts):
        if c > cap:
            overflow += c - cap
            counts[i] = cap
    for i in remainder:
        while overflow > 0 and counts[i] < cap:
            counts[i] += 1
            overflow -= 1
    return counts


def ambulance_bases(stations: list[Station], hospitals: list, cfg: FleetConfig) -> list[Station]:
    """OSM tags 590 "hospitals", most of them small nursing homes (D-013), so prefer the ones
    with a trauma or burns capability and then the largest by ICU beds."""
    rule = cfg.placement.ambulance_bases
    capable = {h.id for h in hospitals if h.has_trauma or h.has_burns}
    beds = {h.id: h.icu_beds for h in hospitals}
    key = (
        (lambda s: (s.id not in capable, -beds.get(s.id, 0), s.id))
        if rule.prefer_capable
        else (lambda s: (-beds.get(s.id, 0), s.id))
    )
    return sorted(stations, key=key)[: rule.max_bases.value]


def build_fleet(
    stations: list[Station],
    hospitals: list,
    demand_by_ward: dict[str, float],
    cfg: FleetConfig,
) -> list[Unit]:
    """One ``Unit`` per vehicle, standing at its home station and idle."""
    by_agency: dict[Agency, list[Station]] = {a: [] for a in Agency}
    for station in stations:
        by_agency[station.agency].append(station)
    units: list[Unit] = []
    for agency, count in sorted(cfg.counts.items(), key=lambda kv: kv[0].value):
        pool = by_agency[agency]
        if agency == Agency.AMBULANCE:
            pool = ambulance_bases(pool, hospitals, cfg)
        cap = cfg.placement.max_units_per_station.value
        needed = min(count.value, len(pool) * cap)
        chosen = _rank_stations(pool, demand_by_ward, limit=min(len(pool), needed))
        weights = [demand_by_ward.get(s.ward_id, 0.0) for s in chosen]
        for station, n in zip(chosen, _allocate(needed, weights, cap), strict=True):
            for k in range(n):
                units.append(
                    Unit(
                        id=f"{agency.value[:3].upper()}-{station.id}-{k}",
                        agency=agency,
                        home_station=station.id,
                        node=station.node,
                        state=UnitState.IDLE,
                        eta_fn_ref=ETA_PROFILE,
                        position=station.location,
                    )
                )
    return units


def roster_summary(units: list[Unit]) -> dict[str, object]:
    by_agency = Counter(u.agency.value for u in units)
    by_station = Counter(u.home_station for u in units)
    return {
        "units": len(units),
        "by_agency": dict(sorted(by_agency.items())),
        "stations_used": len(by_station),
        "max_units_at_one_station": max(by_station.values(), default=0),
    }
