"""ETA tables, the green corridor and the fleet roster (phase 6.1)."""

import numpy as np
import pytest

from resqcity.common.schemas import Agency, FacilitySource, GeoPoint, Hospital, Station
from resqcity.optimization.config import (
    OptimizationConfig,
    load_fleet_config,
    load_optimization_config,
)
from resqcity.optimization.fleet import _allocate, ambulance_bases, build_fleet, roster_summary
from resqcity.optimization.routing import EtaTable, Router, nearest_unit
from tests.conftest import at, node_id

CFG = load_optimization_config()
FLEET = load_fleet_config()


def _cfg(rho: float) -> OptimizationConfig:
    data = CFG.model_dump(mode="json")
    data["routing"]["green_corridor"]["rho"]["value"] = rho
    return OptimizationConfig.model_validate(data)


def _router(rho: float = 1.0) -> Router:
    router = Router.__new__(Router)
    router.cfg = _cfg(rho)
    return router


# -- routing ------------------------------------------------------------------------------------


def test_eta_table_matches_a_direct_shortest_path(toy_network):
    """The reversed search must give the same times as the forward one, which is what the
    network's own shortest_times computes."""
    router = Router(toy_network, CFG)
    units = [node_id(0, 0), node_id(4, 4)]
    incidents = [node_id(2, 2), node_id(0, 4)]
    table = router.eta_table(units, incidents, at(0, 10))
    forward = toy_network.shortest_times(units, at(0, 10), None, CFG.routing.profile, incidents)
    assert np.allclose(table.seconds, forward)
    assert table.seconds.shape == (2, 2)


def test_eta_table_handles_no_incidents(toy_network):
    table = Router(toy_network, CFG).eta_table([node_id(0, 0)], [], at(0, 10))
    assert table.seconds.shape == (1, 0)


def test_eta_table_rejects_a_mismatched_matrix():
    with pytest.raises(ValueError, match="shape"):
        EtaTable([1, 2], [3], np.zeros((1, 1)))


def test_green_corridor_scales_only_eligible_ambulances():
    etas = EtaTable([1, 2, 3], [1], np.array([[600.0], [100.0], [300.0]]))
    agencies = [Agency.AMBULANCE, Agency.AMBULANCE, Agency.POLICE]
    out = _router(0.5).green_corridor_etas(etas, agencies)
    assert out[0, 0] == pytest.approx(300.0)  # 0.5*600 = 300, floored at the police arrival
    assert out[1, 0] == pytest.approx(100.0)  # arrives before the police: not helped
    assert out[2, 0] == pytest.approx(300.0)  # police itself unchanged


def test_green_corridor_never_beats_the_police_arrival():
    """An ambulance whose sped-up arrival would precede the earliest police unit is not
    eligible at all: there is no corridor to travel through, so it keeps its own time."""
    etas = EtaTable([1, 2], [1], np.array([[1200.0], [600.0]]))
    out = _router(0.1).green_corridor_etas(etas, [Agency.AMBULANCE, Agency.POLICE])
    assert out[0, 0] == pytest.approx(1200.0)  # 0.1 * 1200 = 120 s is before any police arrive
    helped = _router(0.6).green_corridor_etas(etas, [Agency.AMBULANCE, Agency.POLICE])
    assert helped[0, 0] == pytest.approx(720.0)  # 0.6 * 1200 = 720 >= 600, so it is eligible


def test_green_corridor_is_a_no_op_without_police_or_at_rho_one():
    etas = EtaTable([1, 2], [1], np.array([[600.0], [800.0]]))
    both_amb = _router(0.5).green_corridor_etas(etas, [Agency.AMBULANCE, Agency.AMBULANCE])
    assert np.array_equal(both_amb, etas.seconds)
    off = _router(1.0).green_corridor_etas(etas, [Agency.AMBULANCE, Agency.POLICE])
    assert np.array_equal(off, etas.seconds)


def test_green_corridor_ignores_unreachable_units():
    etas = EtaTable([1, 2], [1], np.array([[600.0], [np.inf]]))
    out = _router(0.5).green_corridor_etas(etas, [Agency.AMBULANCE, Agency.POLICE])
    assert np.array_equal(out, etas.seconds)  # no reachable police: no corridor


def test_nearest_unit_picks_the_closest_of_its_agency():
    etas = EtaTable([1, 2, 3], [1], np.array([[500.0], [200.0], [np.inf]]))
    agencies = [Agency.POLICE, Agency.POLICE, Agency.AMBULANCE]
    assert nearest_unit(etas, agencies, Agency.POLICE, 0) == 1
    assert nearest_unit(etas, agencies, Agency.AMBULANCE, 0) is None  # unreachable
    assert nearest_unit(etas, agencies, Agency.FIRE, 0) is None  # none exist


# -- fleet --------------------------------------------------------------------------------------


def _station(sid: str, agency: Agency, ward: str) -> Station:
    return Station(
        id=sid, name=sid, agency=agency, location=GeoPoint(lat=18.5, lon=73.85),
        node=1, ward_id=ward, source=FacilitySource.OSM,
    )  # fmt: skip


def test_allocation_gives_every_chosen_station_a_unit_then_follows_demand():
    assert _allocate(3, [1.0, 1.0, 1.0], cap=4) == [1, 1, 1]
    assert sum(_allocate(7, [5.0, 1.0, 1.0], cap=4)) == 7
    assert _allocate(7, [5.0, 1.0, 1.0], cap=4)[0] >= 3  # busiest ward gets the most
    assert _allocate(2, [1.0, 1.0, 1.0], cap=4) == [1, 1, 0]  # fewer units than stations
    assert _allocate(0, [1.0], cap=4) == [0]


def test_allocation_respects_the_cap():
    counts = _allocate(9, [10.0, 1.0, 1.0], cap=4)
    assert max(counts) <= 4
    assert sum(counts) == 9


def test_ambulance_bases_prefer_capable_hospitals():
    stations = [_station(f"H{i}", Agency.AMBULANCE, "W01") for i in range(4)]
    hospitals = [
        Hospital(id="H0", name="small", location=GeoPoint(lat=18.5, lon=73.85), node=1,
                 has_trauma=False, has_burns=False, icu_beds=2, beds_available=1),
        Hospital(id="H1", name="trauma", location=GeoPoint(lat=18.5, lon=73.85), node=1,
                 has_trauma=True, has_burns=False, icu_beds=1, beds_available=1),
        Hospital(id="H2", name="burns", location=GeoPoint(lat=18.5, lon=73.85), node=1,
                 has_trauma=False, has_burns=True, icu_beds=0, beds_available=0),
        Hospital(id="H3", name="big", location=GeoPoint(lat=18.5, lon=73.85), node=1,
                 has_trauma=False, has_burns=False, icu_beds=50, beds_available=10),
    ]  # fmt: skip
    data = FLEET.model_dump(mode="json")
    data["placement"]["ambulance_bases"]["max_bases"]["value"] = 2
    cfg = type(FLEET).model_validate(data)
    chosen = {s.id for s in ambulance_bases(stations, hospitals, cfg)}
    assert chosen == {"H1", "H2"}  # capability beats bed count


def test_build_fleet_respects_counts_and_sends_units_to_busy_wards():
    stations = [_station("P1", Agency.POLICE, "BUSY"), _station("P2", Agency.POLICE, "QUIET")]
    data = FLEET.model_dump(mode="json")
    data["counts"] = {"police": {"value": 3, "source": "UNVERIFIED"},
                      "fire": {"value": 0, "source": "UNVERIFIED"},
                      "ambulance": {"value": 0, "source": "UNVERIFIED"}}  # fmt: skip
    cfg = type(FLEET).model_validate(data)
    units = build_fleet(stations, [], {"BUSY": 10.0, "QUIET": 1.0}, cfg)
    assert len(units) == 3
    assert all(u.agency == Agency.POLICE for u in units)
    at_busy = sum(u.home_station == "P1" for u in units)
    assert at_busy == 2  # one each, then the spare goes to the busier ward
    assert roster_summary(units)["stations_used"] == 2


def test_build_fleet_is_capped_by_the_stations_available():
    stations = [_station("F1", Agency.FIRE, "W01")]
    data = FLEET.model_dump(mode="json")
    data["counts"] = {"fire": {"value": 99, "source": "UNVERIFIED"},
                      "police": {"value": 0, "source": "UNVERIFIED"},
                      "ambulance": {"value": 0, "source": "UNVERIFIED"}}  # fmt: skip
    data["placement"]["max_units_per_station"]["value"] = 3
    units = build_fleet(stations, [], {"W01": 1.0}, type(FLEET).model_validate(data))
    assert len(units) == 3  # one station x cap 3, not 99


def test_fleet_counts_are_flagged_unverified():
    from resqcity.common.config import find_unverified

    flagged = set(find_unverified(FLEET))
    assert {"counts.police", "counts.fire", "counts.ambulance"} <= flagged
    assert FLEET.synthetic_roster
