"""Facilities from OSM features: snapping, synthetic hospital capabilities, persistence."""

import geopandas as gpd
from shapely.geometry import Point, box

from resqcity.common.schemas import Agency, FacilitySource
from resqcity.geo.facilities import build_facilities, load_facilities, save_facilities
from tests.conftest import LAT0, LON0, STEP


def _raw() -> gpd.GeoDataFrame:
    rows = [
        (
            "node",
            1,
            "police",
            "Vishrambaug Police Station",
            "",
            Point(LON0 + 0.0001, LAT0 + 0.0001),
        ),
        ("node", 2, "fire_station", "", "", Point(LON0 + 2 * STEP, LAT0 + 2 * STEP)),
        # Large footprint -> "major" hospital tier.
        (
            "way",
            3,
            "hospital",
            "Big Hospital",
            "",
            box(LON0 + STEP, LAT0 + STEP, LON0 + 1.8 * STEP, LAT0 + 1.8 * STEP),
        ),
        ("node", 4, "hospital", "Small Clinic", "12", Point(LON0 + 3 * STEP, LAT0)),
        ("node", 5, "hospital", "Far Away", "", Point(LON0 + 1.0, LAT0)),  # dropped: too far
        ("node", 6, "school", "Not a facility", "", Point(LON0, LAT0)),
    ]
    cols = ["osm_type", "osm_id", "amenity", "name", "beds", "geometry"]
    return gpd.GeoDataFrame(
        [dict(zip(cols, r, strict=True)) for r in rows], geometry="geometry", crs="EPSG:4326"
    )


def test_build_facilities(toy_network, geo_cfg):
    fac = build_facilities(_raw(), toy_network, geo_cfg.facilities)
    assert [s.id for s in fac.by_agency(Agency.POLICE)] == ["PS-n1"]
    fire = fac.by_agency(Agency.FIRE)
    assert len(fire) == 1 and fire[0].name.startswith("Unnamed")
    assert {h.id for h in fac.hospitals} == {"H-w3", "H-n4"}
    assert all(h.capabilities_synthetic for h in fac.hospitals)
    amb = fac.by_agency(Agency.AMBULANCE)
    assert {s.id for s in amb} == {"AB-H-w3", "AB-H-n4"}
    assert all(s.source == FacilitySource.SYNTHETIC for s in amb)  # base role is assumed
    others = [s for s in fac.stations if s.agency != Agency.AMBULANCE]
    assert all(s.source == FacilitySource.OSM for s in others)
    assert len(fac.dropped) == 1 and fac.dropped[0].startswith("H-n5")
    node_ids = set(toy_network.nodes["node_id"])
    assert all(s.node in node_ids for s in fac.stations)


def test_major_hospital_gets_major_bed_range(toy_network, geo_cfg):
    fac = build_facilities(_raw(), toy_network, geo_cfg.facilities)
    big = next(h for h in fac.hospitals if h.id == "H-w3")
    lo, hi = geo_cfg.facilities.hospital_capabilities.icu_beds.major
    assert lo <= big.icu_beds <= hi


def test_deterministic_and_round_trip(tmp_path, toy_network, geo_cfg):
    a = build_facilities(_raw(), toy_network, geo_cfg.facilities)
    b = build_facilities(_raw(), toy_network, geo_cfg.facilities)
    assert a == b
    path = tmp_path / "facilities.json"
    save_facilities(a, path, {"source": "test"})
    assert load_facilities(path) == a
