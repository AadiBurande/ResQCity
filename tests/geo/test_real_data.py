"""Checks on the real processed Pune data. Skipped when data/processed is absent (e.g. in CI)."""

import time

import pytest

from common.config import CONFIG_DIR, load_base_config
from geo.cameras import load_cameras
from geo.config import load_geo_config
from geo.facilities import load_facilities
from geo.network import RoadNetwork

BASE, GEO = load_base_config(), load_geo_config()
PROCESSED = BASE.paths.resolve("data_processed")
pytestmark = pytest.mark.skipif(
    not (PROCESSED / GEO.network.edges_file).exists(), reason="processed data not built (make data)"
)
LOAD_LIMIT_S = 10.0  # phase 2 acceptance criterion


@pytest.fixture(scope="module")
def network() -> RoadNetwork:
    return RoadNetwork.load(PROCESSED, GEO)


def test_fast_cache_loads_quickly():
    t0 = time.perf_counter()
    RoadNetwork.load(PROCESSED, GEO)
    assert time.perf_counter() - t0 < LOAD_LIMIT_S


@pytest.mark.slow
def test_graphml_loads_quickly():
    import osmnx as ox

    t0 = time.perf_counter()
    ox.load_graphml(PROCESSED / GEO.network.graphml_file)
    assert time.perf_counter() - t0 < LOAD_LIMIT_S


def test_network_is_plausible(network):
    assert network.n_nodes > 10_000
    assert network.nodes["in_city"].mean() > 0.8
    assert network.nodes["ward_id"].nunique() == 58


def test_cameras_are_in_city_junctions(network):
    cameras = load_cameras(CONFIG_DIR / GEO.camera_sites.output_file)
    assert len(cameras) == GEO.camera_sites.n
    nodes = network.nodes.set_index("node_id")
    assert all(nodes.loc[c.junction_node, "in_city"] for c in cameras)


def test_facilities_snap_to_network(network):
    fac = load_facilities(PROCESSED / GEO.network.facilities_file)
    ids = set(network.nodes["node_id"])
    assert fac.hospitals and all(h.capabilities_synthetic for h in fac.hospitals)
    assert all(s.node in ids for s in fac.stations)


def test_every_festival_street_matches_the_real_network(network):
    from geo.scenario import ScenarioState, load_scenario

    state = ScenarioState(load_scenario("ganeshotsav"), network, BASE.project.timezone)
    unmatched = [name for name, n in state.route_match_counts.items() if n == 0]
    assert not unmatched, f"streets with no OSM match: {unmatched}"


def test_ambulance_bases_are_labelled_synthetic():
    from common.schemas import Agency, FacilitySource

    fac = load_facilities(PROCESSED / GEO.network.facilities_file)
    assert all(s.source == FacilitySource.SYNTHETIC for s in fac.by_agency(Agency.AMBULANCE))
