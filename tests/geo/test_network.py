"""Road-network preparation, travel times and routing on the toy grid."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from resqcity.geo.network import (
    KPH_TO_MPS,
    RoadNetwork,
    graph_tables,
    parse_maxspeed,
    prepare_graph,
    road_class,
)
from tests.conftest import ISOLATED, at, node_id


@pytest.mark.parametrize(
    ("highway", "expected"),
    [
        ("primary", "primary"),
        (["residential", "primary"], "primary"),
        ("['secondary', 'tertiary']", "secondary"),
        ("motorway_link", "trunk"),
        ("track", "service"),
        (None, "service"),
    ],
)
def test_road_class(geo_cfg, highway, expected):
    assert road_class(highway, geo_cfg.road_classes) == expected


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("40", 40.0),
        ("30 mph", pytest.approx(48.28, abs=0.01)),
        (["50", "30"], 30.0),
        ("signals", None),
        (None, None),
    ],
)
def test_parse_maxspeed(tag, expected):
    assert parse_maxspeed(tag) == expected


def test_prepare_graph_keeps_largest_scc_and_sets_speeds(geo_cfg, toy_graph):
    g = prepare_graph(toy_graph, geo_cfg)
    assert ISOLATED not in g
    assert nx.is_strongly_connected(g)
    data = g.edges[node_id(2, 0), node_id(2, 1), 0]
    assert data["road_class"] == "primary"
    assert data["speed_kph"] == geo_cfg.base_speeds_kph["primary"].value
    assert data["travel_time"] == pytest.approx(data["length"] / (data["speed_kph"] * KPH_TO_MPS))
    capped = g.edges[node_id(4, 3), node_id(4, 4), 0]
    assert capped["speed_kph"] == 10.0  # maxspeed below the class speed caps it


def test_compact_network_applies_class_speed_and_cap(geo_cfg, toy_network):
    free = toy_network.edge_index(node_id(2, 0), node_id(2, 1))
    capped = toy_network.edge_index(node_id(4, 3), node_id(4, 4))
    assert toy_network.speed_kph[free] == geo_cfg.base_speeds_kph["primary"].value
    assert toy_network.speed_kph[capped] == 10.0


def test_parallel_edges_collapse_to_fastest(geo_cfg, toy_graph):
    _, edges = graph_tables(prepare_graph(toy_graph, geo_cfg), geo_cfg.network.projected_crs)
    pair = edges[(edges["u"] == node_id(4, 0)) & (edges["v"] == node_id(4, 1))]
    assert len(pair) == 1
    assert pair["name"].iloc[0] != "Detour"


def test_save_load_round_trip(tmp_path: Path, geo_cfg, toy_network):
    toy_network.save(tmp_path, geo_cfg)
    loaded = RoadNetwork.load(tmp_path, geo_cfg)
    pd.testing.assert_frame_equal(
        loaded.nodes, toy_network.nodes[loaded.nodes.columns], check_dtype=False
    )
    pd.testing.assert_frame_equal(
        loaded.edges, toy_network.edges[loaded.edges.columns], check_dtype=False
    )
    np.testing.assert_allclose(loaded.base_time_s, toy_network.base_time_s)


def test_travel_time_by_hour_and_profile(geo_cfg, toy_network):
    u, v = node_id(2, 0), node_id(2, 1)  # primary
    base = toy_network.base_time_s[toy_network.edge_index(u, v)]
    off_peak = toy_network.travel_time(u, v, at(0, 7))  # 07:00 is outside every period
    assert off_peak == pytest.approx(base)
    midday = geo_cfg.traffic_periods[1].multipliers["primary"].value
    assert toy_network.travel_time(u, v, at(0, 14)) == pytest.approx(base / midday)
    am = geo_cfg.traffic_periods[0].multipliers["primary"].value
    assert toy_network.travel_time(u, v, at(0, 10)) == pytest.approx(base / am)
    assert toy_network.travel_time(u, v, at(0, 3)) < off_peak  # night is faster
    amb = geo_cfg.vehicle_profiles["ambulance"].value
    assert toy_network.travel_time(u, v, at(0, 7), profile="ambulance") == pytest.approx(base / amb)


def test_unknown_profile_rejected(toy_network):
    with pytest.raises(KeyError, match="profile"):
        toy_network.edge_times(at(0, 14), profile="hovercraft")


def test_missing_edge_rejected(toy_network):
    with pytest.raises(KeyError):
        toy_network.travel_time(node_id(0, 0), node_id(4, 4), at(0, 14))


def test_shortest_times_match_networkx(toy_network):
    t = at(0, 10)
    times = toy_network.edge_times(t)
    g = nx.DiGraph()
    for (u, v), w in zip(toy_network.edges[["u", "v"]].itertuples(index=False), times, strict=True):
        g.add_edge(u, v, w=w)
    src, dst = node_id(0, 0), node_id(4, 4)
    expected = nx.shortest_path_length(g, src, dst, weight="w")
    got = toy_network.shortest_times([src], t, targets=[dst])[0, 0]
    assert got == pytest.approx(expected)
    path, seconds = toy_network.shortest_path(src, dst, t)
    assert path[0] == src and path[-1] == dst
    assert seconds == pytest.approx(expected)
    assert toy_network.path_length_m(path) > 0


def test_peak_is_never_faster_than_off_peak(toy_network):
    src = [node_id(0, 0)]
    peak = toy_network.shortest_times(src, at(0, 18))
    off = toy_network.shortest_times(src, at(0, 7))
    assert np.all(peak >= off - 1e-9)


def test_nearest_nodes(toy_network):
    row = toy_network.nodes.iloc[7]
    ids, dist = toy_network.nearest_nodes([row["lat"]], [row["lon"]])
    assert ids[0] == row["node_id"]
    assert dist[0] == pytest.approx(0.0, abs=1e-6)


def test_wards_and_city_flag(toy_network):
    nodes = toy_network.nodes.set_index("node_id")
    assert nodes.loc[node_id(1, 1), "ward_id"] == "W01"
    assert nodes.loc[node_id(1, 3), "ward_id"] == "W02"
    outside = nodes.loc[node_id(1, 4)]
    assert not outside["in_city"]
    assert outside["ward_id"] == "W02"  # nearest ward


def test_river_distance(toy_network):
    edges = toy_network.edges
    row0 = edges[(edges["u"] == node_id(0, 0)) & (edges["v"] == node_id(0, 1))][
        "river_dist_m"
    ].iloc[0]
    row3 = edges[(edges["u"] == node_id(3, 0)) & (edges["v"] == node_id(3, 1))][
        "river_dist_m"
    ].iloc[0]
    assert row0 == pytest.approx(55.6, abs=8)
    assert row3 > 600


def test_shortest_lengths_and_unproject(toy_network):
    from tests.conftest import node_id

    a, b = node_id(2, 0), node_id(2, 2)
    d = toy_network.shortest_lengths([a], [b])[0, 0]
    two_edges = toy_network.path_length_m([a, node_id(2, 1), b])
    assert d == pytest.approx(two_edges)
    assert toy_network.shortest_lengths([a], [b], limit_m=d / 2)[0, 0] == float("inf")
    i = toy_network.index_of(a)
    lat, lon = toy_network.unproject(*toy_network.xy[i])
    assert lat == pytest.approx(toy_network.nodes["lat"][i], abs=1e-7)
    assert lon == pytest.approx(toy_network.nodes["lon"][i], abs=1e-7)
