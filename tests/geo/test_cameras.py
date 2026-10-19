"""Camera-site selection on the toy grid."""

import itertools

import numpy as np

from geo.cameras import load_cameras, select_camera_sites, write_cameras_yaml
from geo.config import CameraSitesConfig


def _cfg(**kw) -> CameraSitesConfig:
    data = dict(
        n=3,
        road_classes=["primary", "secondary"],
        min_degree=3,
        weight_degree=0.4,
        weight_betweenness=0.6,
        betweenness_samples=50,
        min_spacing_m=300,
        seed=1,
        output_file="cameras.yaml",
    )
    data.update(kw)
    return CameraSitesConfig(**data)


def test_sites_are_arterial_junctions_spread_out(toy_network):
    cams = select_camera_sites(toy_network, _cfg())
    assert [c.id for c in cams] == ["C01", "C02", "C03"]
    degree = toy_network.distinct_degree()
    nodes = toy_network.nodes.set_index("node_id")
    for cam in cams:
        i = toy_network.index_of(cam.junction_node)
        assert degree[i] >= 3
        assert nodes.loc[cam.junction_node, "in_city"]
        assert "Laxmi Road" in cam.fov_note or "Tilak Road" in cam.fov_note
    for a, b in itertools.combinations(cams, 2):
        d = np.linalg.norm(
            toy_network.xy[toy_network.index_of(a.junction_node)]
            - toy_network.xy[toy_network.index_of(b.junction_node)]
        )
        assert d >= 300


def test_centre_junction_ranks_first(toy_network):
    from tests.conftest import node_id

    assert select_camera_sites(toy_network, _cfg(n=1))[0].junction_node == node_id(2, 2)


def test_deterministic_and_yaml_round_trip(tmp_path, toy_network):
    a = select_camera_sites(toy_network, _cfg())
    assert a == select_camera_sites(toy_network, _cfg())
    path = tmp_path / "cameras.yaml"
    write_cameras_yaml(a, path, _cfg())
    assert load_cameras(path) == a
    assert path.read_text(encoding="utf-8").startswith("# GENERATED")


def test_impossible_spacing_returns_fewer_sites(toy_network):
    assert len(select_camera_sites(toy_network, _cfg(n=10, min_spacing_m=5000))) == 1
