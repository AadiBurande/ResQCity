"""Shared test fixtures: a toy road network near Pune (no network access, no real data)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, box

from resqcity.geo.build import edge_river_distance
from resqcity.geo.config import GeoConfig, load_geo_config
from resqcity.geo.network import RoadNetwork, graph_tables, prepare_graph
from resqcity.geo.wards import assign_nodes

IST = ZoneInfo("Asia/Kolkata")
LAT0, LON0, STEP = 18.500, 73.850, 0.002  # ~222 m (lat) x ~211 m (lon) grid spacing
N = 5  # 5 x 5 grid; row 2 = "Laxmi Road" (primary), column 2 = "Tilak Road" (secondary)
ISOLATED = 999  # one-way spur: not in the largest strongly connected component


def node_id(r: int, c: int) -> int:
    return 100 + r * N + c


def at(day: int, hour: int, minute: int = 0) -> datetime:
    """A Tuesday-based local timestamp (2026-02-03 is a Tuesday)."""
    return datetime(2026, 2, 3 + day, hour, minute, tzinfo=IST)


def _toy_graph() -> nx.MultiDiGraph:
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    g = nx.MultiDiGraph(crs="EPSG:4326")
    for r in range(N):
        for c in range(N):
            g.add_node(node_id(r, c), y=LAT0 + r * STEP, x=LON0 + c * STEP)

    def add(a: int, b: int, highway: str, name: str, **extra: object) -> None:
        xa, ya = to_xy.transform(g.nodes[a]["x"], g.nodes[a]["y"])
        xb, yb = to_xy.transform(g.nodes[b]["x"], g.nodes[b]["y"])
        length = float(np.hypot(xa - xb, ya - yb))
        for u, v in ((a, b), (b, a)):
            g.add_edge(u, v, highway=highway, length=length, name=name, **extra)

    for r in range(N):
        for c in range(N):
            if c + 1 < N:
                hw, nm = ("primary", "Laxmi Road") if r == 2 else ("residential", f"Lane {r}")
                add(node_id(r, c), node_id(r, c + 1), hw, nm)
            if r + 1 < N:
                hw, nm = ("secondary", "Tilak Road") if c == 2 else ("residential", f"Gali {c}")
                add(node_id(r, c), node_id(r + 1, c), hw, nm)
    # A slower parallel edge (must be collapsed away) and a capped speed limit.
    a, b = node_id(4, 0), node_id(4, 1)
    g.add_edge(a, b, highway="residential", length=g.edges[a, b, 0]["length"] * 3, name="Detour")
    g.edges[node_id(4, 3), node_id(4, 4), 0]["maxspeed"] = "10"
    g.add_node(ISOLATED, y=LAT0 - STEP, x=LON0)
    g.add_edge(ISOLATED, node_id(0, 0), highway="service", length=200.0)
    return g


@pytest.fixture(scope="session")
def geo_cfg() -> GeoConfig:
    return load_geo_config()


@pytest.fixture
def toy_graph() -> nx.MultiDiGraph:
    return _toy_graph()


@pytest.fixture(scope="session")
def toy_wards() -> gpd.GeoDataFrame:
    """W01 covers columns 0-2, W02 column 3; column 4 lies outside every ward."""
    mid = LON0 + 2.5 * STEP
    return gpd.GeoDataFrame(
        {"ward_id": ["W01", "W02"], "name": ["West", "East"]},
        geometry=[
            box(LON0 - STEP / 2, LAT0 - STEP / 2, mid, LAT0 + (N - 0.5) * STEP),
            box(mid, LAT0 - STEP / 2, LON0 + 3.5 * STEP, LAT0 + (N - 0.5) * STEP),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture(scope="session")
def toy_river_xy(geo_cfg: GeoConfig) -> np.ndarray:
    """River line ~55 m south of grid row 0 (projected points every 10 m)."""
    lat = LAT0 - 0.0005
    line = LineString([(LON0 - STEP, lat), (LON0 + N * STEP, lat)])
    proj = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(geo_cfg.network.projected_crs)
    return np.array(proj.iloc[0].segmentize(10).coords)


@pytest.fixture(scope="session")
def toy_network(
    geo_cfg: GeoConfig, toy_wards: gpd.GeoDataFrame, toy_river_xy: np.ndarray
) -> RoadNetwork:
    graph = prepare_graph(_toy_graph(), geo_cfg)
    nodes, edges = graph_tables(graph, geo_cfg.network.projected_crs)
    nodes["ward_id"], nodes["in_city"] = assign_nodes(
        nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), toy_wards, geo_cfg.network.projected_crs
    )
    index = {int(n): i for i, n in enumerate(nodes["node_id"])}
    edges["river_dist_m"] = edge_river_distance(
        nodes[["x", "y"]].to_numpy(),
        edges["u"].map(index).to_numpy(),
        edges["v"].map(index).to_numpy(),
        edges[["mid_x", "mid_y"]].to_numpy(),
        toy_river_xy,
    )
    return RoadNetwork.from_tables(nodes, edges, geo_cfg)
