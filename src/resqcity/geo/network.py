"""Road network: OSMnx graph preparation and a compact array form for fast routing.

``prepare_graph`` turns the raw OSMnx graph into the canonical graph (road class, base speed and
travel time on every edge; largest strongly connected component only). ``RoadNetwork`` holds the
same network as numpy arrays, cached as two parquet files. It provides the time-dependent edge
travel times and shortest-path queries used by every other module.

Travel time of edge e at time t:
    t_e = length_e / (v_e * m_hour[class_e] * m_rain * f_e(t) * m_profile)
where v_e is the base speed and f_e(t) the scenario factor (0 = impassable, flooded or closed).
"""

import itertools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from resqcity.geo.config import GeoConfig, RoadClassesConfig
from resqcity.geo.traffic import TrafficModel

if TYPE_CHECKING:
    from resqcity.geo.scenario import ScenarioState

KPH_TO_MPS = 1000.0 / 3600.0
WGS84 = "EPSG:4326"
NODE_COLUMNS = ["node_id", "lat", "lon", "x", "y", "ward_id", "in_city"]
EDGE_COLUMNS = [
    "u",
    "v",
    "length_m",
    "maxspeed_kph",
    "road_class",
    "name",
    "mid_x",
    "mid_y",
    "river_dist_m",
]

# ---------------------------------------------------------------------------------------------
# Graph preparation (OSMnx MultiDiGraph -> canonical graph)
# ---------------------------------------------------------------------------------------------


def _as_list(value: object) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    text = str(value)
    if text.startswith("[") and text.endswith("]"):  # list serialised by graphml
        return [s.strip(" '\"") for s in text[1:-1].split(",") if s.strip(" '\"")]
    return [value]


def road_class(highway: object, cfg: RoadClassesConfig) -> str:
    """Map an OSM ``highway`` tag (possibly a list after simplification) to a road class."""
    classes = [cfg.mapping.get(str(h), cfg.default_class) for h in _as_list(highway)]
    if not classes:
        return cfg.default_class
    return min(classes, key=cfg.priority.index)


_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mph)?", re.IGNORECASE)


def parse_maxspeed(value: object) -> float | None:
    """Smallest numeric speed (km/h) in an OSM ``maxspeed`` tag, or None if not parseable."""
    speeds: list[float] = []
    for item in _as_list(value):
        match = _SPEED_RE.search(str(item))
        if match:
            speed = float(match.group(1)) * (1.609344 if match.group(2) else 1.0)
            if speed > 0:
                speeds.append(speed)
    return min(speeds) if speeds else None


def edge_name(value: object) -> str:
    return " / ".join(str(v) for v in _as_list(value))


def prepare_graph(graph: nx.MultiDiGraph, cfg: GeoConfig) -> nx.MultiDiGraph:
    """Largest strongly connected component with road_class, speed_kph and travel_time (s).

    ``maxspeed_kph`` (NaN if untagged) is kept so the compact network can recompute speeds from
    config at load time: changing a configured speed needs no rebuild.
    """
    largest = max(nx.strongly_connected_components(graph), key=len)
    g = graph.subgraph(largest).copy()
    for _, _, data in g.edges(data=True):
        cls = road_class(data.get("highway"), cfg.road_classes)
        speed = cfg.base_speeds_kph[cls].value
        cap = parse_maxspeed(data.get("maxspeed"))
        if cap is not None:
            speed = min(speed, cap)
        data["road_class"] = cls
        data["maxspeed_kph"] = float("nan") if cap is None else float(cap)
        data["speed_kph"] = float(speed)
        data["travel_time"] = float(data["length"]) / (speed * KPH_TO_MPS)
        data["name"] = edge_name(data.get("name"))
    return g


def graph_tables(graph: nx.MultiDiGraph, projected_crs: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Node and edge tables. Parallel edges collapse to the fastest one (routing needs a CSR).

    Ward assignment and river distances are added by the caller (columns default to empty).
    """
    to_xy = Transformer.from_crs(WGS84, projected_crs, always_xy=True)
    node_ids = np.fromiter(graph.nodes, dtype=np.int64, count=graph.number_of_nodes())
    lon = np.array([graph.nodes[n]["x"] for n in node_ids], dtype=float)
    lat = np.array([graph.nodes[n]["y"] for n in node_ids], dtype=float)
    x, y = to_xy.transform(lon, lat)
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "lat": lat,
            "lon": lon,
            "x": x,
            "y": y,
            "ward_id": "",
            "in_city": True,
        }
    )

    pos = dict(zip(node_ids.tolist(), zip(x, y, strict=True), strict=True))
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        key = (int(u), int(v))
        if key in best and best[key]["travel_time"] <= data["travel_time"]:
            continue
        best[key] = data
    rows = []
    for (u, v), data in best.items():
        (xu, yu), (xv, yv) = pos[u], pos[v]
        rows.append(
            (
                u,
                v,
                float(data["length"]),
                float(data.get("maxspeed_kph", np.nan)),
                data["road_class"],
                data.get("name", ""),
                (xu + xv) / 2.0,
                (yu + yv) / 2.0,
                np.inf,
            )
        )
    edges = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    return nodes, edges.sort_values(["u", "v"], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Compact network
# ---------------------------------------------------------------------------------------------


@dataclass
class RoadNetwork:
    """Array form of the road graph. Node indices (0..n-1) are internal; ids are OSM ids."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    traffic: TrafficModel
    projected_crs: str
    min_edge_time_s: float
    _index: dict[int, int] = field(init=False, repr=False)
    _kdtree: cKDTree = field(init=False, repr=False)
    _to_xy: Transformer = field(init=False, repr=False)
    _to_wgs84: Transformer = field(init=False, repr=False)
    _length_matrix: csr_matrix | None = field(init=False, repr=False)
    _matrix_cache: dict[tuple[Any, ...], csr_matrix] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        missing = set(NODE_COLUMNS) - set(self.nodes) | (set(EDGE_COLUMNS) - set(self.edges))
        if missing:
            raise ValueError(f"network tables missing columns: {sorted(missing)}")
        self.nodes = self.nodes.reset_index(drop=True)
        self.edges = self.edges.reset_index(drop=True)
        self.node_ids = self.nodes["node_id"].to_numpy(np.int64)
        self._index = {int(n): i for i, n in enumerate(self.node_ids)}
        self.u_idx = self.edges["u"].map(self._index).to_numpy(np.int64)
        self.v_idx = self.edges["v"].map(self._index).to_numpy(np.int64)
        self.class_code = np.array(
            [self.traffic.class_code(c) for c in self.edges["road_class"]], dtype=np.int64
        )
        # Base speed from config (class speed, capped by a tagged maxspeed) -> no rebuild needed.
        cap = self.edges["maxspeed_kph"].to_numpy(float)
        self.speed_kph = np.fmin(self.traffic.base_speed_kph[self.class_code], cap)
        self.base_time_s = self.edges["length_m"].to_numpy(float) / (self.speed_kph * KPH_TO_MPS)
        self.xy = self.nodes[["x", "y"]].to_numpy(float)
        self._kdtree = cKDTree(self.xy)
        self._to_xy = Transformer.from_crs(WGS84, self.projected_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.projected_crs, WGS84, always_xy=True)
        self._length_matrix = None
        self._edge_lookup = {
            (int(u), int(v)): i for i, (u, v) in enumerate(zip(self.u_idx, self.v_idx, strict=True))
        }
        self._matrix_cache = {}

    # -- construction / persistence ---------------------------------------------------------

    @classmethod
    def from_tables(cls, nodes: pd.DataFrame, edges: pd.DataFrame, cfg: GeoConfig) -> "RoadNetwork":
        return cls(
            nodes=nodes,
            edges=edges,
            traffic=TrafficModel(cfg),
            projected_crs=cfg.network.projected_crs,
            min_edge_time_s=cfg.network.min_edge_time_s,
        )

    @classmethod
    def load(cls, processed_dir: Path, cfg: GeoConfig) -> "RoadNetwork":
        nodes = pd.read_parquet(processed_dir / cfg.network.nodes_file)
        edges = pd.read_parquet(processed_dir / cfg.network.edges_file)
        return cls.from_tables(nodes, edges, cfg)

    def save(self, processed_dir: Path, cfg: GeoConfig) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.nodes[NODE_COLUMNS].to_parquet(processed_dir / cfg.network.nodes_file, index=False)
        self.edges[EDGE_COLUMNS].to_parquet(processed_dir / cfg.network.edges_file, index=False)

    # -- basic lookups ------------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.u_idx)

    def index_of(self, node_id: int) -> int:
        return self._index[int(node_id)]

    def undirected_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(pair id per edge, first edge index per pair): both directions of a road share an id."""
        a = np.minimum(self.u_idx, self.v_idx)
        b = np.maximum(self.u_idx, self.v_idx)
        _, first, pair_of_edge = np.unique(
            a * self.n_nodes + b, return_index=True, return_inverse=True
        )
        return pair_of_edge, first

    def distinct_degree(self) -> np.ndarray:
        """Number of distinct neighbouring nodes per node (ignoring direction)."""
        _, first = self.undirected_pairs()
        degree = np.zeros(self.n_nodes, dtype=int)
        np.add.at(degree, self.u_idx[first], 1)
        np.add.at(degree, self.v_idx[first], 1)
        return degree

    def project(self, lat: Sequence[float] | float, lon: Sequence[float] | float) -> np.ndarray:
        x, y = self._to_xy.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])

    def unproject(self, x: float, y: float) -> tuple[float, float]:
        """(lat, lon) of a point in the projected CRS (metres)."""
        lon, lat = self._to_wgs84.transform(float(x), float(y))
        return float(lat), float(lon)

    def nearest_nodes(
        self, lat: Sequence[float] | float, lon: Sequence[float] | float, in_city_only: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """(node ids, distance in metres) of the nearest graph node to each point."""
        pts = self.project(lat, lon)
        if not in_city_only:
            dist, idx = self._kdtree.query(pts)
            return self.node_ids[idx], dist
        mask = self.nodes["in_city"].to_numpy(bool)
        subset = np.flatnonzero(mask)
        dist, j = cKDTree(self.xy[subset]).query(pts)
        return self.node_ids[subset[j]], dist

    def nodes_within(self, x: float, y: float, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        """(node indices, straight-line distances in m) of nodes within ``radius_m`` of the
        projected point (x, y)."""
        idx = np.asarray(self._kdtree.query_ball_point([x, y], radius_m), dtype=np.int64)
        return idx, np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y)

    def edge_index(self, u: int, v: int) -> int:
        """Row in ``edges`` of the edge u -> v (OSM node ids)."""
        try:
            return self._edge_lookup[(self.index_of(u), self.index_of(v))]
        except KeyError as exc:
            raise KeyError(f"no edge {u} -> {v}") from exc

    def path_length_m(self, path: Sequence[int]) -> float:
        lengths = self.edges["length_m"].to_numpy(float)
        return float(sum(lengths[self.edge_index(a, b)] for a, b in itertools.pairwise(path)))

    # -- time-dependent travel times ----------------------------------------------------------

    def speed_factors(
        self, t: datetime, scenario: "ScenarioState | None" = None, profile: str = "default"
    ) -> np.ndarray:
        """Multiplier on base speed per edge at time ``t`` (0 means impassable)."""
        hour = t.hour if scenario is None else scenario.local_hour(t)
        factors = self.traffic.hour_multipliers(hour)[self.class_code] * self.traffic.profile(
            profile
        )
        if scenario is not None:
            factors = factors * self.traffic.rain_multiplier(scenario.rain_mm_h(t))
            factors = factors * scenario.edge_factors(t)
        return factors

    def edge_times(
        self, t: datetime, scenario: "ScenarioState | None" = None, profile: str = "default"
    ) -> np.ndarray:
        """Travel time (s) of every edge when entered at time ``t``; ``inf`` if impassable."""
        factors = self.speed_factors(t, scenario, profile)
        with np.errstate(divide="ignore"):
            times = np.where(factors > 0, self.base_time_s / factors, np.inf)
        return np.maximum(times, self.min_edge_time_s)

    def travel_time(
        self,
        u: int,
        v: int,
        t: datetime,
        scenario: "ScenarioState | None" = None,
        profile: str = "default",
    ) -> float:
        """Travel time (s) of the edge u -> v (OSM node ids) entered at time ``t``."""
        return float(self.edge_times(t, scenario, profile)[self.edge_index(u, v)])

    def _matrix(
        self, t: datetime, scenario: "ScenarioState | None", profile: str, reverse: bool = False
    ) -> csr_matrix:
        hour = t.hour if scenario is None else scenario.local_hour(t)
        key = (hour, profile, None if scenario is None else scenario.state_key(t), reverse)
        cached = self._matrix_cache.get(key)
        if cached is None:
            times = self.edge_times(t, scenario, profile)
            ok = np.isfinite(times)
            rows, cols = (self.v_idx, self.u_idx) if reverse else (self.u_idx, self.v_idx)
            cached = csr_matrix(
                (times[ok], (rows[ok], cols[ok])), shape=(self.n_nodes, self.n_nodes)
            )
            self._matrix_cache[key] = cached
        return cached

    def shortest_times(
        self,
        sources: Iterable[int],
        t: datetime,
        scenario: "ScenarioState | None" = None,
        profile: str = "default",
        targets: Iterable[int] | None = None,
        reverse: bool = False,
    ) -> np.ndarray:
        """Shortest travel times (s) from each source node id, using edge times at departure ``t``.

        Static-bucket approximation: every edge uses its time for the departure hour and scenario
        state. Exact time-dependent routing is phase 6. Returns shape (n_sources, n_targets) or
        (n_sources, n_nodes) when ``targets`` is None; unreachable = inf.

        With ``reverse`` the graph is transposed, so the result is the time **to** each source
        from every other node. Dispatch needs times from many units to a few incidents, and one
        reversed search per incident is far cheaper than one forward search per unit.
        """
        src = [self.index_of(s) for s in sources]
        dist = dijkstra(self._matrix(t, scenario, profile, reverse), directed=True, indices=src)
        dist = np.atleast_2d(dist)
        if targets is None:
            return dist
        return dist[:, [self.index_of(n) for n in targets]]

    def shortest_lengths(
        self,
        sources: Iterable[int],
        targets: Iterable[int] | None = None,
        limit_m: float = np.inf,
    ) -> np.ndarray:
        """Shortest road-graph distances (m) from each source node id, ignoring direction (a
        distance on the map, not a drive); beyond ``limit_m`` the result is inf. Shape as in
        :meth:`shortest_times`."""
        if self._length_matrix is None:
            lengths = self.edges["length_m"].to_numpy(float)
            self._length_matrix = csr_matrix(
                (lengths, (self.u_idx, self.v_idx)), shape=(self.n_nodes, self.n_nodes)
            )
        src = [self.index_of(s) for s in sources]
        dist = dijkstra(self._length_matrix, directed=False, indices=src, limit=limit_m)
        dist = np.atleast_2d(dist)
        if targets is None:
            return dist
        return dist[:, [self.index_of(n) for n in targets]]

    def shortest_path(
        self,
        source: int,
        target: int,
        t: datetime,
        scenario: "ScenarioState | None" = None,
        profile: str = "default",
    ) -> tuple[list[int], float]:
        """(node ids along the fastest path, travel time in s); ([], inf) if unreachable."""
        s, g = self.index_of(source), self.index_of(target)
        dist, pred = dijkstra(
            self._matrix(t, scenario, profile), directed=True, indices=s, return_predecessors=True
        )
        if not np.isfinite(dist[g]):
            return [], float("inf")
        path = [g]
        while path[-1] != s:
            path.append(int(pred[path[-1]]))
        return [int(self.node_ids[i]) for i in reversed(path)], float(dist[g])
