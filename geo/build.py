"""Build processed geographic data from the raw downloads (offline, deterministic).

    uv run python -m geo.build [--skip-cameras]

Reads data/raw (see geo.download) and writes:
- data/processed/pune_drive.graphml          canonical OSMnx graph (largest SCC, speeds, times)
- data/processed/network_{nodes,edges}.parquet  fast-loading RoadNetwork cache
- data/processed/wards.geojson               wards used by the project
- data/processed/facilities.json             Stations + Hospitals (capabilities SYNTHETIC)
- configs/cameras.yaml                       selected camera junctions
- data/processed/build_meta.json             counts, timings, input hashes
"""

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import shapely
from scipy.spatial import cKDTree

from common.config import CONFIG_DIR, load_base_config
from geo.cameras import select_camera_sites, write_cameras_yaml
from geo.config import GeoConfig, load_geo_config
from geo.download import sha256
from geo.facilities import build_facilities, save_facilities
from geo.network import RoadNetwork, graph_tables, prepare_graph
from geo.wards import assign_nodes, load_wards, ward_table

log = logging.getLogger(__name__)


def river_points(rivers: gpd.GeoDataFrame, projected_crs: str, step_m: float) -> np.ndarray:
    """River lines (projected) densified to points ``step_m`` apart."""
    geoms = rivers.to_crs(projected_crs).geometry.to_numpy()
    return shapely.get_coordinates(shapely.segmentize(geoms, step_m))


def edge_river_distance(
    network_xy: np.ndarray,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    mids: np.ndarray,
    river_xy: np.ndarray,
) -> np.ndarray:
    """Distance (m) from each edge to the river: min over its two end nodes and midpoint."""
    if len(river_xy) == 0:
        return np.full(len(u_idx), np.inf)
    tree = cKDTree(river_xy)
    node_dist, _ = tree.query(network_xy)
    mid_dist, _ = tree.query(mids)
    return np.minimum.reduce([node_dist[u_idx], node_dist[v_idx], mid_dist])


def build(
    cfg: GeoConfig, raw_dir: Path, processed_dir: Path, cameras: bool = True
) -> dict[str, object]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_graph = ox.load_graphml(raw_dir / cfg.network.raw_graphml_file)
    graph = prepare_graph(raw_graph, cfg)
    ox.save_graphml(graph, processed_dir / cfg.network.graphml_file)
    timings["graph_s"] = time.perf_counter() - t0
    log.info(
        "graph: %d -> %d nodes (largest SCC)", raw_graph.number_of_nodes(), graph.number_of_nodes()
    )

    nodes, edges = graph_tables(graph, cfg.network.projected_crs)
    wards = load_wards(raw_dir, cfg)
    nodes["ward_id"], nodes["in_city"] = assign_nodes(
        nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), wards, cfg.network.projected_crs
    )
    wards.to_file(processed_dir / cfg.network.wards_file, driver="GeoJSON")

    rivers = gpd.read_file(raw_dir / cfg.rivers.raw_file)
    river_xy = river_points(rivers, cfg.network.projected_crs, cfg.network.river_sample_step_m)
    index = {int(n): i for i, n in enumerate(nodes["node_id"])}
    u_idx = edges["u"].map(index).to_numpy()
    v_idx = edges["v"].map(index).to_numpy()
    edges["river_dist_m"] = edge_river_distance(
        nodes[["x", "y"]].to_numpy(), u_idx, v_idx, edges[["mid_x", "mid_y"]].to_numpy(), river_xy
    )
    network = RoadNetwork.from_tables(nodes, edges, cfg)
    network.save(processed_dir, cfg)
    timings["tables_s"] = time.perf_counter() - t0 - sum(timings.values())

    raw_facilities = gpd.read_file(raw_dir / cfg.facilities.raw_file)
    facilities = build_facilities(raw_facilities, network, cfg.facilities)
    save_facilities(
        facilities,
        processed_dir / cfg.network.facilities_file,
        {"source": "OpenStreetMap (ODbL)", "built_from": cfg.facilities.raw_file.as_posix()},
    )

    n_cameras = 0
    if cameras:
        t1 = time.perf_counter()
        sites = select_camera_sites(network, cfg.camera_sites)
        write_cameras_yaml(sites, CONFIG_DIR / cfg.camera_sites.output_file, cfg.camera_sites)
        n_cameras = len(sites)
        timings["cameras_s"] = time.perf_counter() - t1

    wt = ward_table(wards, cfg.network.projected_crs)
    meta: dict[str, object] = {
        "nodes": network.n_nodes,
        "edges": network.n_edges,
        "nodes_in_city": int(nodes["in_city"].sum()),
        "wards": len(wards),
        "ward_area_km2_total": round(float(wt["area_km2"].sum()), 1),
        "edges_by_class": edges["road_class"].value_counts().to_dict(),
        "edge_length_km": round(float(edges["length_m"].sum()) / 1000, 1),
        "river_points": len(river_xy),
        "stations_by_agency": {
            a.value: len(facilities.by_agency(a))
            for a in sorted({s.agency for s in facilities.stations})
        },
        "hospitals": len(facilities.hospitals),
        "facilities_dropped": len(facilities.dropped),
        "cameras": n_cameras,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "inputs_sha256": {
            p.as_posix(): sha256(raw_dir / p)
            for p in (
                cfg.network.raw_graphml_file,
                cfg.wards.raw_file,
                cfg.rivers.raw_file,
                cfg.facilities.raw_file,
            )
        },
    }
    (processed_dir / cfg.network.build_meta_file).write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build processed geographic data")
    parser.add_argument(
        "--skip-cameras", action="store_true", help="keep configs/cameras.yaml as is"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_geo_config()
    meta = build(
        cfg,
        base.paths.resolve("data_raw"),
        base.paths.resolve("data_processed"),
        not args.skip_cameras,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
