"""Download raw geographic inputs. This is the ONLY module in geo/ that touches the network.

    uv run python -m geo.download [--force]

Outputs (data/raw, gitignored):
- wards/pune-electoral-wards_2022.geojson  PMC 2022 electoral wards (DataMeet, CC BY-SA 2.5 IN)
- pune_drive_raw.graphml                   OSM drivable network inside the buffered ward union
- osm_rivers.geojson                       Mula / Mutha / Mula-Mutha river lines
- osm_facilities.geojson                   amenity=police / fire_station / hospital
- download_meta.json                       download time, versions, counts, file hashes
"""

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from common.config import load_base_config
from geo.config import GeoConfig, load_geo_config
from geo.facilities import FACILITY_COLUMNS
from geo.wards import buffered, load_wards, study_boundary

log = logging.getLogger(__name__)
USER_AGENT = "ResQCity research prototype (university project)"
RIVER_SEARCH_EXTRA_M = 2000.0
LINE_TYPES = ("LineString", "MultiLineString")
# Public Overpass servers are often at capacity: retry with exponential backoff, then try a mirror.
OVERPASS_ENDPOINTS = ("https://overpass-api.de/api", "https://overpass.kumi.systems/api")
RETRY_ATTEMPTS = 4
RETRY_FIRST_WAIT_S = 20.0
R = TypeVar("R")


def with_retries(fn: Callable[[], R], what: str) -> R:
    """Call ``fn`` with exponential backoff, cycling through the Overpass endpoints."""
    wait = RETRY_FIRST_WAIT_S
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        ox.settings.overpass_url = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            return fn()
        except Exception as exc:
            last = exc
            log.warning(
                "%s failed on %s (%s); retrying in %.0f s",
                what,
                ox.settings.overpass_url,
                type(exc).__name__,
                wait,
            )
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(f"{what}: all {RETRY_ATTEMPTS} attempts failed") from last


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_osmnx(raw_dir: Path, cfg: GeoConfig) -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(raw_dir / cfg.network.osm_cache_dir)
    ox.settings.http_user_agent = USER_AGENT
    ox.settings.requests_timeout = 300
    if cfg.network.osm_snapshot_date:
        # Overpass "attic" query: data as it was at this date, independent of later OSM edits.
        date = cfg.network.osm_snapshot_date
        ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}]{{maxsize}}[date:"{date}"]'


def download_wards(raw_dir: Path, cfg: GeoConfig, force: bool) -> Path:
    path = raw_dir / cfg.wards.raw_file
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(cfg.wards.url, timeout=120, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def download_area(raw_dir: Path, cfg: GeoConfig) -> BaseGeometry:
    """Polygon to download: configured bbox, else the ward union buffered by boundary_buffer_m."""
    if cfg.network.bbox is not None:
        b = cfg.network.bbox
        return box(b.west, b.south, b.east, b.north)
    wards = load_wards(raw_dir, cfg)
    return buffered(study_boundary(wards), cfg.network.boundary_buffer_m, cfg.network.projected_crs)


def download_graph(area: BaseGeometry, raw_dir: Path, cfg: GeoConfig, force: bool) -> Path:
    path = raw_dir / cfg.network.raw_graphml_file
    if path.exists() and not force:
        return path
    graph = with_retries(
        lambda: ox.graph_from_polygon(
            area,
            network_type=cfg.network.network_type,
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        ),
        "road network",
    )
    ox.save_graphml(graph, path)
    return path


def _name_matches(row: pd.Series, names: list[str]) -> bool:
    wanted = [n.lower() for n in names]
    for col in ("name", "name:en", "name:mr", "alt_name"):
        value = row.get(col)
        if isinstance(value, str) and any(w in value.lower() for w in wanted):
            return True
    return False


def download_rivers(area: BaseGeometry, raw_dir: Path, cfg: GeoConfig, force: bool) -> Path:
    path = raw_dir / cfg.rivers.raw_file
    if path.exists() and not force:
        return path
    search = buffered(area, RIVER_SEARCH_EXTRA_M, cfg.network.projected_crs)
    feats = with_retries(
        lambda: ox.features_from_polygon(search, tags={"waterway": "river"}), "rivers"
    )
    feats = feats[feats.geometry.geom_type.isin(LINE_TYPES)]
    feats = feats[feats.apply(_name_matches, axis=1, names=cfg.rivers.names)]
    if feats.empty:
        raise RuntimeError(f"no river lines matched {cfg.rivers.names}")
    out = gpd.GeoDataFrame(
        {"name": feats["name"].astype(str).to_numpy()},
        geometry=feats.geometry.to_numpy(),
        crs=feats.crs,
    )
    out.to_file(path, driver="GeoJSON")
    return path


def download_facilities(area: BaseGeometry, raw_dir: Path, cfg: GeoConfig, force: bool) -> Path:
    path = raw_dir / cfg.facilities.raw_file
    if path.exists() and not force:
        return path
    tags = cfg.facilities.tags
    amenities = sorted({tags.police["amenity"], tags.fire["amenity"], tags.hospital["amenity"]})
    feats = with_retries(
        lambda: ox.features_from_polygon(area, tags={"amenity": amenities}), "facilities"
    ).reset_index()
    out = gpd.GeoDataFrame(
        {
            "osm_type": feats["element"].astype(str),
            "osm_id": feats["id"].astype("int64"),
            "amenity": feats["amenity"].astype(str),
            "name": feats.get("name", pd.Series(index=feats.index, dtype=str)).fillna(""),
            "beds": feats.get("beds", pd.Series(index=feats.index, dtype=str))
            .fillna("")
            .astype(str),
        },
        geometry=feats.geometry,
        crs=feats.crs,
    )
    out = out[out["amenity"].isin(amenities)][FACILITY_COLUMNS]
    out.to_file(path, driver="GeoJSON")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw geographic inputs")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base, cfg = load_base_config(), load_geo_config()
    raw_dir = base.paths.resolve("data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    configure_osmnx(raw_dir, cfg)

    log.info("wards ...")
    wards_path = download_wards(raw_dir, cfg, args.force)
    area = download_area(raw_dir, cfg)
    log.info("road network (this can take a few minutes) ...")
    graph_path = download_graph(area, raw_dir, cfg, args.force)
    log.info("rivers ...")
    rivers_path = download_rivers(area, raw_dir, cfg, args.force)
    log.info("facilities ...")
    facilities_path = download_facilities(area, raw_dir, cfg, args.force)

    files = {
        p.relative_to(raw_dir).as_posix(): sha256(p)
        for p in (wards_path, graph_path, rivers_path, facilities_path)
    }
    meta = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "osm_attribution": "Map data © OpenStreetMap contributors, available under the ODbL",
        "wards_source": cfg.wards.url,
        "wards_licence": "CC BY-SA 2.5 IN (DataMeet Pune chapter)",
        "area": "bbox" if cfg.network.bbox else f"ward union + {cfg.network.boundary_buffer_m} m",
        "osm_snapshot_date": cfg.network.osm_snapshot_date,
        "osmnx_version": ox.__version__,
        "python": platform.python_version(),
        "files_sha256": files,
    }
    meta_path = raw_dir / cfg.network.download_meta_file
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log.info("done: %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
