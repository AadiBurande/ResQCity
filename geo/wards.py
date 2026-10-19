"""Wards: PMC 2022 electoral wards (DataMeet) or an H3 hexagon fallback, and node assignment."""

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from geo.config import GeoConfig, WardsConfig

WGS84 = "EPSG:4326"


def load_wards_geojson(path: Path, cfg: WardsConfig) -> gpd.GeoDataFrame:
    raw = gpd.read_file(path).to_crs(WGS84)
    ids = raw[cfg.id_field].astype(int)
    width = max(2, len(str(ids.max())))
    wards = gpd.GeoDataFrame(
        {
            "ward_id": [f"{cfg.id_prefix}{i:0{width}d}" for i in ids],
            "name": raw[cfg.name_field].astype(str),
        },
        geometry=raw.geometry.make_valid(),
        crs=WGS84,
    )
    if wards["ward_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ward ids")
    return wards.sort_values("ward_id", ignore_index=True)


def h3_wards(boundary: BaseGeometry, resolution: int) -> gpd.GeoDataFrame:
    """Hexagon wards covering ``boundary`` (WGS84). Fallback when no ward GeoJSON is available."""
    cells = sorted(h3.geo_to_cells(boundary.__geo_interface__, resolution))
    polys = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"ward_id": [f"H3-{c}" for c in cells], "name": cells}, geometry=polys, crs=WGS84
    )


def load_wards(raw_dir: Path, cfg: GeoConfig) -> gpd.GeoDataFrame:
    """Wards per ``cfg.wards.method``; the H3 fallback covers ``network.bbox``."""
    if cfg.wards.method == "geojson":
        return load_wards_geojson(raw_dir / cfg.wards.raw_file, cfg.wards)
    b = cfg.network.bbox
    if b is None:  # guarded by GeoConfig validation
        raise ValueError("h3 wards need network.bbox")
    return h3_wards(box(b.west, b.south, b.east, b.north), cfg.wards.h3_resolution)


def study_boundary(wards: gpd.GeoDataFrame) -> BaseGeometry:
    """Union of all wards (WGS84)."""
    return wards.union_all()


def buffered(geom: BaseGeometry, metres: float, projected_crs: str) -> BaseGeometry:
    series = gpd.GeoSeries([geom], crs=WGS84).to_crs(projected_crs).buffer(metres)
    return series.to_crs(WGS84).iloc[0]


def assign_nodes(
    lat: np.ndarray, lon: np.ndarray, wards: gpd.GeoDataFrame, projected_crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """(ward_id, in_city) per point. Points outside every ward take the nearest ward's id."""
    points = gpd.GeoDataFrame(
        {"i": np.arange(len(lat))}, geometry=gpd.points_from_xy(lon, lat), crs=WGS84
    ).to_crs(projected_crs)
    polys = wards[["ward_id", "geometry"]].to_crs(projected_crs)
    inside = gpd.sjoin(points, polys, how="left", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")]  # border points: first ward wins
    ward = inside["ward_id"].reindex(points.index)
    in_city = ward.notna().to_numpy()
    if (~in_city).any():
        outside = points[~in_city]
        nearest = gpd.sjoin_nearest(outside, polys, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        ward.loc[outside.index] = nearest["ward_id"]
    return ward.astype(str).to_numpy(), in_city


def ward_table(wards: gpd.GeoDataFrame, projected_crs: str) -> pd.DataFrame:
    """Ward id, name and area (km²), without geometry."""
    area = wards.to_crs(projected_crs).area / 1e6
    return pd.DataFrame({"ward_id": wards["ward_id"], "name": wards["name"], "area_km2": area})
