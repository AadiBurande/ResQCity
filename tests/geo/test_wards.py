"""Ward loading and node assignment."""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from resqcity.geo.config import WardsConfig
from resqcity.geo.wards import assign_nodes, h3_wards, load_wards_geojson, study_boundary

CFG = WardsConfig(
    method="geojson",
    url="https://example.invalid/w.geojson",
    raw_file="w.geojson",
    id_field="wardnum",
    name_field="Name2",
    id_prefix="W",
    h3_resolution=7,
)


def test_geojson_ids_are_prefixed_and_padded(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"wardnum": [2, 1], "Name2": ["Kasba", "Dhanori"]},
        geometry=[box(73.85, 18.50, 73.86, 18.51), box(73.86, 18.50, 73.87, 18.51)],
        crs="EPSG:4326",
    )
    path = tmp_path / "w.geojson"
    gdf.to_file(path, driver="GeoJSON")
    wards = load_wards_geojson(path, CFG)
    assert list(wards["ward_id"]) == ["W01", "W02"]
    assert list(wards["name"]) == ["Dhanori", "Kasba"]
    assert study_boundary(wards).area > 0


def test_assign_nodes_inside_and_nearest(toy_wards):
    lat = np.array([18.502, 18.502, 18.502])
    lon = np.array([73.851, 73.856, 73.870])  # W01, W02, far east (outside)
    ward, in_city = assign_nodes(lat, lon, toy_wards, "EPSG:32643")
    assert list(ward) == ["W01", "W02", "W02"]
    assert list(in_city) == [True, True, False]


def test_h3_fallback_covers_boundary():
    boundary = box(73.84, 18.50, 73.88, 18.53)
    wards = h3_wards(boundary, 7)
    assert len(wards) >= 2
    assert wards["ward_id"].str.startswith("H3-").all()
    assert wards.union_all().contains(Point(73.86, 18.515))
