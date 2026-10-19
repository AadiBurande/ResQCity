"""Typed models for ``configs/geo.yaml``."""

from pathlib import Path
from typing import Generic, Literal, Self, TypeVar

from pydantic import Field, model_validator

from common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model

T = TypeVar("T")


class BBox(ConfigModel):
    """Geographic bounding box in degrees."""

    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (self.west < self.east and self.south < self.north):
            raise ValueError("bbox must satisfy west < east and south < north")
        return self


class WardsConfig(ConfigModel):
    method: Literal["geojson", "h3"]
    url: str
    raw_file: Path
    id_field: str
    name_field: str
    id_prefix: str
    h3_resolution: int = Field(ge=0, le=15)


class NetworkConfig(ConfigModel):
    boundary_buffer_m: float = Field(ge=0)
    bbox: BBox | None = None
    network_type: str
    osm_snapshot_date: str | None = None
    projected_crs: str
    osm_cache_dir: Path
    raw_graphml_file: Path
    graphml_file: Path
    nodes_file: Path
    edges_file: Path
    wards_file: Path
    facilities_file: Path
    download_meta_file: Path
    build_meta_file: Path
    river_sample_step_m: float = Field(gt=0)
    min_edge_time_s: float = Field(gt=0)


class RoadClassesConfig(ConfigModel):
    default_class: str
    priority: list[str] = Field(min_length=1)
    mapping: dict[str, str]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        known = set(self.priority)
        if len(known) != len(self.priority):
            raise ValueError("road class priority has duplicates")
        unknown = ({self.default_class} | set(self.mapping.values())) - known
        if unknown:
            raise ValueError(f"road classes not listed in priority: {sorted(unknown)}")
        return self


class TrafficPeriod(ConfigModel):
    name: str
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=24)
    multipliers: dict[str, SourcedValue[float]]

    def hours(self) -> set[int]:
        if self.start_hour < self.end_hour:
            return set(range(self.start_hour, self.end_hour))
        return set(range(self.start_hour, 24)) | set(range(0, self.end_hour))


class RainSpeedBand(ConfigModel):
    min_mm_h: float = Field(ge=0)
    multiplier: SourcedValue[float]


class RiversConfig(ConfigModel):
    names: list[str] = Field(min_length=1)
    raw_file: Path


class FacilityTags(ConfigModel):
    police: dict[str, str]
    fire: dict[str, str]
    hospital: dict[str, str]


class ExtraBase(ConfigModel):
    id: str
    name: str
    lat: float
    lon: float


class ByTier(ConfigModel, Generic[T]):
    major: T
    minor: T


class HospitalCapabilities(ConfigModel):
    seed: int
    major_footprint_m2: float = Field(ge=0)
    p_trauma: ByTier[float]
    p_burns: ByTier[float]
    icu_beds: ByTier[tuple[int, int]]
    beds_available: ByTier[tuple[int, int]]


class FacilitiesConfig(ConfigModel):
    raw_file: Path
    tags: FacilityTags
    max_snap_distance_m: float = Field(gt=0)
    extra_ambulance_bases: list[ExtraBase]
    hospital_capabilities: HospitalCapabilities


class CameraSitesConfig(ConfigModel):
    n: int = Field(ge=1)
    road_classes: list[str] = Field(min_length=1)
    min_degree: int = Field(ge=1)
    weight_degree: float = Field(ge=0)
    weight_betweenness: float = Field(ge=0)
    betweenness_samples: int = Field(ge=1)
    min_spacing_m: float = Field(ge=0)
    seed: int
    output_file: Path


class GeoConfig(ConfigModel):
    wards: WardsConfig
    network: NetworkConfig
    road_classes: RoadClassesConfig
    base_speeds_kph: dict[str, SourcedValue[float]]
    traffic_periods: list[TrafficPeriod]
    rain_speed_bands: list[RainSpeedBand]
    vehicle_profiles: dict[str, SourcedValue[float]]
    rivers: RiversConfig
    facilities: FacilitiesConfig
    camera_sites: CameraSitesConfig

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        classes = set(self.road_classes.priority)
        if set(self.base_speeds_kph) != classes:
            raise ValueError("base_speeds_kph must define exactly the road classes in priority")
        if any(v.value <= 0 for v in self.base_speeds_kph.values()):
            raise ValueError("base speeds must be positive")
        seen: set[int] = set()
        for period in self.traffic_periods:
            if set(period.multipliers) != classes:
                raise ValueError(
                    f"traffic period {period.name}: multipliers must cover all classes"
                )
            if seen & period.hours():
                raise ValueError(f"traffic period {period.name} overlaps another period")
            seen |= period.hours()
        mins = [b.min_mm_h for b in self.rain_speed_bands]
        if mins != sorted(mins):
            raise ValueError("rain_speed_bands must be ordered by min_mm_h")
        if "default" not in self.vehicle_profiles:
            raise ValueError("vehicle_profiles must define 'default'")
        if not set(self.camera_sites.road_classes) <= classes:
            raise ValueError("camera_sites.road_classes must be known road classes")
        if self.wards.method == "h3" and self.network.bbox is None:
            raise ValueError("wards.method = h3 needs network.bbox (there is no ward boundary)")
        return self


def load_geo_config(path: Path | None = None) -> GeoConfig:
    return load_model(GeoConfig, path or CONFIG_DIR / "geo.yaml")
