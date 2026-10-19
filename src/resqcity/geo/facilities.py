"""Police stations, fire stations, hospitals and ambulance bases, snapped to the road network.

Locations come from OpenStreetMap. Hospital capabilities (trauma, burns, ICU beds, free beds) are
SYNTHETIC: OSM has no reliable capability data for Pune. Every Hospital carries
``capabilities_synthetic = True``. Ambulance bases are ASSUMED (every hospital is a candidate
base; real 108 base locations are not public), so they carry ``source = synthetic``.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from resqcity.common.schemas import Agency, FacilitySource, GeoPoint, Hospital, Station
from resqcity.geo.config import FacilitiesConfig
from resqcity.geo.network import RoadNetwork

log = logging.getLogger(__name__)
WGS84 = "EPSG:4326"
FACILITY_COLUMNS = ["osm_type", "osm_id", "amenity", "name", "beds", "geometry"]
_PREFIX = {"police": "PS", "fire": "FS", "hospital": "H"}


@dataclass(frozen=True)
class Facilities:
    stations: list[Station]
    hospitals: list[Hospital]
    dropped: list[str]

    def by_agency(self, agency: Agency) -> list[Station]:
        return [s for s in self.stations if s.agency == agency]


def facility_kind(amenity: str, cfg: FacilitiesConfig) -> str | None:
    for kind, tags in (
        ("police", cfg.tags.police),
        ("fire", cfg.tags.fire),
        ("hospital", cfg.tags.hospital),
    ):
        if tags.get("amenity") == amenity:
            return kind
    return None


def _points_and_footprints(
    raw: gpd.GeoDataFrame, projected_crs: str
) -> tuple[gpd.GeoSeries, np.ndarray]:
    projected = raw.geometry.to_crs(projected_crs)
    points = projected.representative_point().to_crs(WGS84)
    return points, projected.area.to_numpy(float)


def _parse_beds(value: object) -> int | None:
    try:
        beds = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return beds if beds > 0 else None


def build_facilities(
    raw: gpd.GeoDataFrame, network: RoadNetwork, cfg: FacilitiesConfig
) -> Facilities:
    """Turn raw OSM features (columns ``FACILITY_COLUMNS``) into snapped Stations and Hospitals."""
    if raw.empty:
        return Facilities([], [], [])
    points, footprint = _points_and_footprints(raw, network.projected_crs)
    node_ids, dist = network.nearest_nodes(points.y.to_numpy(), points.x.to_numpy())
    ward_of = dict(zip(network.nodes["node_id"], network.nodes["ward_id"], strict=True))
    caps = cfg.hospital_capabilities
    rng = np.random.default_rng(caps.seed)

    stations: list[Station] = []
    hospitals: list[Hospital] = []
    dropped: list[str] = []
    order = np.lexsort((raw["osm_id"].to_numpy(), raw["osm_type"].astype(str).to_numpy()))
    for i in order:
        row = raw.iloc[i]
        kind = facility_kind(str(row["amenity"]), cfg)
        if kind is None:
            continue
        fid = f"{_PREFIX[kind]}-{str(row['osm_type'])[0]}{int(row['osm_id'])}"
        if dist[i] > cfg.max_snap_distance_m:
            dropped.append(f"{fid} ({dist[i]:.0f} m from nearest node)")
            continue
        name = (
            row["name"]
            if isinstance(row["name"], str) and row["name"].strip()
            else f"Unnamed {kind} {fid}"
        )
        location = GeoPoint(lat=float(points.iloc[i].y), lon=float(points.iloc[i].x))
        node = int(node_ids[i])
        ward = str(ward_of[node])
        if kind == "hospital":
            beds_tag = _parse_beds(row.get("beds"))
            major = footprint[i] >= caps.major_footprint_m2 or (
                beds_tag is not None and beds_tag >= 100
            )
            tier = "major" if major else "minor"
            icu_lo, icu_hi = getattr(caps.icu_beds, tier)
            bed_lo, bed_hi = getattr(caps.beds_available, tier)
            hospitals.append(
                Hospital(
                    id=fid,
                    name=name,
                    location=location,
                    node=node,
                    has_trauma=bool(rng.random() < getattr(caps.p_trauma, tier)),
                    has_burns=bool(rng.random() < getattr(caps.p_burns, tier)),
                    icu_beds=int(rng.integers(icu_lo, icu_hi + 1)),
                    beds_available=int(rng.integers(bed_lo, bed_hi + 1)),
                    capabilities_synthetic=True,
                )
            )
            stations.append(
                Station(
                    id=f"AB-{fid}",
                    name=f"Assumed ambulance base at {name}",
                    agency=Agency.AMBULANCE,
                    location=location,
                    node=node,
                    ward_id=ward,
                    source=FacilitySource.SYNTHETIC,
                )
            )
        else:
            stations.append(
                Station(
                    id=fid,
                    name=name,
                    agency=Agency.POLICE if kind == "police" else Agency.FIRE,
                    location=location,
                    node=node,
                    ward_id=ward,
                    source=FacilitySource.OSM,
                )
            )

    for base in cfg.extra_ambulance_bases:
        nid, d = network.nearest_nodes(base.lat, base.lon)
        stations.append(
            Station(
                id=base.id,
                name=base.name,
                agency=Agency.AMBULANCE,
                location=GeoPoint(lat=base.lat, lon=base.lon),
                node=int(nid[0]),
                ward_id=str(ward_of[int(nid[0])]),
                source=FacilitySource.SYNTHETIC,
            )
        )
        if d[0] > cfg.max_snap_distance_m:
            log.warning("extra base %s is %.0f m from the nearest node", base.id, d[0])
    for item in dropped:
        log.warning("dropped facility %s", item)
    return Facilities(stations, hospitals, dropped)


_STATIONS = TypeAdapter(list[Station])
_HOSPITALS = TypeAdapter(list[Hospital])


def save_facilities(facilities: Facilities, path: Path, meta: dict[str, object]) -> None:
    payload = {
        "meta": {**meta, "hospital_capabilities_synthetic": True},
        "stations": _STATIONS.dump_python(facilities.stations, mode="json"),
        "hospitals": _HOSPITALS.dump_python(facilities.hospitals, mode="json"),
        "dropped": facilities.dropped,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


def load_facilities(path: Path) -> Facilities:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Facilities(
        _STATIONS.validate_python(payload["stations"]),
        _HOSPITALS.validate_python(payload["hospitals"]),
        list(payload.get("dropped", [])),
    )


def facilities_frame(facilities: Facilities) -> pd.DataFrame:
    rows = [
        {
            "id": s.id,
            "kind": s.agency.value,
            "name": s.name,
            "lat": s.location.lat,
            "lon": s.location.lon,
            "node": s.node,
        }
        for s in facilities.stations
    ]
    rows += [
        {
            "id": h.id,
            "kind": "hospital",
            "name": h.name,
            "lat": h.location.lat,
            "lon": h.location.lon,
            "node": h.node,
        }
        for h in facilities.hospitals
    ]
    return pd.DataFrame(rows)
