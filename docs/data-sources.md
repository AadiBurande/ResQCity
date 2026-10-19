# Data sources

Every dataset used by ResQCity, with licence, access date and how it is used. If a licence
forbids redistribution, only the download script is committed (never the data).
Synthetic data is listed too, so readers can see which inputs are real.

| Dataset | Source / URL | Licence | Downloaded on | Used for | Real or synthetic | Redistributable? |
|---|---|---|---|---|---|---|
| PMC 2022 electoral wards (58) | [DataMeet Municipal_Spatial_Data/Pune](https://github.com/datameet/Municipal_Spatial_Data/tree/master/Pune), traced from [PMC prabhag rachna 2022](https://www.pmc.gov.in/en/pmc-final-prabhag-rachna-2022) | CC BY-SA 2.5 IN | 2026-10-20 | wards, study boundary | real (manual trace) | yes, share-alike (kept out of git anyway) |
| OSM drivable road network | OpenStreetMap via Overpass (OSMnx 2.1.1) | ODbL 1.0 | 2026-10-20 | road graph, travel times | real | yes, with attribution (kept out of git: size) |
| OSM rivers (Mula, Mutha, Mula-Mutha) | OpenStreetMap `waterway=river` | ODbL 1.0 | 2026-10-20 | flood-prone edges | real | yes, with attribution |
| OSM police / fire / hospital locations | OpenStreetMap `amenity=police/fire_station/hospital` | ODbL 1.0 | 2026-10-20 | stations, hospitals, ambulance bases | real locations; **synthetic** capabilities | yes, with attribution |
| Camera sites (40 junctions) | `geo/cameras.py` over the OSM network | derived (ODbL) | generated | simulated CCTV locations | **algorithm-chosen**, not real CCTV sites | yes |
| Pune rush-hour speed (calibration target) | Punekar News, 2026-01-22, reporting TomTom Traffic Index 2025 (33 min 20 s per 10 km; 15.1 km/h rush hour) | press article | 2026-10-20 | speed calibration (D-014) | real aggregate, secondary source | n/a |
| Synthetic incidents | `sim/generator.py` + `configs/incidents.yaml` | n/a | generated | ground truth for the simulator | **synthetic** | n/a |
| Scenario weather / festival timings | `configs/scenarios/*.yaml` | n/a | authored | monsoon and Ganeshotsav scenarios | **synthetic, illustrative** | n/a |

Exact download time, OSMnx version and SHA-256 of every raw file: `data/raw/download_meta.json`
(written by `python -m geo.download`).

## Notes per dataset

### PMC 2022 electoral wards
- Citation: DataMeet Pune chapter, "Pune electoral wards 2022", Municipal_Spatial_Data repository;
  traced by SeerMaps Technologies (commissioned by Nikhil VJ) from the PMC final prabhag
  rachna 2022 PDFs.
- Download: `uv run python -m geo.download` (URL in `configs/geo.yaml`).
- Pre-processing: ids `W01..W58` from `wardnum`; geometries `make_valid`; the union is the study
  boundary (506.9 km²).
- Known issues: hand-traced from PDF screenshots, so boundaries are approximate.

### OpenStreetMap extracts
- Download area: ward union buffered by 1 km. Drivable network (`network_type=drive`),
  simplified; the largest strongly connected component is kept.
- Known issues: speed tags sparse (D-008); many small clinics tagged `amenity=hospital` (D-013);
  completeness of police posts varies.

<!--
### <Dataset name>
- Citation:
- Version / snapshot:
- Download command or script:
- Pre-processing applied:
- Known issues / biases:
-->

## Required attributions

- Map data © OpenStreetMap contributors, available under the Open Database License (ODbL).
- Ward boundaries © DataMeet Trust (Pune chapter), CC BY-SA 2.5 India.
