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
| Camera sites (40 junctions) | `resqcity.geo.cameras` over the OSM network | derived (ODbL) | generated | simulated CCTV locations | **algorithm-chosen**, not real CCTV sites | yes |
| Pune rush-hour speed (calibration target) | Punekar News, 2026-01-22, reporting TomTom Traffic Index 2025 (33 min 20 s per 10 km; 15.1 km/h rush hour) | press article | 2026-10-20 | speed calibration (D-014) | real aggregate, secondary source | n/a |
| PMC fire call statistics 2011–2020 + fire-station map | [OpenCity / Pune Municipal Corporation](https://data.opencity.in/dataset/pune-fire-stations) | Public domain (per portal) | 2026-10-20 | base incident rate (D-018); fire-station count check | real | yes |
| Pune road deaths 2022–2023 | Punekar News, 2024-01-13, citing Pune city police | press article | 2026-10-20 | base incident rate (D-018) | real aggregate, secondary source | n/a |
| MoRTH severity index 2022 (36.5 deaths/100 accidents) | MoRTH "Road Accidents in India 2022", via Outlook Business | government report (via press) | 2026-10-20 | base incident rate (D-018) | real aggregate | n/a |
| UCF-Crime (RoadAccidents + 60 normal test videos, temporal annotations) | Sultani, Chen & Shah (CVPR 2018); HF mirror [jinmang2/ucf_crime](https://huggingface.co/datasets/jinmang2/ucf_crime) | research use (CRCV, UCF); mirror lists CC0 | 2026-10-20 | accident detector training/evaluation; demo camera clips | real CCTV footage (public) | do not redistribute; download script only |
| CADP annotations (frames not obtained) | Shah et al. 2018, [project page](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | non-commercial research use | 2026-10-20 | (pending frames; D-020) | real | download script only |
| D-Fire | Venâncio et al. 2022, [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset); HF mirror badsaarow/d-fire | per original providers (see README) | 2026-10-20 | fire/smoke detector fine-tuning | real images | download script only |
| YOLO11n weights, MobileNetV3-small ImageNet weights, OpenCV YuNet face model | Ultralytics (AGPL-3.0); torchvision (BSD); OpenCV Zoo (MIT) | as listed | 2026-10-20 | detection, frame features, face blurring | pretrained models | per licence |
| Synthetic incidents | `resqcity.sim.generator` + `configs/incidents.yaml` | n/a | generated | ground truth for the simulator | **synthetic** | n/a |
| Scenario weather / festival timings | `configs/scenarios/*.yaml` | n/a | authored | monsoon and Ganeshotsav scenarios | **synthetic, illustrative** | n/a |

Exact download time, OSMnx version and SHA-256 of every raw file: `data/raw/download_meta.json`
(written by `python -m resqcity.geo.download`).

## Notes per dataset

### PMC 2022 electoral wards
- Citation: DataMeet Pune chapter, "Pune electoral wards 2022", Municipal_Spatial_Data repository;
  traced by SeerMaps Technologies (commissioned by Nikhil VJ) from the PMC final prabhag
  rachna 2022 PDFs.
- Download: `uv run python -m resqcity.geo.download` (URL in `configs/geo.yaml`).
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
