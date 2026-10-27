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
| Pune rush-hour speed (superseded calibration target) | Punekar News, 2026-01-22, reporting TomTom Traffic Index 2025 | press article | 2026-10-20 | speed calibration D-014 (now baseline only; D-026 uses Google Maps) | real aggregate, secondary source | n/a |
| PMC fire call statistics 2011–2020 + fire-station map | [OpenCity / Pune Municipal Corporation](https://data.opencity.in/dataset/pune-fire-stations) | Public domain (per portal) | 2026-10-20 | base incident rate (D-018); fire-station count check | real | yes |
| MoRTH "Road Accidents in India 2022" (Table 6.2 Pune, Annexure 42 Maharashtra time of day, Annexure 45 Pune weather) | Ministry of Road Transport & Highways, Transport Research Wing (Oct 2023); PDF provided by the user, SHA-256 in `results/calibration/morth_2022_extract.json` | Government of India publication | 2026-10-20 | base incident rate, hour profile (D-026) | real aggregates (police-registered) | public report; PDF kept out of git |
| Google Maps typical travel times, 10 Pune routes × 3 departure times | Collected manually by the user (`results/geo/maps_check_google_2026-10-20.csv`) | user-collected observations | 2026-10-20 | speed calibration (D-026) | real (aggregate estimates by Google) | yes (our own observations) |
| Superseded: Pune road deaths 2023 (Punekar News) and national severity index (MoRTH via Outlook Business) | press articles | — | 2026-10-20 | former base-rate proxy (D-018/D-025) | — | — |
| UCF-Crime (RoadAccidents + 60 normal test videos, temporal annotations) | Sultani, Chen & Shah (CVPR 2018); HF mirror [jinmang2/ucf_crime](https://huggingface.co/datasets/jinmang2/ucf_crime) | research use (CRCV, UCF); mirror lists CC0 | 2026-10-20 | accident detector training/evaluation; demo camera clips | real CCTV footage (public) | do not redistribute; download script only |
| CADP annotations (frames not obtained) | Shah et al. 2018, [project page](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | non-commercial research use | 2026-10-20 | (pending frames; D-020) | real | download script only |
| D-Fire | Venâncio et al. 2022, [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset); HF mirror badsaarow/d-fire | per original providers (see README) | 2026-10-20 | fire/smoke detector fine-tuning | real images | download script only |
| ACCIDENT real clips (2,027 fixed-view CCTV crashes; impact time, box, collision type; official IID split) | Picek et al. (CVPR 2026 W), [Kaggle picekl/accident](https://www.kaggle.com/datasets/picekl/accident), [paper](https://arxiv.org/abs/2604.09819) | CC BY-NC-SA 4.0 (Kaggle; annotations CC BY 4.0 per paper) | 2026-10-22 | phase 3b: training (official train split) and the never-seen evaluation (official test split, sealed labels) (D-027) | real CCTV footage (public sources) | no (download script only; Kaggle account needed) |
| UCA: UCF-Crime event timestamps and sentences | Yuan et al. (CVPR 2024), [GitHub](https://github.com/Xuange923/Surveillance-Video-Understanding) at commit e478cb5 | Apache-2.0, academic research only | 2026-10-21 | accident intervals for UCF-Crime TRAINING videos (D-027) | real annotations | yes (research) |
| UVH-26 YOLO11-S detector (14 Indian vehicle classes incl. auto-rickshaw) | IISc AIM, [HF iisc-aim/UVH-26](https://huggingface.co/iisc-aim/UVH-26), [paper](https://arxiv.org/abs/2511.02563) | Apache-2.0 (models), CC BY 4.0 (data) | 2026-10-22 | vehicle detection for heuristics, stage C (D-027) | pretrained model (Bengaluru CCTV) | per licence |
| SigLIP2-base, V-JEPA 2 ViT-L (SSv2) | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224), [facebook/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | Apache-2.0, MIT | 2026-10-22 | frozen backbones for the window classifier, stage A (D-027) | pretrained models | per licence |
| Verifier VLMs: Qwen3.5-2B, Qwen3-VL-2B, Gemma-4-E2B, HolmesVAU-2B, Molmo2-4B, TAU-R1-Classifier | Hugging Face, revisions pinned in `configs/vision.yaml` | Apache-2.0 (Qwen, Gemma, Molmo2), MIT (HolmesVAU, TAU-R1) | 2026-10-22 | offline second-stage alarm check, stage B (D-027) | pretrained models (run locally) | per licence |
| YOLO11n weights, MobileNetV3-small ImageNet weights, OpenCV YuNet face model | Ultralytics (AGPL-3.0); torchvision (BSD); OpenCV Zoo (MIT) | as listed | 2026-10-20 | detection, frame features, face blurring | pretrained models | per licence |
| LightGBM 4.7.0 (gradient boosting, Poisson objective) | [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | MIT | 2026-10-26 | ward-hour demand forecaster (D-030) | library (no pretrained weights) | per licence |
| OR-Tools 9.15.6755 (CP-SAT) | [google/or-tools](https://github.com/google/or-tools) | Apache-2.0 | 2026-10-26 | dispatch, pre-positioning and camera-placement solvers (D-031) | library | per licence |
| Synthetic history for the forecaster (3 years of rain and festivals) | `resqcity.sim.make_history` + `configs/forecasting.yaml` -> `history` | n/a | generated | phase-5 training data | **synthetic**; rainfall profile UNVERIFIED, festival dates ILLUSTRATIVE | n/a |
| Synthetic incidents | `resqcity.sim.generator` + `configs/incidents.yaml` | n/a | generated | ground truth for the simulator | **synthetic** | n/a |
| Scenario weather / festival timings | `configs/scenarios/*.yaml` | n/a | authored | monsoon and Ganeshotsav scenarios | **synthetic, illustrative** | n/a |
| Synthetic 112 calls and caller texts (en/hi/mr) | `resqcity.sim.calls` + `configs/calls.yaml`, `configs/call_templates.yaml` (authored) | n/a | generated | call process, triage check | **synthetic** | n/a |
| Pune Dial 112 call volume (23,887 calls, July 2026) | [Free Press Journal, 2026-08-03](https://freepressjournal.in/pune/pune-city-police-rank-29th-in-maharashtras-dial-112-emergency-response-sangli-tops-state-list) | press article (citation only) | 2026-10-24 (read) | context for the false-call assumption (D-028) | real (reported figure) | citation only |
| Android Emergency Location Service accuracy ("up to 50 m"; India launch, UP only) | press reports, December 2025, e.g. [The News Minute](https://www.thenewsminute.com/news/google-activates-android-emergency-location-service-in-india) | press article (citation only) | 2026-10-24 (read) | caller-location scenario `els` | real (reported figure) | citation only |

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

### UCF-Crime (vision, phase 3)
- Citation: W. Sultani, C. Chen, M. Shah, "Real-world Anomaly Detection in Surveillance Videos",
  CVPR 2018. Mirror: Hugging Face `jinmang2/ucf_crime` (original zips + split files).
- Used: all 150 `RoadAccidents` videos; 60 seeded-random `Testing_Normal_Videos` (59 readable);
  `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` (23 accident onsets).
- Download: `uv run python -m resqcity.vision.datasets.download --only ucf_crime` (range reads
  of the zip members, with CRC checks; SHA-256 in `data/raw/vision/manifest.json`).
- Known issues: 320×240, mixed countries; "Normal" videos are mostly not road scenes; only
  23 onsets; possible near-duplicates (L16). Terms: research use; not redistributed.

### D-Fire (vision, phase 3)
- Citation: P. V. A. B. de Venâncio et al., Neural Computing and Applications, 2022. Mirror:
  Hugging Face `badsaarow/d-fire`. Class ids inferred from README box totals (exact match,
  `data/processed/vision/dfire_yolo/dataset_check.json`).
- Known issues: mostly wildfire/building scenes, few vehicle fires on Indian roads (L13).

### CADP (vision, not available)
- Annotation JSON downloaded; frames blocked by Google Drive for automated download (D-020) and
  not available to the user either (D-026). Not used.

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
