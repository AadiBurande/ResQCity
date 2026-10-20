# Phase 3 — Simulated CCTV streams and the vision pipeline

Status: implemented (development evaluation; see Outcome) · Date: 2026-10-20 · Skill: `vision-eval` · Decisions D-018…D-025

## User decisions (2026-10-20)
Allowed: download public datasets (no account or sign-up, cap ≈ 10 GB), GPU training overnight on
the RTX 4050, start Docker and pull MediaMTX. Ultralytics YOLO (AGPL-3.0) accepted. Google Maps check:
the user fills in a template later. "MADP" turned out to be the Multiagent Decision Process
Toolbox (software, not video), so it is dropped as a vision dataset.

## Datasets (all under `data/raw/vision/`, gitignored; download scripts committed)
| Dataset | What we use | Why | Access |
|---|---|---|---|
| UCF-Crime | `RoadAccidents` class (real CCTV crashes) + temporal annotations of the test videos | detection delay needs onset times; fixed cameras | HF mirror of the original zips; only the RoadAccidents members are extracted via HTTP range reads |
| CADP | frames of accident clips + spatio-temporal annotations | second CCTV-style source (YouTube traffic cams) | Google Drive / direct link; non-commercial research terms |
| D-Fire | ≈21k fire/smoke images, YOLO labels | fine-tune the fire/smoke detector | HF mirror of gaiasd/DFireDataset |
| Normal traffic | pre-onset segments + normal clips | false alarms per camera-hour | from the above; a dedicated normal-traffic set if one is openly downloadable |
TAD (freeway CCTV) is used if a public release is found; IEEE DataPort needs an account (left to the user).

Unified annotation format: `data/processed/vision/annotations.parquet` with video_id, source,
path, fps, n_frames, duration_s, accident_onset_s (or null), type, split. Splits are by VIDEO,
stratified by source, seeded, and saved to `data/processed/vision/splits.csv`.

## Pipeline (`src/resqcity/vision/`)
1. `ingest`: one `FrameSource` for files, RTSP and HTTP (phone IP camera); sampling to N fps (default 5); reconnects on drop.
2. `privacy`: faces (OpenCV YuNet) and plate regions (lower band of each vehicle box), plus optional
   person pixelation. Applied before ANY frame is stored, streamed or shown. Verified on a saved sample
   (`results/vision/privacy_check.png`).
3. `detect_track`: YOLO (small, CPU-friendly) + ByteTrack. Output is per-frame `TrackObs`. A protocol
   allows a fake detector in tests.
4. Accident detection in two explainable stages:
   - `heuristics`: per-track kinematics give sudden deceleration, overlap-then-stop between vehicles,
     box-aspect flip (overturn proxy), and a person low and still on the carriageway. These are
     combined into a heuristic score with the components kept.
   - `clip_classifier`: frozen pretrained CNN features per sampled frame, temporal pooling over a
     2–4 s window, then a small MLP trained on GPU. Calibrated (temperature or isotonic) on the val split.
   - `scoring`: a stacked logistic combination fitted on val produces a calibrated `Event.confidence`,
     with `confidence_components` = {heuristic parts, classifier}.
5. `fire_smoke`: YOLO fine-tuned on D-Fire. Persistence over consecutive frames is required.
6. `severity`: tracks give `SeverityFeatures`, then `common.requirements` gives severity_score and required_services.
7. `alarm`: state machine (threshold, k consecutive windows, cooldown) emitting `common.schemas.Event`.
8. `run`: `python -m resqcity.vision.run --camera C01`. It reads `configs/cameras.yaml`, publishes
   Events to the bus (`resqcity.common.bus`: in-process + JSONL file; Redis/HTTP come in phase 8) and
   serves an optional blurred, annotated MJPEG preview.

## Streams (`streams/docker-compose.yml`, `src/resqcity/streams/`)
MediaMTX in Docker. `loop_cameras` runs one `ffmpeg -re -stream_loop -1 … -f rtsp` publisher per camera
from `data/videos/<camera_id>/`. `scenario_player` switches a camera from its normal clip to an
accident clip at configured times (`configs/streams.yaml`). Phone IP-camera URLs are supported as
`stream_url` overrides.

## Evaluation (`src/resqcity/vision/eval/`, test split only)
Per dataset and overall: clip-level precision/recall/F1/AUROC and a reliability diagram; detection
delay (onset to first alarm) per threshold; false alarms per camera-hour on normal footage; CPU
latency (ms/frame) and fps. Written to `results/vision/<run>/` with the model hash. Detector behaviour
for the simulator goes to `results/vision/sim_inputs.yaml`.

## Training
`train_fire` (YOLO fine-tune) and `train_clip` (features, then MLP) run locally on GPU.
`notebooks/vision/train.ipynb` reproduces both on Colab.

## Acceptance
- `make streams` serves ≥ 4 RTSP cameras.
- `python -m resqcity.vision.run --camera C01` emits Events when an accident clip plays.
- Metrics report on the test split. Privacy check image saved.
- Unit tests use synthetic videos and fake detectors (no network, no GPU).

## Outcome (2026-10-20)
- Streams: 4 RTSP cameras served by MediaMTX (localhost only), scenario cues, live acceptance runner.
- Vision: pipeline runs in real time on CPU (`results/vision/latency_cpu.json`). Privacy blur is
  verified on a real frame (mechanism check; coverage needs hand labels).
- Accident detector (UCF-Crime development-test, v3, D-025): onset detection 3/12, road false alarms
  ~126/h with wide intervals, over-confident above its threshold. It is weak, and this is reported
  as such. Detector quality is a phase-10 sensitivity factor (break-even analysis).
- Fire/smoke detector (D-Fire test split): mAP50 0.76 (`results/vision/fire_main/fire_metrics.json`).
- Datasets actually used: UCF-Crime, D-Fire (CADP frames, TAD and IEEE DataPort not obtained; D-020).
- Open actions: CADP frames by hand (held-out evaluation); hand-labelled privacy coverage check;
  Pune-specific footage if the college letter yields data.
