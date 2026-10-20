# Phase 3b — Detector upgrade with never-seen evaluation footage

Status: PLAN (draft pre-registration, written before any ACCIDENT clip or label was downloaded)
· Date: 2026-10-21 · Skill: `vision-eval` · Follows D-026 · Awaiting user inputs (section 9)

## 1. Decision: do this now, before phase 4
Later phases cannot fix the detector. Fusion with 112 calls (phase 4) and confidence-aware dispatch
(phase 7) can limit the harm of false alarms, but they cannot create detections. The research answer
("does CCTV-triggered dispatch help?") depends on a credible detector operating point, and v4 is
measured on 23 onset videos and 4.5 minutes of road footage, none of it truly unseen (L11, L15).
The interface to later phases (`sim_inputs.yaml`, `Event`) stays the same. Changing `Event` (an
optional verifier result and description) is cheapest now, before fusion and the dashboard consume
it. This phase is time-boxed (section 10). If the detector stays weak, that is reported, and the
study proceeds with the phase-10 break-even analysis.

## 2. Why v4 is weak (diagnosis)
1. Labels: training accident videos have only video-level labels (MIL on ~100 videos).
2. Backbone: ImageNet MobileNetV3 frame features are not about collisions; the Kinetics video
   backbone did not help (D-026).
3. Data: one source (UCF-Crime, 320×240), no Indian scenes, no never-seen evaluation footage.
4. Measurement: road false alarms are estimated from 0.075 h of footage (CI ≈ 22–156 per hour).

## 3. Resources checked (2026-10-21; access, licence and size verified at the source)
### Datasets
| Resource | What it gives | Access · licence · size | Use |
|---|---|---|---|
| ACCIDENT (Picek et al., CVPR 2026 W; [paper](https://arxiv.org/abs/2604.09819), [site](https://accidentbench.github.io), [code](https://github.com/accidentbench/ACCIDENT)) | 2,027 real fixed-view CCTV crash clips (≈20 % outside the US, incl. India), impact time, box, collision type; 2,211 synthetic CARLA clips | Kaggle `picekl/accident` (free account + API token) · CC BY-NC-SA 4.0 on Kaggle (paper: annotations CC BY 4.0) · real clips 4.92 GB, synthetic 15.24 GB | **primary never-seen evaluation**; IID train split for training |
| UCA (Yuan et al., CVPR 2024; [repo](https://github.com/Xuange923/Surveillance-Video-Understanding)) | time-stamped event sentences (0.1 s) for 1,854 UCF-Crime videos, incl. 148 of our 150 RoadAccidents videos | GitHub · Apache-2.0, academic research only · 3 MB | turn weak labels into accident intervals for training videos |
| UVH-26 (IISc Bengaluru; [paper](https://arxiv.org/abs/2511.02563), [data](https://huggingface.co/datasets/iisc-aim/UVH-26)) | 26,646 Bengaluru traffic-CCTV images, 14 Indian vehicle classes incl. auto-rickshaw | Hugging Face · CC BY 4.0 (data), Apache-2.0 (models) | Indian vehicle detector (model only, below) |
| IITH-Accident ([page](https://sites.google.com/site/dineshsinghindian/iith_accident-dataset)) | real accidents from Hyderabad city CCTV, 30 fps | download page dead; request by e-mail to the authors | Indian test set, if the user obtains it |
| UA-DETRAC (HF mirror `abhineet123/ua_detrac`) | ~10 h of normal road CCTV (China) | no account · mirror says CC BY 4.0 (original terms non-commercial) · 9.9 GB | optional: false alarms in dense traffic |
| XD-Violence (HF mirror `jherng/xd-violence`) | 550 "car accident" videos, mostly film/YouTube | no account · research use · 5.0 GB for those videos | optional, low priority (moving cameras) |

Rejected: dashcam sets (DoTA, CCD, VRU-Accident, CrashChat; moving camera), AccidentBench (QA
benchmark), TU-DAT (YOLOv8-generated annotations, mixed simulated footage), TAD-Lv (frames only,
split and labels never released), CC0 Roboflow accident images (unknown origin; usable only after a
near-duplicate check against all evaluation videos).

### Pretrained models (all run offline; sizes are the published weights)
| Model | Role | Licence · size | Fit on RTX 4050 (6 GB, ~4.8 GB free) |
|---|---|---|---|
| UVH-26 YOLO11-S ([HF](https://huggingface.co/iisc-aim/UVH-26)) | vehicles incl. auto-rickshaw (no person class: keep COCO YOLO11n for people) | Apache-2.0 · 57 MB | yes, also CPU |
| SigLIP2-base-patch16-224 ([HF](https://huggingface.co/google/siglip2-base-patch16-224)) | frame embeddings + zero-shot text cues ("a car crash", "overturned vehicle") | Apache-2.0 · 375M params, 1.5 GB | yes |
| Qwen3.5-2B ([HF](https://huggingface.co/Qwen/Qwen3.5-2B)) | verifier VLM (image + video) and offline text LLM, 201 languages | Apache-2.0 · 2.27B, 4.55 GB bf16 | 8-bit or 4-bit |
| Qwen3-VL-2B-Instruct ([HF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) | verifier VLM with timestamp grounding | Apache-2.0 · 2.13B, 4.26 GB bf16; official GGUF | 8-bit or GGUF |
| Gemma-4-E2B-it ([HF](https://huggingface.co/google/gemma-4-E2B-it)) | verifier VLM (image, video, audio) | Apache-2.0 · 5.1B incl. embeddings (2.3B effective); official QAT q4_0 GGUF | 4-bit |
| HolmesVAU-2B ([HF](https://huggingface.co/ppxin321/HolmesVAU-2B), [code](https://github.com/pipixin321/HolmesVAU)) | anomaly-specialised VLM (InternVL2-2B), frame anomaly scores + text | MIT · 4.41 GB; repo states < 9 GB for inference | 8-bit; trained on the official UCF-Crime TRAIN split, so it may be evaluated only on official-test videos and ACCIDENT |
| sarvam-translate ([HF](https://huggingface.co/sarvamai/sarvam-translate)) | Marathi/Hindi ↔ English (phase 4, optional) | GPL-3.0 · 4.3B, 8.6 GB bf16 | 4-bit |

Not used: community VideoMAE "UCF-Crime" fine-tunes (unknown splits, likely trained on our test
videos), the YOLO11x accident model (trained on Roboflow images of unknown origin), 7B+ VLMs
(do not fit comfortably). Expectation management: in the ACCIDENT paper, zero-shot 7B VLMs locate
the impact time hardly better than optical-flow heuristics, so a VLM is used here as a VERIFIER of
candidate alarms and as a describer, not as the detector.

## 4. Pipeline changes
- A. **Labels:** UCA accident intervals for UCF training videos. Accident sentences are found by a
  fixed keyword rule, cross-checked by the offline LLM. Agreement with the 23 official onsets is
  reported (`results/vision/uca_label_check.json`) and never used to tune anything.
- B. **Data:** ACCIDENT real clips; impact time t* gives window labels.
- C. **Backbone:** SigLIP2 frame embeddings + zero-shot text-similarity features → temporal head
  trained with interval labels (no MIL), with the v4 MIL head kept as a candidate.
- D. **Vehicles:** UVH-26 YOLO11-S for vehicles and COCO YOLO11n for people (heuristics, severity).
- E. **Verifier:** when an alarm fires, 8 anonymised frames from the last 4 s go to the VLM with a
  fixed prompt: "Does this CCTV clip show a traffic accident (a collision involving a vehicle, or a
  vehicle overturning)? Answer yes or no." The yes-probability is the verifier score, and the
  model's one-line description becomes optional `Event` metadata. Frames never leave `vision/`.
- F. **Simulator interface:** `sim_inputs.yaml` v5 from the evaluation below; `Event` gains optional
  `verified` and `description` fields (TypeScript types regenerated).

## 5. Evaluation v5 (pre-registered)
**Data roles.** ACCIDENT IID test clips (1,520, official split) are touched once, in the final run.
ACCIDENT IID train (507) and UCF train/val are used for fitting and selection. Clips sharing a
source-video id (file-name prefix) stay in the same fold. If the download does not carry the
official split, a seeded grouped split is created before any label is read, and this is disclosed.
UCF's 23 onset videos keep the v4 cross-fitting for comparability.

**Rules on ACCIDENT** (t* = annotated impact time; alarms need the warm-up of window + persistence):
- detection = first alarm at t ∈ [t* − 1 s, t* + 10 s] (−1 s for annotation uncertainty; +10 s
  = `DETECTION_GRACE_S`);
- false alarm = alarm before t* − 1 s; road footage = Σ max(0, t* − 1 s − warm-up) over clips;
- late alarm = after t* + 10 s (counted separately, neither detection nor false alarm).

**Primary endpoints (ACCIDENT IID test, operating point fixed beforehand):**
E1 detection rate (Wilson 95 % CI); E2 road false alarms per camera-hour (Poisson and
clip-clustered bootstrap CIs); E3 median detection delay. **External benchmark:** ACCIDENT's
temporal score 𝒯 (σ = 0.5, 1, 2 s; time of the highest window score per clip), comparable with the
published baselines. Secondary: geo-OOD split; UCF 23 onsets (v4 cross-fitting); latency on this
laptop (CPU and GPU).

**Candidate pool (fixed now), staged selection on validation only:**
- Stage A: backbone {MobileNetV3 (v4), SigLIP2} × head {MIL, interval-supervised} × training data
  {UCF only, UCF + ACCIDENT-train} = 8 candidates; choose by validation Youden J (as v4).
- Stage B: verifier {none, Qwen3.5-2B, Qwen3-VL-2B, Gemma-4-E2B, HolmesVAU-2B}. Choose the largest
  false-alarm reduction that keeps ≥ 90 % of validation detections, with latency ≤ 3 s per check on
  the RTX 4050; HolmesVAU is validated on ACCIDENT-train folds only.
- Stage C: vehicle detector {YOLO11n COCO, UVH-26 YOLO11-S + COCO persons}, chosen by validation J.

No change after the test run except bug fixes, and every one is disclosed in `docs/decisions.md`.
Results are reported whatever they are; the simulator takes E1–E3 (with intervals) from the
deployed detector.

## 6. Offline LLM roles (privacy by design: nothing leaves the machine)
- Phase 3b: verifier and describer (section 4E).
- Phase 4: parse simulated 112 call transcripts (English, Hindi, Marathi; generated from templates
  with known ground truth) into `Call` fields. Extraction accuracy is measured on a held-out set, and
  the simulator uses that MEASURED confusion matrix rather than calling the LLM in each of the ≥ 30
  replications (which would be too slow on a laptop and is unnecessary for the statistics).
- Phase 8: optional paraphrase of structured explanations for operators, with an automatic
  faithfulness check (every number and unit in the text must appear in the structured explanation).

## 7. Phase-2 fixes that can run in parallel (need the user)
- **Speeds:** TomTom Routing API free tier (2,500 requests/day, no credit card;
  [pricing](https://developer.tomtom.com/pricing)) returns `historicTrafficTravelTimeInSeconds`
  for any departure time ([docs](https://docs.tomtom.com/routing-api/documentation/tomtom-maps/calculate-route)).
  Planned: ~60 seeded origin–destination pairs × 24 hours × 2 days (Tuesday, Saturday), a 30 %
  route hold-out fixed before querying, and the user's Google Maps sheet as an external check.
  TomTom's terms on storing results must be read before running.
- **Incident rate and fire hour profile:** only obtainable from PMC / Pune Traffic Police records
  (RTI request by the user); otherwise they stay sensitivity factors.

## 8. Compute and storage on this laptop
RTX 4050 Laptop 6 GB (about 1.2 GB used by other programs), Ryzen 7 7840HS, 16 GB RAM (about 5 GB
free while other Docker projects run), D: 18.5 GB free, G: 77 GB free. ACCIDENT real clips go to
`data/raw/vision/` (4.9 GB). Model weights (about 15 GB for the verifier candidates) go to a
Hugging Face cache on G: if the user agrees. GPU jobs run one at a time (D-024): feature extraction
≈ 2027 clips, YOLO tracking, and verifier checks on validation candidates only, overnight.

## 9. Inputs needed from the user
1. Kaggle account + API token (`kaggle.json`) for ACCIDENT; acceptance of CC BY-NC-SA 4.0
   (non-commercial research).
2. Permission to download about 20 GB of models and data, and to keep model weights on G:.
3. Whether the other Docker projects may be stopped during overnight runs (frees RAM).
4. Optional: TomTom developer key (section 7); an e-mail request to the IITH-Accident authors;
   whether phase-4 calls should include Hindi/Marathi transcripts.

## 10. Time box, acceptance and stop rule
Steps: UCA labels → ACCIDENT download and split audit → SigLIP2 features → stage A → stage B →
stage C → single test run → `sim_inputs.yaml` v5, deployed detector, live demo → write-up (D-027).
Acceptance: all endpoints reported with intervals, tests and lint pass, `make vision` reproduces
v5, and `research-auditor` reviews before closing. Stop rule: if stage A and B do not improve
validation J over v4, skip further tuning, run the test once with the best validation candidate,
and move to phase 4.
