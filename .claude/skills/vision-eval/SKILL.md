---
name: vision-eval
description: Prepare accident/fire video datasets, train or fine-tune detection models, and evaluate the CCTV accident detector (precision, recall, detection delay, false alarms, CPU latency). Use for any work in vision/ involving datasets, training, calibration or metrics.
---

# Vision dataset, training and evaluation rules

## Datasets
- Supported sources: TAD, CADP, MADP, IEEE DataPort traffic-accident videos, plus normal-traffic
  footage for false-alarm testing. Record source URL, licence and download date in
  `docs/data-sources.md`. If a licence forbids redistribution, keep only download scripts in git.
- Convert everything to one annotation format in `data/processed/vision/` with fields:
  video_id, source, fps, accident_onset_s (or null), type, bbox tracks if available.
- Split by VIDEO (never by frame) into train/val/test, stratified by source; save split files.

## Training
- Must run on CPU for inference; training may use GPU. Keep `notebooks/vision/train.ipynb`
  runnable on Google Colab (installs, dataset mount, training, export).
- Calibrate the final probability on the validation split (temperature scaling or isotonic).
- Save model weights with a content hash; record hyperparameters in `results/vision/<run>/config.yaml`.

## Evaluation (test split only)
Report, per dataset and overall:
- Clip-level precision, recall, F1, AUROC, and the reliability diagram (calibration).
- Detection delay: seconds from accident onset to first alarm above each threshold.
- False alarms per camera-hour on normal-traffic footage at each threshold.
- CPU inference latency (ms/frame) and achieved fps on the demo machine.
Save tables to `results/vision/` and export the delay and false-alarm distributions to
`results/vision/sim_inputs.yaml` — the simulator reads detector behaviour from this file.

## Privacy
Verify face and plate blurring on sample frames before any frame is saved or streamed to the UI.
