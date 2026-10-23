"""Feasibility smoke test of one verifier candidate on a UCF-Crime TRAINING video (no val/test).

    uv run python -m resqcity.vision.eval.verifier_smoke <candidate>

Loads the candidate alone in a fresh process and reports load time, median latency of three
checks, peak VRAM and the scores of a crash and a pre-crash span; writes
results/vision/v5/verifier_smoke/<candidate>.json (D-027: the stage-B latency fix)."""

import json
import sys
import time

import pandas as pd
import torch

from resqcity.common.config import PROJECT_ROOT
from resqcity.vision.config import load_vision_config
from resqcity.vision.detect_track import YoloTracker
from resqcity.vision.privacy import Anonymizer
from resqcity.vision.verifier import clip_frames, load_verifier

name = sys.argv[1]
cfg = load_vision_config()
ann = pd.read_parquet(PROJECT_ROOT / "data/processed/vision/annotations.parquet")
uca = pd.read_parquet(PROJECT_ROOT / "data/processed/vision/uca_intervals.parquet")
train = ann[ann.source.eq("ucf_crime") & ann.split.eq("train") & ann.label.eq("accident")]
row = train.merge(uca, on="video_id").iloc[0]
raw = PROJECT_ROOT / "data/raw/vision"
d = cfg.detector
anon, tracker = Anonymizer(cfg.privacy, d.vehicle_classes, d.person_class), YoloTracker(d)
crash = clip_frames(raw / row.path, row.train_offset_s, cfg.verifier, anon, tracker)
tracker.reset()
calm = clip_frames(raw / row.path, max(4.0, row.train_onset_s - 1.0), cfg.verifier, anon, tracker)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
v = load_verifier(name, cfg.verifier)
load_s = time.perf_counter() - t0
v.score(crash)  # warm-up
lat = []
for frames in (crash, calm, crash):
    t0 = time.perf_counter()
    p = v.score(frames)
    lat.append(time.perf_counter() - t0)
p_crash, p_calm = v.score(crash), v.score(calm)
desc = v.describe(crash)
out = PROJECT_ROOT / "results/vision/v5/verifier_smoke" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
record = {
    "candidate": name,
    "load_s": load_s,
    "latency_median_s": sorted(lat)[1],
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    "p_crash": p_crash,
    "p_before": p_calm,
    "video": row.video_id,
}
out.write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
print(
    f"RESULT {name}: load {load_s:.0f} s | latency median {sorted(lat)[1]:.2f} s | "
    f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | p(crash) {p_crash:.3f} "
    f"p(before) {p_calm:.3f} | {row.video_id} | describe: {desc[:150]!r}"
)
