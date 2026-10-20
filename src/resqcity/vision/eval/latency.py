"""CPU latency of the live pipeline per frame (demo machine; acceptance: runs on a laptop CPU).

    uv run python -m resqcity.vision.eval.latency [--frames 200] [--clip-run data/models/clip/<run>]

Measures, on CPU only: YOLO+ByteTrack, frame encoder, heuristics and anonymisation, per sampled
frame of a test video. Writes ``results/vision/latency_cpu.json`` with machine details.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import load_vision_config
from resqcity.vision.heuristics import HeuristicScorer
from resqcity.vision.ingest import file_frames
from resqcity.vision.privacy import Anonymizer


def main(argv: list[str] | None = None) -> int:
    import torch

    from resqcity.vision.clip_classifier import FrameEncoder
    from resqcity.vision.detect_track import YoloTracker

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args(argv)
    base, cfg = load_base_config(), load_vision_config()
    cpu_cfg = cfg.model_copy(update={"detector": cfg.detector.model_copy(update={"device": "cpu"})})
    raw = base.paths.resolve("data_raw") / "vision"
    ann = pd.read_parquet(base.paths.resolve("data_processed") / "vision" / "annotations.parquet")
    video = (
        ann[(ann["split"] == "test") & (ann["kind"] == "video")].sort_values("duration_s").iloc[-1]
    )
    tracker = YoloTracker(cpu_cfg.detector)
    encoder = FrameEncoder(cfg.clip_classifier, "cpu")
    scorer = HeuristicScorer(
        cfg.heuristics, set(cfg.detector.vehicle_classes), cfg.detector.person_class
    )
    anon = Anonymizer(cfg.privacy, cfg.detector.vehicle_classes, cfg.detector.person_class)
    timings: dict[str, list[float]] = {
        "detect_track": [],
        "encode": [],
        "heuristics": [],
        "anonymise": [],
        "total": [],
    }
    for i, frame in enumerate(file_frames(raw / video["path"], cfg.ingest)):
        if i >= args.frames:
            break
        t0 = time.perf_counter()
        det = tracker.update(frame)
        t1 = time.perf_counter()
        encoder.encode([frame.image])
        t2 = time.perf_counter()
        scorer.update(det)
        t3 = time.perf_counter()
        anon.apply(frame.image, det.tracks)
        t4 = time.perf_counter()
        if i >= 5:  # skip warm-up frames
            for k, v in (
                ("detect_track", t1 - t0),
                ("encode", t2 - t1),
                ("heuristics", t3 - t2),
                ("anonymise", t4 - t3),
                ("total", t4 - t0),
            ):
                timings[k].append(v * 1000)
    summary = {
        k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
        for k, v in timings.items()
    }
    fps = 1000.0 / summary["total"]["mean_ms"]
    report = {
        "video_id": video["video_id"],
        "frames_measured": len(timings["total"]),
        "frame_width_px": cfg.ingest.resize_width,
        "stages": summary,
        "achieved_fps_cpu": fps,
        "real_time_at_sample_fps": fps >= cfg.ingest.sample_fps,
        "sample_fps": cfg.ingest.sample_fps,
        "machine": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    out: Path = base.paths.resolve("results") / "vision" / "latency_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[CONFIG_DIR / "vision.yaml"], synthetic=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
