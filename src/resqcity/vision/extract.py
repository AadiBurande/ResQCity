"""Single-pass offline extraction (phase 3b, D-027): decode each video ONCE and feed every
requested frame encoder and heuristic tracker, skipping outputs that are already cached.

    uv run python -m resqcity.vision.extract --sources ucf_crime accident \
        --encoders mobilenet_v3_small siglip2_base --detectors coco uvh26

Outputs use the same formats and paths as ``resqcity.vision.offline``:
``features/<feature tag>/<video>.npz`` and ``heuristics[_<detector>]/<video>.parquet``.
Trackers are reset between videos (equivalent to a fresh tracker; checked in D-027).
Failed videos are listed in ``data/processed/vision/extract_failures.json``.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import load_base_config
from resqcity.vision.clip_classifier import (
    VIDEO_BACKBONES,
    ClipEncoder,
    FrameEncoder,
    WindowStream,
    feature_tag,
    is_video_backbone,
    resize_short,
)
from resqcity.vision.config import VisionConfig, load_vision_config
from resqcity.vision.detect_track import (
    DualTracker,
    YoloTracker,
    resolve_device,
    uvh26_vehicle_classes,
)
from resqcity.vision.offline import (
    HeuristicRecorder,
    feature_path,
    heuristics_path,
    video_frames,
)
from resqcity.vision.types import FramePacket

log = logging.getLogger(__name__)
BATCH = 64
CLIP_BATCH = 4  # video clips per GPU batch (V-JEPA 2 ViT-L, 16 frames at 256 px)
DETECTORS = ("coco", "uvh26")
# Video backbones run with their own clip settings (the same as their v5 sweep overlays).
VIDEO_SETTINGS = {"vjepa2_vitl": {"input_size": 256, "clip_frames": 16}}


class FeatureSink:
    """Batched frame encoding for one video; saves ``times`` and float16 ``feats``."""

    def __init__(self, encoder: FrameEncoder, out: Path) -> None:
        self.encoder, self.out = encoder, out
        self.batch: list[np.ndarray] = []
        self.batch_t: list[float] = []
        self.times: list[float] = []
        self.feats: list[np.ndarray] = []

    def add(self, frame: FramePacket) -> None:
        self.batch.append(frame.image)
        self.batch_t.append(frame.t)
        if len(self.batch) == BATCH:
            self._flush()

    def _flush(self) -> None:
        if self.batch:
            self.feats.append(self.encoder.encode(self.batch))
            self.times.extend(self.batch_t)
            self.batch, self.batch_t = [], []

    def finish(self) -> None:
        self._flush()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        dim = self.encoder.dim or 1
        arr = np.concatenate(self.feats).astype(np.float16) if self.feats else np.zeros((0, dim))
        np.savez_compressed(self.out, times=np.asarray(self.times, np.float64), feats=arr)


class ClipSink:
    """Video backbone: one embedding per window, built as frames stream in (bounded memory).
    Saves ``starts``, ``ends`` and float16 ``feats`` like ``offline.extract_window_features``."""

    def __init__(self, encoder: ClipEncoder, out: Path, window_s: float, stride_s: float) -> None:
        self.encoder, self.out = encoder, out
        self.stream = WindowStream(window_s, stride_s)
        self.pending: list[tuple[float, float, list]] = []
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.feats: list[np.ndarray] = []

    def add(self, frame: FramePacket) -> None:
        small = resize_short(frame.image, self.encoder.short)
        self.pending += self.stream.push(frame.t, small)
        if len(self.pending) >= CLIP_BATCH:
            self._flush()

    def _flush(self) -> None:
        if self.pending:
            self.feats.append(self.encoder.encode_clips([w[2] for w in self.pending], CLIP_BATCH))
            self.starts += [w[0] for w in self.pending]
            self.ends += [w[1] for w in self.pending]
            self.pending = []

    def finish(self) -> None:
        self.pending += self.stream.finish()
        self._flush()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        dim = VIDEO_BACKBONES[self.encoder.backbone]
        feats = np.concatenate(self.feats) if self.feats else np.zeros((0, dim))
        np.savez_compressed(
            self.out,
            starts=np.asarray(self.starts, np.float64),
            ends=np.asarray(self.ends, np.float64),
            feats=feats.astype(np.float16),
        )


class HeuristicSink:
    def __init__(self, recorder: HeuristicRecorder, out: Path) -> None:
        self.recorder, self.out = recorder, out

    def add(self, frame: FramePacket) -> None:
        self.recorder.add(frame)

    def finish(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.recorder.rows).to_parquet(self.out, index=False)


def detector_tag(name: str) -> str:
    return "" if name == "coco" else name  # COCO keeps the v1-v4 cache folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="+", default=["ucf_crime", "accident", "ua_detrac"])
    parser.add_argument(
        "--splits", nargs="*", default=None, help="only these splits (e.g. train val first)"
    )
    parser.add_argument(
        "--shard", default=None, help="i/n: every n-th video from i (parallel workers, disjoint)"
    )
    parser.add_argument(
        "--encoders", nargs="*", default=["mobilenet_v3_small", "siglip2_base", "vjepa2_vitl"]
    )
    parser.add_argument("--detectors", nargs="*", choices=DETECTORS, default=list(DETECTORS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    raw = base.paths.resolve("data_raw") / "vision"
    cache = base.paths.resolve("data_processed") / "vision"
    ann = pd.read_parquet(cache / "annotations.parquet")
    ann = ann[ann["source"].isin(args.sources)]  # video files and frame folders (UA-DETRAC)
    if args.splits:
        ann = ann[ann["split"].isin(args.splits)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        ann = ann.sort_values("video_id").iloc[i::n]
    device = resolve_device(cfg.detector.device)
    encoders: dict[str, FrameEncoder | ClipEncoder] = {}
    for name in args.encoders:
        update = {"backbone": name}
        if is_video_backbone(name):
            update |= VIDEO_SETTINGS.get(name, {})
        cc = cfg.clip_classifier.model_copy(update=update)
        encoders[feature_tag(cc)] = (ClipEncoder if is_video_backbone(name) else FrameEncoder)(
            cc, device
        )
    trackers: dict[str, tuple[object, set[str] | None]] = {}
    for name in args.detectors:
        if name == "coco":
            trackers[name] = (YoloTracker(cfg.detector), None)
        else:
            trackers[name] = (DualTracker(cfg.detector), uvh26_vehicle_classes(cfg.detector))
    failures, t_start = [], time.time()
    for i, (_, row) in enumerate(ann.iterrows(), start=1):
        sinks = _sinks(row, cfg, cache, encoders, trackers)
        if sinks:
            try:
                for frame in video_frames(raw, row, cfg):
                    for sink in sinks:
                        sink.add(frame)
                for sink in sinks:
                    sink.finish()
            except Exception as exc:  # one broken file must not stop an overnight run
                log.error("%s: %s: %s", row["video_id"], type(exc).__name__, exc)
                failures.append({"video_id": row["video_id"], "error": str(exc)})
        if i % 25 == 0 or i == len(ann):
            rate = (time.time() - t_start) / i
            log.info("%d/%d videos, ETA %.0f min", i, len(ann), rate * (len(ann) - i) / 60)
    (cache / "extract_failures.json").write_text(
        json.dumps(failures, indent=1), encoding="utf-8", newline="\n"
    )
    return 1 if failures else 0


def _sinks(
    row: pd.Series,
    cfg: VisionConfig,
    cache: Path,
    encoders: dict[str, FrameEncoder | ClipEncoder],
    trackers: dict[str, tuple[object, set[str] | None]],
) -> list[FeatureSink | ClipSink | HeuristicSink]:
    """Sinks whose outputs are missing for this video (trackers are reset for a fresh start)."""
    sinks: list[FeatureSink | ClipSink | HeuristicSink] = []
    cc = cfg.clip_classifier
    for tag, encoder in encoders.items():
        out = feature_path(cache, tag, row["video_id"])
        if out.exists():
            continue
        if isinstance(encoder, ClipEncoder):
            sinks.append(ClipSink(encoder, out, cc.window_s, cc.stride_s))
        else:
            sinks.append(FeatureSink(encoder, out))
    for name, (tracker, vehicles) in trackers.items():
        out = heuristics_path(cache, row["video_id"], detector_tag(name))
        if not out.exists():
            tracker.reset()  # type: ignore[attr-defined]
            sinks.append(HeuristicSink(HeuristicRecorder(tracker, cfg, vehicles), out))  # type: ignore[arg-type]
    return sinks


if __name__ == "__main__":
    raise SystemExit(main())
