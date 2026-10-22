"""Offline per-video processing shared by training and evaluation (results cached per video).

For each annotated video, at ``ingest.sample_fps``:
- ``features/<backbone>/<video_id>.npz``: times (s) and frozen-encoder frame features (float16);
- ``heuristics/<video_id>.parquet``: per sampled frame, the heuristic score and its components
  (YOLO + ByteTrack + ``HeuristicScorer``), plus timing for the CPU/GPU latency report.
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from resqcity.vision.clip_classifier import (
    ClipEncoder,
    FrameEncoder,
    feature_tag,
    make_windows,
    resize_short,
)
from resqcity.vision.config import VisionConfig
from resqcity.vision.datasets.annotations import video_frame_paths
from resqcity.vision.heuristics import COMPONENTS, HeuristicScorer
from resqcity.vision.ingest import file_frames, resize
from resqcity.vision.types import FramePacket, Tracker

log = logging.getLogger(__name__)


def video_frames(raw_root: Path, row: pd.Series, cfg: VisionConfig) -> Iterator[FramePacket]:
    """Sampled frames of an annotated video (file or frame folder) with media-time stamps."""
    if row["kind"] == "video":
        yield from file_frames(raw_root / row["path"], cfg.ingest)
        return
    fps = float(row["fps"])
    step = max(1, round(fps / cfg.ingest.sample_fps))
    for i, path in enumerate(video_frame_paths(raw_root, row["path"])):
        if i % step == 0:
            image = cv2.imread(str(path))
            if image is not None:
                yield FramePacket(t=i / fps, image=resize(image, cfg.ingest.resize_width), index=i)


def feature_path(cache: Path, backbone: str, video_id: str) -> Path:
    return cache / "features" / backbone / f"{video_id}.npz"


def heuristics_path(cache: Path, video_id: str, detector: str = "") -> Path:
    """``heuristics/`` for the default COCO detector (v1-v4); ``heuristics_<detector>/`` else."""
    folder = f"heuristics_{detector}" if detector else "heuristics"
    return cache / folder / f"{video_id}.parquet"


def extract_features(
    raw_root: Path, row: pd.Series, cfg: VisionConfig, encoder: FrameEncoder, cache: Path
) -> Path:
    out = feature_path(cache, feature_tag(cfg.clip_classifier), row["video_id"])
    if out.exists():
        return out
    times, feats, batch, batch_t = [], [], [], []
    for frame in video_frames(raw_root, row, cfg):
        batch.append(frame.image)
        batch_t.append(frame.t)
        if len(batch) == 64:
            feats.append(encoder.encode(batch))
            times.extend(batch_t)
            batch, batch_t = [], []
    if batch:
        feats.append(encoder.encode(batch))
        times.extend(batch_t)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.concatenate(feats).astype(np.float16) if feats else np.zeros((0, 1), np.float16)
    np.savez_compressed(out, times=np.asarray(times, np.float64), feats=arr)
    return out


def extract_window_features(
    raw_root: Path, row: pd.Series, cfg: VisionConfig, encoder: ClipEncoder, cache: Path
) -> Path:
    """Video backbones: one clip embedding per window (same windows as ``make_windows``).

    Saves ``starts``, ``ends`` (s) and ``feats`` (N, D float16)."""
    cc = cfg.clip_classifier
    out = feature_path(cache, cc.backbone, row["video_id"])
    if out.exists():
        return out
    times, frames = [], []
    for frame in video_frames(raw_root, row, cfg):
        times.append(frame.t)
        frames.append(resize_short(frame.image))  # small copies: a long video fits in memory
    t = np.asarray(times, np.float64)
    windows = make_windows(t, cc.window_s, cc.stride_s)
    feats = encoder.encode_clips([frames[w.first : w.last] for w in windows])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        starts=np.asarray([w.start_s for w in windows], np.float64),
        ends=np.asarray([w.end_s for w in windows], np.float64),
        feats=feats.astype(np.float16),
    )
    return out


def load_window_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["starts"], data["ends"], data["feats"].astype(np.float32)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["times"], data["feats"].astype(np.float32)


def run_heuristics(
    raw_root: Path,
    row: pd.Series,
    cfg: VisionConfig,
    tracker_factory: "type[Tracker] | object",
    cache: Path,
    detector: str = "",
    vehicle_classes: set[str] | None = None,
) -> Path:
    """Track + score every sampled frame. ``tracker_factory()`` returns a fresh tracker;
    ``detector`` names the cache folder (see ``heuristics_path``)."""
    out = heuristics_path(cache, row["video_id"], detector)
    if out.exists():
        return out
    tracker = tracker_factory()  # type: ignore[operator]
    rows = heuristic_rows(video_frames(raw_root, row, cfg), tracker, cfg, vehicle_classes)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


class HeuristicRecorder:
    """Per-frame heuristic score, components, counts and timing for ONE video (fresh tracker)."""

    def __init__(
        self, tracker: Tracker, cfg: VisionConfig, vehicle_classes: set[str] | None = None
    ) -> None:
        self.tracker, self.cfg = tracker, cfg
        self.vehicles = vehicle_classes or set(cfg.detector.vehicle_classes)
        self.scorer = HeuristicScorer(cfg.heuristics, self.vehicles, cfg.detector.person_class)
        self.rows: list[dict] = []

    def add(self, frame: FramePacket) -> None:
        t0 = time.perf_counter()
        det = self.tracker.update(frame)
        t1 = time.perf_counter()
        res = self.scorer.update(det)
        t2 = time.perf_counter()
        person = self.cfg.detector.person_class
        self.rows.append(
            {
                "t": frame.t,
                "score": res.score,
                **{f"c_{k}": res.components[k] for k in COMPONENTS},
                "n_vehicles": sum(1 for o in det.tracks if o.cls in self.vehicles),
                "n_persons": sum(1 for o in det.tracks if o.cls == person),
                "overturned": bool(res.overturned_ids),
                "person_down": bool(res.person_down_ids),
                "detect_ms": (t1 - t0) * 1000,
                "heuristic_ms": (t2 - t1) * 1000,
            }
        )


def heuristic_rows(
    frames: "Iterator[FramePacket]",
    tracker: Tracker,
    cfg: VisionConfig,
    vehicle_classes: set[str] | None = None,
) -> list[dict]:
    recorder = HeuristicRecorder(tracker, cfg, vehicle_classes)
    for frame in frames:
        recorder.add(frame)
    return recorder.rows
