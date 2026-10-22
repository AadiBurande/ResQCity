"""Unified video annotations and video-level splits (never by frame: prevents leakage).

    uv run python -m resqcity.vision.datasets.annotations

Writes ``data/processed/vision/annotations.parquet`` and ``splits.csv`` with one row per video:
video_id, source, path, kind (video | frames), fps, n_frames, duration_s, label (accident |
normal), accident_onset_s, accident_offset_s (null when unknown), has_onset, split.
Splits are stratified by (source, label, has_onset) and seeded. ACCIDENT clips (D-027) are added
AFTER that split, with their own split from the official benchmark split
(``resqcity.vision.datasets.accident``), so the UCF-Crime splits are unchanged; their never-seen
test clips get split ``holdout`` and no labels here (sealed in ``accident_holdout_labels.parquet``).
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import SplitConfig, VisionConfig, load_vision_config
from resqcity.vision.datasets.accident import HOLDOUT_FILE, accident_records

log = logging.getLogger(__name__)
COLUMNS = [
    "video_id",
    "source",
    "path",
    "kind",
    "fps",
    "n_frames",
    "duration_s",
    "label",
    "accident_onset_s",
    "accident_offset_s",
    "has_onset",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PRE_V5_SOURCES = ("ucf_crime", "cadp")  # the scope of evaluations v1-v4 (ACCIDENT came in v5)
CADP_FPS = 30.0  # ASSUMPTION: CADP releases frames, not fps; YouTube traffic clips are ~30 fps


@dataclass
class VideoRecord:
    video_id: str
    source: str
    path: str
    kind: str
    fps: float
    n_frames: int
    duration_s: float
    label: str
    accident_onset_s: float | None
    accident_offset_s: float | None
    has_onset: bool


def probe_video(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return float(fps), n


def parse_ucf_temporal(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """``name class s1 e1 s2 e2`` (frames; -1 = none) -> {name: (s1, e1, s2, e2)}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 6:
            out[parts[0]] = tuple(int(p) for p in parts[2:6])
    return out


def ucf_records(root: Path, cfg: VisionConfig) -> list[VideoRecord]:
    base = root / "ucf_crime"
    temporal = parse_ucf_temporal(base / Path(cfg.datasets.ucf_crime.temporal_annotations).name)
    records = []
    for label, folder in (
        ("accident", cfg.datasets.ucf_crime.accident_prefix),
        ("normal", "Normal"),
    ):
        for path in sorted((base / folder).glob("*.mp4")):
            fps, n = probe_video(path)
            if n <= 0:
                log.warning("skip unreadable %s", path.name)
                continue
            onset = offset = None
            spans = temporal.get(path.name)
            if label == "accident" and spans is not None and spans[0] >= 0:
                s1, e1, _s2, e2 = spans
                onset, offset = s1 / fps, max(e1, e2) / fps
            records.append(
                VideoRecord(
                    video_id=f"ucf_{path.stem}",
                    source="ucf_crime",
                    path=path.relative_to(root).as_posix(),
                    kind="video",
                    fps=fps,
                    n_frames=n,
                    duration_s=n / fps,
                    label=label,
                    accident_onset_s=onset,
                    accident_offset_s=offset,
                    has_onset=onset is not None,
                )
            )
    return records


def _frame_dirs(frames_root: Path) -> list[Path]:
    dirs = {p.parent for p in frames_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    return sorted(dirs)


def cadp_records(root: Path) -> list[VideoRecord]:
    """CADP: one directory of extracted frames per clip + annotations JSON (temporal extents for
    the annotated subset). The JSON is keyed by clip name; frame indices of the accident span are
    read from whatever fields carry them (see ``_cadp_span``)."""
    base = root / "cadp"
    ann_path = base / "annotations_CADP.json"
    if not (base / "frames").exists():
        return []
    annotations = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    records = []
    for d in _frame_dirs(base / "frames"):
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        span = _cadp_span(annotations, d.name)
        onset = offset = None
        if span is not None:
            onset, offset = span
        records.append(
            VideoRecord(
                video_id=f"cadp_{d.name}",
                source="cadp",
                path=d.relative_to(root).as_posix(),
                kind="frames",
                fps=CADP_FPS,
                n_frames=n,
                duration_s=n / CADP_FPS,
                label="accident",  # every CADP clip contains an accident
                accident_onset_s=onset,
                accident_offset_s=offset,
                has_onset=onset is not None,
            )
        )
    return records


def _cadp_span(annotations: object, clip: str) -> tuple[float, float] | None:
    """Accident span for a clip from the CADP JSON: min/max keyframe "frame" value.

    The values are non-integer (e.g. 22.0335), so they are read as SECONDS. This is unverified
    against the frames, which could not be downloaded automatically (D-020)."""
    entry = annotations.get(clip) if isinstance(annotations, dict) else None
    if entry is None:
        return None
    frames: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in {"frame", "frame_id", "frame_number", "frameid"} and isinstance(
                    value, int | float
                ):
                    frames.append(float(value))
                elif key.isdigit() and isinstance(value, dict | list):
                    frames.append(float(key))
                    walk(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(entry)
    return (min(frames), max(frames)) if frames else None


def video_frame_paths(root: Path, rel: str) -> list[Path]:
    """Sorted frame files of a frames-kind video."""
    return sorted(p for p in (root / rel).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Video-level split, stratified by (source, label, has_onset); every stratum with >= 3 videos
    gets at least one val and one test video. With ``onset_videos_eval_only``, accident videos
    with annotated onsets go only to val/test (half each)."""
    rng = np.random.default_rng(cfg.seed)
    split = pd.Series("train", index=df.index)
    for (_, label, has_onset), group in df.groupby(["source", "label", "has_onset"], sort=True):
        idx = group.index.to_numpy()[rng.permutation(len(group))]
        n = len(idx)
        if cfg.onset_videos_eval_only and label == "accident" and has_onset:
            n_test = (n + 1) // 2  # all to evaluation: half test, half val
            split.loc[idx[:n_test]] = "test"
            split.loc[idx[n_test:]] = "val"
            continue
        n_test = max(1, round(n * cfg.test)) if n >= 3 else 0
        n_val = max(1, round(n * cfg.val)) if n >= 3 else 0
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test : n_test + n_val]] = "val"
    return split


def pre_v5(ann: pd.DataFrame) -> pd.DataFrame:
    """Videos within the scope of evaluations v1-v4 (keeps them reproducible after ACCIDENT was
    added: its validation clips share the split name ``val``)."""
    return ann[ann["source"].isin(PRE_V5_SOURCES)].reset_index(drop=True)


def build_annotations(root: Path, cfg: VisionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(annotations, sealed ACCIDENT holdout labels; empty if ACCIDENT is not downloaded)."""
    records = ucf_records(root, cfg) + cadp_records(root)
    df = pd.DataFrame([asdict(r) for r in records], columns=COLUMNS)
    if df.empty:
        raise RuntimeError(
            f"no videos found under {root}; run resqcity.vision.datasets.download first"
        )
    df["split"] = assign_splits(df, cfg.datasets.split)  # UCF/CADP only: unchanged by ACCIDENT
    holdout = pd.DataFrame()
    if (root / "accident" / "metadata-real.csv").exists():
        acc, holdout = accident_records(root, cfg.datasets.accident)
        df = pd.concat([df, acc], ignore_index=True)
    return df.sort_values("video_id", ignore_index=True), holdout


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    root = base.paths.resolve("data_raw") / "vision"
    out_dir = base.paths.resolve("data_processed") / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, holdout = build_annotations(root, cfg)
    out = out_dir / "annotations.parquet"
    df.to_parquet(out, index=False)
    if not holdout.empty:
        holdout.to_parquet(out_dir / HOLDOUT_FILE, index=False)  # read only by the final run
    df[["video_id", "source", "label", "has_onset", "split"]].to_csv(
        out_dir / "splits.csv", index=False, lineterminator="\n"
    )
    write_meta(
        out,
        configs=[CONFIG_DIR / "vision.yaml"],
        synthetic=False,
        seeds={
            "split_seed": cfg.datasets.split.seed,
            "accident_split_seed": cfg.datasets.accident.split_seed,
        },
    )
    print(df.groupby(["source", "label", "has_onset", "split"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
