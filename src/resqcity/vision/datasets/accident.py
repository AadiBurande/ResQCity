"""ACCIDENT benchmark clips (Picek et al., 2026) in the unified annotation format (D-027).

The OFFICIAL in-distribution split is used. Its 1,520 test clips are the never-seen evaluation set
of evaluation v5 (pre-registered in docs/plans/phase-3b.md):
- in ``annotations.parquet`` they appear with split ``holdout`` and WITHOUT impact times, so no
  training or selection code can see their labels;
- their labels go to ``accident_holdout_labels.parquet``, which only the final evaluation reads
  (:func:`load_holdout_labels`; every read is logged to ``results/vision/holdout_access.log``).
The official train clips are split train/val here, grouped by source video (file-name prefix
before the last ``_``); sources with clips on both sides of the official split are ``excluded``.
Window labels use [impact, impact + ``offset_after_impact_s``] as the accident interval.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import PROJECT_ROOT
from resqcity.vision.config import AccidentConfig

log = logging.getLogger(__name__)
METADATA = "metadata-real.csv"
HOLDOUT_FILE = "accident_holdout_labels.parquet"
HOLDOUT_LOG = PROJECT_ROOT / "results" / "vision" / "holdout_access.log"
LABEL_COLUMNS = [
    "accident_time",
    "accident_frame",
    "center_x",
    "center_y",
    "x1",
    "y1",
    "x2",
    "y2",
    "type",
    "rollover",
]
SPLIT_COLUMN = "split_in_distribution"
HOLDOUT = "holdout"


def source_id(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[0]


def assign_accident_splits(meta: pd.DataFrame, cfg: AccidentConfig) -> pd.Series:
    """Official test -> holdout; official train -> train/val by source group (seeded)."""
    src = meta["path"].map(source_id)
    official = meta[SPLIT_COLUMN]
    both = set(src[official.eq("test")]) & set(src[official.eq("train")])
    split = pd.Series(HOLDOUT, index=meta.index)
    train_sources = sorted(set(src[official.eq("train")]) - both)
    rng = np.random.default_rng(cfg.split_seed)
    order = rng.permutation(len(train_sources))
    n_val = round(len(train_sources) * cfg.val_fraction)
    val_sources = {train_sources[i] for i in order[:n_val]}
    is_train = official.eq("train")
    split[is_train] = np.where(src[is_train].isin(val_sources), "val", "train")
    split[is_train & src.isin(both)] = "excluded"
    return split


def accident_records(root: Path, cfg: AccidentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(annotation rows, sealed holdout labels) for every real clip present on disk."""
    base = root / "accident"
    meta = pd.read_csv(base / METADATA)
    meta = meta[[(base / p).exists() for p in meta["path"]]].reset_index(drop=True)
    split = assign_accident_splits(meta, cfg)
    video_id = "acc_" + meta["path"].map(lambda p: Path(p).stem)
    visible = ~split.eq(HOLDOUT)
    onset = meta["accident_time"].where(visible)
    rows = pd.DataFrame(
        {
            "video_id": video_id,
            "source": "accident",
            "path": "accident/" + meta["path"],
            "kind": "video",
            "fps": meta["no_frames"] / meta["duration"],
            "n_frames": meta["no_frames"].astype(int),
            "duration_s": meta["duration"].astype(float),
            "label": "accident",
            "accident_onset_s": onset.astype(float),
            "accident_offset_s": (onset + cfg.offset_after_impact_s).astype(float),
            "has_onset": visible,
            "split": split,
        }
    )
    holdout = pd.concat(
        [video_id.rename("video_id"), meta[[*LABEL_COLUMNS, "region", "quality", "duration"]]],
        axis=1,
    )[~visible].reset_index(drop=True)
    return rows, holdout


def load_holdout_labels(processed: Path, purpose: str) -> pd.DataFrame:
    """Labels of the never-seen ACCIDENT test clips. Only the final v5 evaluation may call this;
    every call is appended to ``results/vision/holdout_access.log`` (who/when/why)."""
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\t{purpose}\n")
    log.warning("reading ACCIDENT holdout labels (%s)", purpose)
    return pd.read_parquet(processed / HOLDOUT_FILE)
