"""Accident intervals for UCF-Crime training videos from UCA event timestamps (D-027).

    uv run python -m resqcity.vision.datasets.uca

UCA (Yuan et al., CVPR 2024) describes every event of 1,854 UCF-Crime videos with a start and end
time (0.1 s). For each RoadAccidents video, the accident is the earliest sentence matching the
fixed keyword rule in ``configs/vision.yaml`` (``datasets.uca.accident_keywords``); its start and
end become TRAINING labels (``train_onset_s``, ``train_offset_s``). They never replace the official
temporal annotations, which remain the only evaluation labels.

Outputs:
- ``data/processed/vision/uca_intervals.parquet``: video_id, train_onset_s, train_offset_s,
  sentence;
- ``results/vision/uca_label_check.json``: agreement with the official onsets of the 23 annotated
  videos (a data-quality report; nothing is tuned on it).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import load_vision_config

LINE = re.compile(r"^(\S+)\s+(\d+):(\d+(?:\.\d+)?)\s+(\d+):(\d+(?:\.\d+)?)\s*##(.*)$")


def parse_uca(paths: list[Path]) -> pd.DataFrame:
    """``Video mm:ss.s mm:ss.s ##sentence`` lines -> video, start, end (s), text, uca_split."""
    rows = []
    for path in paths:
        uca_split = path.stem.rsplit("_", 1)[-1].lower()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LINE.match(line.strip())
            if m:
                start = int(m.group(2)) * 60 + float(m.group(3))
                end = int(m.group(4)) * 60 + float(m.group(5))
                rows.append((m.group(1), start, end, m.group(6).strip(), uca_split))
    return pd.DataFrame(rows, columns=["video", "start", "end", "text", "uca_split"])


def accident_intervals(uca: pd.DataFrame, keywords: list[str], prefix: str) -> pd.DataFrame:
    """Earliest keyword-matching sentence per ``prefix`` video -> training interval."""
    pattern = re.compile(r"\b(?:" + "|".join(keywords) + r")\b", re.IGNORECASE)
    hits = uca[uca["video"].str.startswith(prefix) & uca["text"].str.contains(pattern)]
    first = hits.sort_values(["video", "start", "end"]).groupby("video", as_index=False).first()
    return pd.DataFrame(
        {
            "video_id": "ucf_" + first["video"],
            "train_onset_s": first["start"].astype(float),
            "train_offset_s": first["end"].astype(float),
            "sentence": first["text"],
        }
    )


def agreement(intervals: pd.DataFrame, annotations: pd.DataFrame) -> dict[str, object]:
    """UCA interval vs the official [onset, offset] of every annotated accident video."""
    official = annotations[annotations["has_onset"] & annotations["label"].eq("accident")]
    official = official[official["source"].eq("ucf_crime")]
    m = official.merge(intervals, on="video_id", how="left")
    found = m["train_onset_s"].notna()
    d = (m["train_onset_s"] - m["accident_onset_s"])[found]
    overlap = (
        np.minimum(m["train_offset_s"], m["accident_offset_s"])
        - np.maximum(m["train_onset_s"], m["accident_onset_s"])
    )[found]
    contains = (
        (m["train_onset_s"] <= m["accident_onset_s"])
        & (m["train_offset_s"] >= m["accident_onset_s"])
    )[found]
    return {
        "official_onset_videos": len(m),
        "with_uca_interval": int(found.sum()),
        "uca_interval_overlaps_official": int((overlap > 0).sum()),
        "uca_interval_contains_official_onset": int(contains.sum()),
        "uca_start_minus_official_onset_s": {
            "median": float(d.median()),
            "q25": float(d.quantile(0.25)),
            "q75": float(d.quantile(0.75)),
            "within_3s": int((d.abs() <= 3).sum()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base, cfg = load_base_config(), load_vision_config()
    u = cfg.datasets.uca
    raw = base.paths.resolve("data_raw") / "vision" / "uca"
    uca = parse_uca([raw / Path(name).name for name in u.files])
    intervals = accident_intervals(uca, u.accident_keywords, cfg.datasets.ucf_crime.accident_prefix)
    processed = base.paths.resolve("data_processed") / "vision"
    out = processed / "uca_intervals.parquet"
    intervals.to_parquet(out, index=False)
    annotations = pd.read_parquet(processed / "annotations.parquet")
    road = uca["video"].str.startswith(cfg.datasets.ucf_crime.accident_prefix)
    report = {
        "source": f"UCA {u.repo}@{u.commit[:12]} ({u.licence})",
        "rule": "earliest sentence matching datasets.uca.accident_keywords",
        "uca_sentences": len(uca),
        "uca_roadaccident_videos": int(uca.loc[road, "video"].nunique()),
        "roadaccident_videos_with_interval": len(intervals),
        "check_against_official_onsets": agreement(intervals, annotations),
    }
    check = base.paths.resolve("results") / "vision" / "uca_label_check.json"
    check.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    for f in (out, check):
        write_meta(f, configs=[CONFIG_DIR / "vision.yaml"], synthetic=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
