"""Diagnostic numbers quoted in docs/decisions.md (D-021, D-022) - regenerated, never typed.

    uv run python -m resqcity.vision.eval.diagnostics

Writes ``results/vision/diagnostics.json``:
- per clip-classifier run and split: AUROC over all supervised windows, AUROC within onset videos
  (pooled windows), and mean classifier probability on normal-video windows, pre-onset windows
  and accident windows;
- heuristic cue activity per video label (share of sampled frames where each cue fires) from the
  cached YOLO+ByteTrack heuristics of the val/test videos.
"""

import argparse
import json

import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.eval.metrics import auroc
from resqcity.vision.offline import heuristics_path

RUNS = ["clip_main", "clip_s0_t0", "clip_s03_t1", "clip_s1_t1"]


def classifier_rows(cache_dir: object, runs: list[str]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        path = cache_dir / f"windows_{run}.parquet"  # type: ignore[operator]
        if not path.exists():
            continue
        w = pd.read_parquet(path)
        for split in ("val", "test"):
            s = w[(w["split"] == split) & w["label"].notna()]
            within = s[s["video_label"] == "accident"]
            rows.append(
                {
                    "run_id": run,
                    "split": split,
                    "auroc_all_windows": auroc(s["label"].to_numpy(), s["clf_p"].to_numpy()),
                    "auroc_within_onset_videos_pooled": auroc(
                        within["label"].to_numpy(), within["clf_p"].to_numpy()
                    ),
                    "mean_p_normal_videos": float(
                        s.loc[s["video_label"] == "normal", "clf_p"].mean()
                    ),
                    "mean_p_pre_onset": float(within.loc[within["label"] == 0, "clf_p"].mean()),
                    "mean_p_accident_windows": float(
                        within.loc[within["label"] == 1, "clf_p"].mean()
                    ),
                }
            )
    return rows


def heuristic_rows(cache_dir: object, ann: pd.DataFrame) -> list[dict[str, object]]:
    frames = []
    for _, v in ann[ann["split"].isin(["val", "test"])].iterrows():
        path = heuristics_path(cache_dir, v["video_id"])  # type: ignore[arg-type]
        if path.exists():
            frames.append(pd.read_parquet(path).assign(label=v["label"]))
    if not frames:
        return []
    h = pd.concat(frames)
    rows = []
    for label, g in h.groupby("label"):
        rows.append(
            {
                "video_label": label,
                "frames": len(g),
                "collision_any": float((g["c_collision"] > 0).mean()),
                "collision_full": float((g["c_collision"] >= 1).mean()),
                "deceleration_full": float((g["c_deceleration"] >= 1).mean()),
                "orientation_full": float((g["c_orientation"] >= 1).mean()),
                "person_down": float((g["c_person_down"] >= 1).mean()),
                "mean_vehicles": float(g["n_vehicles"].mean()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base = load_base_config()
    cache = base.paths.resolve("data_processed") / "vision"
    ann = pd.read_parquet(cache / "annotations.parquet")
    out = base.paths.resolve("results") / "vision" / "diagnostics.json"
    report = {
        "classifier_runs": classifier_rows(cache, RUNS),
        "heuristic_cue_activity_frames": heuristic_rows(cache, ann),
        "note": "frame shares are over sampled frames (5 fps) of val+test videos",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[CONFIG_DIR / "vision.yaml"], synthetic=False)
    print(pd.DataFrame(report["classifier_runs"]).round(3).to_string(index=False))
    print(pd.DataFrame(report["heuristic_cue_activity_frames"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
