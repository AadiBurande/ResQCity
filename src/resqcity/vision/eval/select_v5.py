"""Evaluation v5 selection on VALIDATION data only (phase 3b, D-027; pre-registered in
docs/plans/phase-3b.md).

    uv run python -m resqcity.vision.eval.select_v5 --stage a

Stage A: the 12 candidates of ``configs/vision_sweeps/v5_*.yaml`` (backbone x labels x training
data), each trained with ``train_clip --run-id <name>``. For each candidate, on the validation
videos of UCF-Crime and ACCIDENT: heuristics joined (COCO detector), a stacker fitted on the
labelled validation windows, and the operating point chosen by Youden's J over the v4
threshold/persistence grid. UCF videos use the v4 timing rule; ACCIDENT clips use the v5 rule
(``accident_alarm_summary``). The candidate with the highest validation J is chosen (ties: fewer
road false alarms, then the listed order). No holdout clip is read.

Stage C (run before B; D-027): the vehicle detector for the heuristics, COCO YOLO11n vs UVH-26
YOLO11-S (+ COCO people), for the stage-A classifier, by the same validation J.

Stage B: the second-stage verifier for the A+C detector at its operating point. For every
candidate VLM and verifier threshold tau, candidate alarms are checked (``verified_alarm_times``);
tau is the one with the largest reduction in validation false alarms (road + non-road) that keeps
>= ``verifier.min_detection_retention`` of the validation detections; the verifier must also have
a median latency <= ``verifier.max_latency_s`` per check. The verifier with the largest reduction
is chosen ("none" if no candidate reduces false alarms). VLM scores are cached per alarm time.

Outputs ``results/vision/v5/stage_<x>.csv`` (one row per candidate) and ``stage_<x>.json``.
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config, load_model
from resqcity.common.provenance import write_meta
from resqcity.vision.config import VisionConfig
from resqcity.vision.eval.evaluate import (
    CONSECUTIVE,
    THRESHOLDS,
    add_scores,
    alarm_cfg,
    pick_operating_point,
)
from resqcity.vision.eval.metrics import (
    accident_alarm_summary,
    summarise_alarms,
    video_alarm_summary,
)
from resqcity.vision.scoring import Stacker, logit

log = logging.getLogger(__name__)
SWEEPS = CONFIG_DIR / "vision_sweeps"
STAGE_A = [
    f"v5_{bb}_{lab}_{data}"
    for bb in ("mnet", "sig", "vjepa")
    for lab in ("mil", "int")
    for data in ("ucf", "all")
]
METRIC_COLUMNS = [
    "youden_j",
    "onset_detected",
    "onset_videos",
    "onset_detection_rate",
    "fp_video_rate",
    "road_false_alarms",
    "road_hours",
    "road_false_alarms_per_hour",
    "nonroad_false_alarms",
    "nonroad_hours",
    "delay_median_s",
]


def per_video(
    w: pd.DataFrame,
    videos: pd.DataFrame,
    col: str,
    cfg: VisionConfig,
    th: float,
    k: int,
    verify: Callable[[str, float], bool] | None = None,
) -> pd.DataFrame:
    """Alarm outcomes per video: v4 rule for UCF, v5 rule for ACCIDENT clips."""
    cc, alarm = cfg.clip_classifier, alarm_cfg(cfg, k)
    acc = videos["source"].eq("accident")
    parts = [video_alarm_summary(w, videos[~acc], col, alarm, th, verify=verify)]
    if acc.any():
        parts.append(
            accident_alarm_summary(
                w, videos[acc], col, alarm, th, cc.window_s, cc.stride_s, verify=verify
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_config(run: str) -> VisionConfig:
    return load_model(VisionConfig, CONFIG_DIR / "vision.yaml", SWEEPS / f"{run}.yaml")


def false_alarms(summary: dict) -> int:
    return int(summary["road_false_alarms"]) + int(summary["nonroad_false_alarms"])


def validate(
    run: str, cache: Path, ann: pd.DataFrame, cfg: VisionConfig, detector: str = ""
) -> tuple[dict[str, object], pd.DataFrame, Stacker]:
    """Validation summary at the J-optimal operating point, the scored windows and the stacker."""
    w = pd.read_parquet(cache / f"windows_{run}.parquet")
    w = add_scores(w[w["split"].eq("val")], cache, detector)
    w["clf_logit"] = logit(w["clf_p"])
    stacker = Stacker.fit(w[w["label"].notna()])
    w["combined_p"] = stacker.proba(w)
    videos = ann[ann["video_id"].isin(w["video_id"].unique())]
    grid = pd.DataFrame(
        [
            {"threshold": th, "consecutive_windows": k, **summarise_alarms(pv)}
            for k in CONSECUTIVE
            for th in THRESHOLDS
            for pv in [per_video(w, videos, "combined_p", cfg, th, k)]
        ]
    )
    th, k = pick_operating_point(grid)
    best = grid[(grid["threshold"] == th) & (grid["consecutive_windows"] == k)].iloc[0]
    summary = {"run": run, "threshold": th, "consecutive_windows": k}
    summary.update({c: best[c] for c in METRIC_COLUMNS})
    return summary, w, stacker


def stage_a(cache: Path, ann: pd.DataFrame, out: Path) -> dict[str, object]:
    rows = []
    for run in STAGE_A:
        summary, _, _ = validate(run, cache, ann, run_config(run))
        rows.append(summary)
        log.info("%s: validation J %.3f", run, summary["youden_j"])
    table = pd.DataFrame(rows)
    table["order"] = range(len(table))
    chosen = table.sort_values(
        ["youden_j", "road_false_alarms", "order"], ascending=[False, True, True]
    ).iloc[0]
    table.drop(columns="order").to_csv(out / "stage_a.csv", index=False, lineterminator="\n")
    choice = {
        "stage": "A",
        "rule": "max validation Youden J (ties: fewer road false alarms, then listed order)",
        "chosen_run": chosen["run"],
        "operating_point": {
            "threshold": float(chosen["threshold"]),
            "consecutive_windows": int(chosen["consecutive_windows"]),
        },
        "validation_youden_j": float(chosen["youden_j"]),
    }
    (out / "stage_a.json").write_text(json.dumps(choice, indent=2), encoding="utf-8", newline="\n")
    return choice


def stage_c(cache: Path, ann: pd.DataFrame, out: Path) -> dict[str, object]:
    run = json.loads((out / "stage_a.json").read_text(encoding="utf-8"))["chosen_run"]
    cfg = run_config(run)
    rows = []
    for name, tag in (("coco", ""), ("uvh26", "uvh26")):
        summary, _, _ = validate(run, cache, ann, cfg, tag)
        rows.append({"detector": name, "detector_tag": tag, **summary})
        log.info("%s: validation J %.3f", name, summary["youden_j"])
    table = pd.DataFrame(rows)
    table["order"] = range(len(table))
    chosen = table.sort_values(
        ["youden_j", "road_false_alarms", "order"], ascending=[False, True, True]
    ).iloc[0]
    table.drop(columns="order").to_csv(out / "stage_c.csv", index=False, lineterminator="\n")
    choice = {
        "stage": "C",
        "rule": "max validation Youden J (ties: fewer road false alarms, then COCO)",
        "chosen_run": run,
        "detector": chosen["detector"],
        "detector_tag": chosen["detector_tag"],
        "operating_point": {
            "threshold": float(chosen["threshold"]),
            "consecutive_windows": int(chosen["consecutive_windows"]),
        },
        "validation_youden_j": float(chosen["youden_j"]),
    }
    (out / "stage_c.json").write_text(json.dumps(choice, indent=2), encoding="utf-8", newline="\n")
    return choice


class CachedVerifier:
    """VLM scores of candidate alarms, cached on disk per (video, alarm time)."""

    def __init__(
        self, name: str, cfg: VisionConfig, videos: pd.DataFrame, raw: Path, path: Path
    ) -> None:
        from resqcity.vision.detect_track import YoloTracker
        from resqcity.vision.privacy import Anonymizer
        from resqcity.vision.verifier import load_verifier

        self.cfg, self.raw, self.path = cfg, raw, path
        self.paths = dict(zip(videos["video_id"], videos["path"], strict=True))
        self.fps = dict(zip(videos["video_id"], videos["fps"], strict=True))
        self.model = load_verifier(name, cfg.verifier)  # type: ignore[arg-type]
        d = cfg.detector
        self.anonymizer = Anonymizer(cfg.privacy, d.vehicle_classes, d.person_class)
        self.tracker = YoloTracker(d)
        self.scores: dict[tuple[str, float], tuple[float, float]] = {}
        if path.exists():
            old = pd.read_parquet(path)
            self.scores = {(r.video_id, r.t): (r.p_yes, r.latency_s) for r in old.itertuples()}

    def __call__(self, video_id: str, t: float) -> float:
        from resqcity.vision.verifier import clip_frames

        key = (video_id, round(float(t), 3))
        if key not in self.scores:
            self.tracker.reset()
            frames = clip_frames(
                self.raw / self.paths[video_id],
                t,
                self.cfg.verifier,
                self.anonymizer,
                self.tracker,
                fps=float(self.fps[video_id]),
            )
            t0 = time.perf_counter()
            p = self.model.score(frames)
            self.scores[key] = (p, time.perf_counter() - t0)
        return self.scores[key][0]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"video_id": v, "t": t, "p_yes": p, "latency_s": lat}
            for (v, t), (p, lat) in self.scores.items()
        ]
        pd.DataFrame(rows).to_parquet(self.path, index=False)

    def release(self) -> None:
        """Free the VLM (GPU memory) before the next candidate is loaded."""
        self.model = None

    def median_latency(self) -> float:
        lat = [lat for _, lat in self.scores.values()]
        return float(pd.Series(lat).median()) if lat else float("nan")


def accepts(scorer: "CachedVerifier", tau: float) -> Callable[[str, float], bool]:
    """Bind the current scorer and threshold (a bare lambda would see the loop's last values)."""
    return lambda video_id, t: scorer(video_id, t) >= tau


def stage_b(cache: Path, ann: pd.DataFrame, out: Path) -> dict[str, object]:
    import torch

    c = json.loads((out / "stage_c.json").read_text(encoding="utf-8"))
    run, tag = c["chosen_run"], c["detector_tag"]
    cfg = run_config(run)
    v = cfg.verifier
    if v is None:
        raise ValueError("configs/vision.yaml has no verifier section")
    summary, w, _ = validate(run, cache, ann, cfg, tag)
    th, k = summary["threshold"], summary["consecutive_windows"]
    videos = ann[ann["video_id"].isin(w["video_id"].unique())]
    base = summarise_alarms(per_video(w, videos, "combined_p", cfg, th, k))
    need = math.ceil(v.min_detection_retention * base["onset_detected"])
    rows = [
        {
            "verifier": "none",
            "tau": None,
            "onset_detected": base["onset_detected"],
            "false_alarms": false_alarms(base),
            "fa_reduction": 0.0,
            "median_latency_s": 0.0,
            "eligible": True,
            "note": "",
        }
    ]
    raw = load_base_config().paths.resolve("data_raw") / "vision"
    for name in v.candidates:
        try:
            scorer = CachedVerifier(
                name, cfg, videos, raw, cache / "verifier" / name / f"{run}_{tag or 'coco'}.parquet"
            )
        except Exception as exc:  # infeasible on this machine/software: excluded, not judged
            log.error("verifier %s excluded: %s: %s", name, type(exc).__name__, exc)
            rows.append({"verifier": name, "eligible": False, "note": f"cannot run: {exc}"[:300]})
            continue
        for tau in v.thresholds:
            s = summarise_alarms(
                per_video(w, videos, "combined_p", cfg, th, k, verify=accepts(scorer, tau))
            )
            fa = false_alarms(s)
            rows.append(
                {
                    "verifier": name,
                    "tau": tau,
                    "onset_detected": s["onset_detected"],
                    "false_alarms": fa,
                    "fa_reduction": 1 - fa / max(1, false_alarms(base)),
                    "median_latency_s": None,
                    "eligible": s["onset_detected"] >= need,
                    "note": "",
                }
            )
            log.info(
                "%s tau %.2f: detected %d, false alarms %d", name, tau, s["onset_detected"], fa
            )
            if scorer.median_latency() > v.max_latency_s:  # same criterion, checked early
                log.info("%s: median latency above %.1f s; stopping", name, v.max_latency_s)
                break
        scorer.save()
        latency = scorer.median_latency()
        for r in rows:
            if r["verifier"] == name and r.get("tau") is not None:
                r["median_latency_s"] = latency
                if latency > v.max_latency_s:
                    r["eligible"], r["note"] = False, "too slow"
        scorer.release()
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(out / "stage_b.csv", index=False, lineterminator="\n")
    ok = table[table["eligible"].fillna(False).astype(bool)]
    best = ok.sort_values(
        ["fa_reduction", "onset_detected", "tau"],
        ascending=[False, False, True],
        na_position="first",
    ).iloc[0]
    chosen = best if best["fa_reduction"] > 0 else table.iloc[0]
    choice = {
        "stage": "B",
        "rule": "largest validation false-alarm reduction keeping >= "
        f"{v.min_detection_retention:.0%} of detections, median latency <= {v.max_latency_s} s",
        "chosen_run": run,
        "detector_tag": tag,
        "operating_point": {"threshold": th, "consecutive_windows": k},
        "verifier": chosen["verifier"],
        "tau": None if pd.isna(chosen["tau"]) else float(chosen["tau"]),
        "baseline_detections": int(base["onset_detected"]),
        "baseline_false_alarms": false_alarms(base),
    }
    (out / "stage_b.json").write_text(json.dumps(choice, indent=2), encoding="utf-8", newline="\n")
    return choice


STAGES = {"a": stage_a, "c": stage_c, "b": stage_b}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    cache = base.paths.resolve("data_processed") / "vision"
    out = base.paths.resolve("results") / "vision" / "v5"
    out.mkdir(parents=True, exist_ok=True)
    ann = pd.read_parquet(cache / "annotations.parquet")
    choice = STAGES[args.stage](cache, ann, out)
    configs = [CONFIG_DIR / "vision.yaml", *(SWEEPS / f"{r}.yaml" for r in STAGE_A)]
    for f in (out / f"stage_{args.stage}.csv", out / f"stage_{args.stage}.json"):
        write_meta(f, configs=configs, synthetic=False, extra={"evaluation_version": 5})
    print(json.dumps(choice, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
