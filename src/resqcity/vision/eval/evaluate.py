"""Evaluate the accident detector on the TEST split and export detector behaviour for the simulator.

    uv run python -m resqcity.vision.eval.evaluate --run-id <clip run id>

Steps: (1) YOLO+ByteTrack+heuristics on val and test videos (cached); (2) stacker fitted on VAL
windows; (3) operating threshold chosen on VAL (max F1 of video-level detection subject to the
false-alarm budget); (4) all metrics on TEST only, per dataset and overall, for the classifier
alone, heuristics alone and the combination (ablation); (5) ``results/vision/sim_inputs.yaml``.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import AlarmConfig, VisionConfig, load_vision_config
from resqcity.vision.eval.metrics import (
    binary_metrics,
    reliability,
    summarise_alarms,
    video_alarm_summary,
)
from resqcity.vision.offline import heuristics_path, run_heuristics
from resqcity.vision.scoring import Stacker, logit, window_heuristics

log = logging.getLogger(__name__)
THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
FALSE_ALARM_BUDGET_PER_HOUR = 1.0  # max false alarms per camera-hour allowed when picking threshold
CONSECUTIVE = [1, 2, 3, 4]  # alarm persistence options, chosen jointly with the threshold on VAL
EVALUATION_VERSION = (
    2  # v1: threshold only (results/.../eval_v1); v2: (threshold, persistence), D-022
)
SCORES = {"classifier": "clf_p", "heuristics": "heur_p", "combined": "combined_p"}


def add_scores(windows: pd.DataFrame, cache: Path) -> pd.DataFrame:
    parts = []
    for vid, w in windows.groupby("video_id"):
        heur = pd.read_parquet(heuristics_path(cache, vid))
        parts.append(window_heuristics(w, heur))
    return pd.concat(parts).sort_index()


def alarm_cfg(cfg: VisionConfig, consecutive: int) -> AlarmConfig:
    return cfg.alarm.model_copy(update={"consecutive_windows": consecutive})


def threshold_sweep(
    windows: pd.DataFrame,
    videos: pd.DataFrame,
    col: str,
    cfg: VisionConfig,
    consecutive: list[int] | None = None,
) -> pd.DataFrame:
    rows = []
    for k in consecutive or [cfg.alarm.consecutive_windows]:
        for th in THRESHOLDS:
            per_video = video_alarm_summary(windows, videos, col, alarm_cfg(cfg, k), th)
            rows.append({"threshold": th, "consecutive_windows": k, **summarise_alarms(per_video)})
    return pd.DataFrame(rows)


def pick_operating_point(sweep: pd.DataFrame) -> tuple[float, int]:
    """Max video F1 within the false-alarm budget (ties: more onsets detected, then the lower
    threshold and shorter persistence, i.e. the faster-alarming setting)."""
    ok = sweep[sweep["false_alarms_per_camera_hour"] <= FALSE_ALARM_BUDGET_PER_HOUR]
    pool = ok if not ok.empty else sweep
    best = pool.sort_values(
        ["video_f1", "onset_detected", "threshold", "consecutive_windows"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(best["threshold"]), int(best["consecutive_windows"])


def main(argv: list[str] | None = None) -> int:
    from resqcity.vision.detect_track import YoloTracker

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    raw = base.paths.resolve("data_raw") / "vision"
    cache = base.paths.resolve("data_processed") / "vision"
    out = base.paths.resolve("results") / "vision" / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    ann = pd.read_parquet(cache / "annotations.parquet")
    windows = pd.read_parquet(cache / f"windows_{args.run_id}.parquet")
    evald = ann[ann["split"].isin(["val", "test"])]
    for i, (_, row) in enumerate(evald.iterrows(), start=1):
        run_heuristics(raw, row, cfg, lambda: YoloTracker(cfg.detector), cache)
        if i % 10 == 0:
            log.info("heuristics: %d/%d videos", i, len(evald))

    w = add_scores(windows[windows["split"].isin(["val", "test"])], cache)
    w["clf_logit"] = logit(w["clf_p"])  # calibrated classifier, on the logit scale
    val_sup = w[(w["split"] == "val") & w["label"].notna()]
    stacker = Stacker.fit(val_sup)
    heur_only = Stacker.fit(val_sup, ["heur_score", *[c for c in w.columns if c.startswith("c_")]])
    w["combined_p"] = stacker.proba(w)
    w["heur_p"] = heur_only.proba(w)
    stacker.save(out / "stacker.json")

    report: dict[str, object] = {
        "run_id": args.run_id,
        "evaluation_version": EVALUATION_VERSION,
        "false_alarm_budget_per_hour": FALSE_ALARM_BUDGET_PER_HOUR,
    }
    val_videos, test_videos = ann[ann["split"] == "val"], ann[ann["split"] == "test"]
    sweeps, val_grids, chosen = [], [], {}
    for name, col in SCORES.items():
        grid = threshold_sweep(w[w["split"] == "val"], val_videos, col, cfg, CONSECUTIVE)
        val_grids.append(grid.assign(score=name))
        th, k = pick_operating_point(grid)
        chosen[name] = {"threshold": th, "consecutive_windows": k}
        sweep = threshold_sweep(w[w["split"] == "test"], test_videos, col, cfg, [k])
        sweeps.append(sweep.assign(score=name))
    pd.concat(val_grids).to_csv(
        out / "val_operating_point_grid.csv", index=False, lineterminator="\n"
    )
    pd.concat(sweeps).to_csv(out / "test_threshold_sweep.csv", index=False, lineterminator="\n")
    report["operating_points_chosen_on_val"] = chosen

    test_sup = w[(w["split"] == "test") & w["label"].notna()]
    window_rows, per_dataset = [], []
    for name, col in SCORES.items():
        within = test_sup[test_sup["video_label"] == "accident"]  # same scene, before vs during
        subsets = [
            ("all", test_sup),
            *list(test_sup.groupby("source")),
            ("within_onset_videos", within),
        ]
        op = chosen[name]
        acfg = alarm_cfg(cfg, op["consecutive_windows"])
        for source, sub in subsets:
            m = binary_metrics(sub["label"].to_numpy(), sub[col].to_numpy(), op["threshold"])
            window_rows.append({"score": name, "dataset": source, **m.__dict__})
        for source, vids in [("all", test_videos), *list(test_videos.groupby("source"))]:
            pv = video_alarm_summary(w[w["split"] == "test"], vids, col, acfg, op["threshold"])
            per_dataset.append({"score": name, "dataset": source, **op, **summarise_alarms(pv)})
            if name == "combined" and source == "all":
                pv.to_csv(out / "test_per_video.csv", index=False, lineterminator="\n")
    pd.DataFrame(window_rows).to_csv(
        out / "test_window_metrics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(per_dataset).to_csv(
        out / "test_video_metrics.csv", index=False, lineterminator="\n"
    )

    rel = reliability(test_sup["label"].to_numpy(), test_sup["combined_p"].to_numpy())
    rel.to_csv(out / "test_reliability.csv", index=False, lineterminator="\n")
    report["test_ece_combined"] = rel.attrs.get("ece")
    _plot_reliability(rel, out / "test_reliability.png")

    heur = pd.concat(pd.read_parquet(heuristics_path(cache, v)) for v in evald["video_id"])
    report["detect_ms_mean_gpu_or_device"] = float(heur["detect_ms"].mean())
    (out / "report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8", newline="\n"
    )
    sim_inputs = base.paths.resolve("results") / "vision" / "sim_inputs.yaml"
    export_sim_inputs(out, sim_inputs, args.run_id, chosen["combined"])
    write_meta(
        sim_inputs,
        configs=[CONFIG_DIR / "vision.yaml"],
        synthetic=False,
        extra={"run_id": args.run_id, "evaluation_version": EVALUATION_VERSION},
    )
    for f in ("test_video_metrics.csv", "test_window_metrics.csv", "report.json"):
        write_meta(
            out / f,
            configs=[CONFIG_DIR / "vision.yaml"],
            synthetic=False,
            extra={"run_id": args.run_id},
        )
    print(pd.DataFrame(per_dataset).to_string(index=False))
    return 0


def _plot_reliability(rel: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    if not rel.empty:
        ax.plot(
            rel["mean_p"], rel["observed"], marker="o", color="#0072B2", label="combined detector"
        )
    ax.set_xlabel("predicted probability (window)")
    ax.set_ylabel("observed accident frequency")
    ax.set_title(
        f"Reliability on test windows (ECE {rel.attrs.get('ece', float('nan')):.3f})", fontsize=10
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _num(x: object) -> float | None:
    return None if pd.isna(x) else float(x)  # type: ignore[arg-type]


def export_sim_inputs(run_dir: Path, path: Path, run_id: str, operating: dict) -> None:
    """Detector behaviour for the simulator, MEASURED on the test split (per threshold, at the
    persistence chosen on val)."""
    sweep = pd.read_csv(run_dir / "test_threshold_sweep.csv")
    sweep = sweep[sweep["score"] == "combined"]
    per_video = pd.read_csv(run_dir / "test_per_video.csv")
    delays = per_video["delay_s"].dropna().astype(float).tolist()
    doc = {
        "source": f"measured: results/vision/{run_id} (test split; UCF-Crime)",
        "evaluation_version": EVALUATION_VERSION,
        "caveat": (
            "Measured on public CCTV footage (UCF-Crime, 320x240), not Pune cameras; "
            "12 test videos with annotated onsets and ~0.6 h of normal test footage, so intervals "
            "are wide. The v2 operating-point procedure was adopted after the v1 test result had "
            "been seen (D-022). Treat as an empirical prior; vary in sensitivity analysis."
        ),
        "synthetic": False,
        "operating_point": operating,
        "per_threshold": [
            {
                "threshold": float(r.threshold),
                "consecutive_windows": int(r.consecutive_windows),
                "recall": _num(r.video_recall),
                "onset_detection_rate": _num(r.onset_detected / r.onset_videos)
                if r.onset_videos
                else None,
                "delay_median_s": _num(r.delay_median_s),
                "delay_p90_s": _num(r.delay_p90_s),
                "false_alarms_per_camera_hour": _num(r.false_alarms_per_camera_hour),
                "false_alarms_per_camera_hour_ci95": [
                    _num(r.false_alarms_per_camera_hour_ci95_lo),
                    _num(r.false_alarms_per_camera_hour_ci95_hi),
                ],
            }
            for r in sweep.itertuples()
        ],
        "delays_at_operating_point_s": sorted(round(d, 2) for d in delays),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
