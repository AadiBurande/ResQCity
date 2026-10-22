"""Evaluate the accident detector and export detector behaviour for the simulator.

    uv run python -m resqcity.vision.eval.evaluate --run-id <clip run id>

The UCF-Crime ``test`` split has been looked at during development (D-021, D-022, D-025), so it
is reported as a DEVELOPMENT-TEST set; a frozen, never-seen evaluation set (e.g. CADP frames) is
needed for a held-out claim.

Steps: (1) YOLO+ByteTrack+heuristics on val and test videos (cached); (2) stacker fitted on VAL
windows; (3) operating point (threshold, alarm persistence) chosen on VAL by Youden's J =
onset-detection rate - share of videos with a false alarm; (4) metrics on the test split for the
classifier alone, heuristics alone and the combination: timing-checked onset detection with Wilson
intervals, false alarms on ROAD footage (pre-onset) and NON-ROAD footage (UCF normal) separately,
with Poisson and video-clustered bootstrap intervals, per-video AUROC with bootstrap intervals,
calibration overall / within onset videos / above the operating threshold; (5)
``results/vision/<run>/sim_inputs_v3.yaml`` (timing-checked recall, delays, ROAD false-alarm
rate). The simulator reads ``results/vision/sim_inputs.yaml``, exported by ``crossfit`` (v4).
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import AlarmConfig, VisionConfig, load_vision_config
from resqcity.vision.eval.metrics import (
    DETECTION_GRACE_S,
    binary_metrics,
    per_video_auroc,
    reliability,
    summarise_alarms,
    video_alarm_summary,
)
from resqcity.vision.offline import heuristics_path, run_heuristics
from resqcity.vision.scoring import Stacker, logit, window_heuristics

log = logging.getLogger(__name__)
THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
CONSECUTIVE = [1, 2, 3, 4]  # alarm persistence options, chosen jointly with the threshold on VAL
# v1: threshold only, F1 within an FA budget (eval_v1); v2: + persistence (eval_v2, D-022);
# v3: timing-checked detection, road/non-road FA, Youden's J on val (D-025).
EVALUATION_VERSION = 3
SCORES = {"classifier": "clf_p", "heuristics": "heur_p", "combined": "combined_p"}


def add_scores(windows: pd.DataFrame, cache: Path, detector: str = "") -> pd.DataFrame:
    """Join per-window heuristics (``detector`` selects the cache, see ``heuristics_path``)."""
    parts = []
    for vid, w in windows.groupby("video_id"):
        heur = pd.read_parquet(heuristics_path(cache, vid, detector))
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
    """Max Youden's J (ties: fewer road false alarms, then lower threshold, shorter persistence)."""
    best = sweep.sort_values(
        ["youden_j", "road_false_alarms", "threshold", "consecutive_windows"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return float(best["threshold"]), int(best["consecutive_windows"])


def sha16(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


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
        "test_split_status": "development-test (looked at during development; D-025)",
        "detection_rule": f"alarm in [onset, offset + {DETECTION_GRACE_S:g} s]",
        "operating_point_rule": "max Youden J on val videos",
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
    write_csv(pd.concat(val_grids), out / "val_operating_point_grid.csv")
    write_csv(pd.concat(sweeps), out / "test_threshold_sweep.csv")
    report["operating_points_chosen_on_val"] = chosen

    test_sup = w[(w["split"] == "test") & w["label"].notna()]
    within = test_sup[test_sup["video_label"] == "accident"]  # same scene, before vs during
    window_rows, per_dataset, auroc_rows = [], [], []
    for name, col in SCORES.items():
        op = chosen[name]
        for source, sub in [
            ("all", test_sup),
            *list(test_sup.groupby("source")),
            ("within_onset_videos", within),
        ]:
            m = binary_metrics(sub["label"].to_numpy(), sub[col].to_numpy(), op["threshold"])
            window_rows.append({"score": name, "dataset": source, **m.__dict__})
        for split in ("val", "test"):
            sub = w[(w["split"] == split) & w["label"].notna() & (w["video_label"] == "accident")]
            mean, (lo, hi), n = per_video_auroc(sub, col)
            auroc_rows.append(
                {
                    "score": name,
                    "split": split,
                    "per_video_auroc_mean": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_videos": n,
                }
            )
        acfg = alarm_cfg(cfg, op["consecutive_windows"])
        for source, vids in [("all", test_videos), *list(test_videos.groupby("source"))]:
            pv = video_alarm_summary(w[w["split"] == "test"], vids, col, acfg, op["threshold"])
            per_dataset.append({"score": name, "dataset": source, **op, **summarise_alarms(pv)})
            if source == "all":
                write_csv(pv, out / f"test_per_video_{name}.csv")
    write_csv(pd.DataFrame(window_rows), out / "test_window_metrics.csv")
    write_csv(pd.DataFrame(per_dataset), out / "test_video_metrics.csv")
    write_csv(pd.DataFrame(auroc_rows), out / "per_video_auroc.csv")

    calib = {}
    above = test_sup[test_sup["combined_p"] >= chosen["combined"]["threshold"]]
    for label, sub in (
        ("all_windows", test_sup),
        ("within_onset_videos", within),
        ("above_threshold", above),
    ):
        rel = reliability(sub["label"].to_numpy(), sub["combined_p"].to_numpy())
        write_csv(rel, out / f"test_reliability_{label}.csv")
        calib[label] = {
            "n": len(sub),
            "ece": rel.attrs.get("ece"),
            "mean_p": float(sub["combined_p"].mean()) if len(sub) else None,
            "observed": float(sub["label"].mean()) if len(sub) else None,
        }
    report["calibration_combined"] = calib
    _plot_reliability(
        {
            k: pd.read_csv(out / f"test_reliability_{k}.csv")
            for k in ("all_windows", "within_onset_videos")
        },
        calib,
        out / "test_reliability.png",
    )

    heur = pd.concat(pd.read_parquet(heuristics_path(cache, v)) for v in evald["video_id"])
    report["detect_ms_mean_eval_device"] = float(heur["detect_ms"].mean())
    models = {
        "clip_weights_sha256_16": sha16(
            PROJECT_ROOT / "data" / "models" / "clip" / args.run_id / "clip_head.pt"
        ),
        "stacker_sha256_16": sha16(out / "stacker.json"),
    }
    report["models"] = models
    (out / "report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8", newline="\n"
    )
    sim_inputs = out / "sim_inputs_v3.yaml"  # the shared sim_inputs.yaml comes from crossfit
    export_sim_inputs(out, sim_inputs, args.run_id, chosen["combined"], models)
    for f in (
        "test_video_metrics.csv",
        "test_window_metrics.csv",
        "per_video_auroc.csv",
        "report.json",
    ):
        write_meta(
            out / f,
            configs=[CONFIG_DIR / "vision.yaml"],
            synthetic=False,
            extra={"run_id": args.run_id},
        )
    write_meta(
        sim_inputs,
        configs=[CONFIG_DIR / "vision.yaml"],
        synthetic=False,
        extra={"run_id": args.run_id, "evaluation_version": EVALUATION_VERSION, **models},
    )
    cols = [
        "score",
        "dataset",
        "threshold",
        "consecutive_windows",
        "onset_detected",
        "onset_videos",
        "delay_median_s",
        "road_false_alarms",
        "road_hours",
        "nonroad_false_alarms",
        "nonroad_hours",
    ]
    print(pd.DataFrame(per_dataset)[cols].to_string(index=False))
    return 0


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def _plot_reliability(tables: dict[str, pd.DataFrame], calib: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"all_windows": "#0072B2", "within_onset_videos": "#D55E00"}
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    for name, rel in tables.items():
        if not rel.empty:
            ece = calib[name]["ece"]
            ax.plot(
                rel["mean_p"],
                rel["observed"],
                marker="o",
                color=colors[name],
                label=f"{name.replace('_', ' ')} (ECE {ece:.3f}, n={calib[name]['n']})",
            )
    ax.set_xlabel("predicted probability (window)")
    ax.set_ylabel("observed accident frequency")
    ax.set_title("Combined detector: reliability (development-test)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _num(x: object) -> float | None:
    return None if pd.isna(x) else float(x)  # type: ignore[arg-type]


def export_sim_inputs(
    run_dir: Path, path: Path, run_id: str, operating: dict, models: dict
) -> None:
    """Detector behaviour for the simulator: timing-checked onset detection and ROAD false alarms
    (the UCF normal footage is mostly not road scenes and is reported, not exported)."""
    sweep = pd.read_csv(run_dir / "test_threshold_sweep.csv")
    sweep = sweep[sweep["score"] == "combined"]
    per_video = pd.read_csv(run_dir / "test_per_video_combined.csv")
    delays = per_video["delay_s"].dropna().astype(float).tolist()
    doc = {
        "source": f"measured: results/vision/{run_id} (UCF-Crime development-test split)",
        "evaluation_version": EVALUATION_VERSION,
        "detection_rule": f"alarm in [onset, offset + {DETECTION_GRACE_S:g} s]",
        "caveat": (
            "Measured on public CCTV footage (UCF-Crime, 320x240), not Pune cameras. Only 12 "
            "onset-annotated videos and ~2 minutes of road footage before onsets: intervals are "
            "very wide. The test split was used during development (D-025), so these are "
            "development estimates, not held-out results. Use as an empirical prior and vary "
            "in sensitivity analysis."
        ),
        "synthetic": False,
        "models": models,
        "operating_point": operating,
        "per_threshold": [
            {
                "threshold": float(r.threshold),
                "consecutive_windows": int(r.consecutive_windows),
                "onset_detected": int(r.onset_detected),
                "onset_videos": int(r.onset_videos),
                "onset_detection_rate": _num(r.onset_detection_rate),
                "onset_detection_rate_ci95": [
                    _num(r.onset_detection_rate_ci95_lo),
                    _num(r.onset_detection_rate_ci95_hi),
                ],
                "n_delays": int(r.n_delays),
                "delay_median_s": _num(r.delay_median_s),
                "road_false_alarms": int(r.road_false_alarms),
                "road_hours": _num(r.road_hours),
                "road_false_alarms_per_hour": _num(r.road_false_alarms_per_hour),
                "road_fa_poisson_ci95": [
                    _num(r.road_fa_poisson_ci95_lo),
                    _num(r.road_fa_poisson_ci95_hi),
                ],
                "road_fa_cluster_ci95": [
                    _num(r.road_fa_cluster_ci95_lo),
                    _num(r.road_fa_cluster_ci95_hi),
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
