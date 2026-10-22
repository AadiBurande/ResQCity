"""Evaluation v4: two-fold cross-fitting over all onset-annotated videos (D-026).

    uv run python -m resqcity.vision.eval.crossfit

Fold A fits on ``val`` and evaluates on ``test``; fold B fits on ``test`` and evaluates on ``val``.
Within a fold EVERYTHING is chosen on the fit split only: the classifier's early stopping and
temperature (trained with ``train_clip --fit-split``), the candidate (backbone x MIL setting, by
best fit-split within-video AUROC from its training log), the stacker, and the alarm operating
point (Youden's J). Evaluation-split predictions of both folds are pooled, so all 23 onset videos,
all val+test normal videos and all val+test unannotated accident videos are scored exactly once by
a pipeline that never saw them during selection. Outputs ``results/vision/crossfit_v4/`` and
re-exports ``results/vision/sim_inputs.yaml``; the fold-A choice is the deployed live detector
(``results/vision/deployed_detector.json``).

Note: the UCF test split had been looked at during development (D-021 ... D-025); these are still
development estimates, but no number here is fitted on the videos it is reported for.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import load_vision_config
from resqcity.vision.datasets.annotations import pre_v5
from resqcity.vision.eval.evaluate import (
    CONSECUTIVE,
    SCORES,
    add_scores,
    alarm_cfg,
    pick_operating_point,
    sha16,
    threshold_sweep,
)
from resqcity.vision.eval.metrics import (
    DETECTION_GRACE_S,
    per_video_auroc,
    reliability,
    summarise_alarms,
    video_alarm_summary,
)
from resqcity.vision.scoring import Stacker, logit

log = logging.getLogger(__name__)
SETTINGS = ["s0_t0", "s03_t1", "s1_t1"]
# candidate -> (fold-A run fitted on val, fold-B run fitted on test)
CANDIDATES = {
    **{f"image_{s}": (f"clip_{s}", f"clip_{s}_fittest") for s in SETTINGS},
    **{f"video_{s}": (f"video_{s}_fitval", f"video_{s}_fittest") for s in SETTINGS},
}
FOLDS = {"A": ("val", "test", 0), "B": ("test", "val", 1)}  # fold -> (fit, eval, run index)
SELECTION_COLUMN = "val_auroc_within"  # computed on the run's FIT split (train_clip --fit-split)


def selection_score(results: Path, run: str) -> float:
    return float(pd.read_csv(results / run / "train_log.csv")[SELECTION_COLUMN].max())


def fit_split_of(results: Path, run: str) -> str:
    """The split a run was fitted on. Runs trained before ``train_clip --fit-split`` existed record
    no ``fit_split``; they were fitted on val (then hard-coded)."""
    cfg = yaml.safe_load((results / run / "config.yaml").read_text(encoding="utf-8")) or {}
    return str(cfg.get("fit_split", "val"))


def run_fold(
    fold: str, cache: Path, results: Path, ann: pd.DataFrame, out: Path
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    fit, ev, idx = FOLDS[fold]
    cfg = load_vision_config()
    for runs in CANDIDATES.values():
        if fit_split_of(results, runs[idx]) != fit:
            raise ValueError(f"fold {fold}: run {runs[idx]} was not fitted on {fit}")
    scores = {c: selection_score(results, runs[idx]) for c, runs in CANDIDATES.items()}
    chosen = max(scores, key=lambda c: scores[c])
    run = CANDIDATES[chosen][idx]
    w = pd.read_parquet(cache / f"windows_{run}.parquet")
    w = add_scores(w[w["split"].isin([fit, ev])], cache)
    w["clf_logit"] = logit(w["clf_p"])
    fit_sup = w[(w["split"] == fit) & w["label"].notna()]
    stacker = Stacker.fit(fit_sup)
    heur = Stacker.fit(fit_sup, ["heur_score", *[c for c in w.columns if c.startswith("c_")]])
    w["combined_p"], w["heur_p"] = stacker.proba(w), heur.proba(w)
    stacker.save(out / f"stacker_fold{fold}.json")
    fit_videos, ev_videos = ann[ann["split"] == fit], ann[ann["split"] == ev]
    per_video, ops = [], {}
    for name, col in SCORES.items():
        grid = threshold_sweep(w[w["split"] == fit], fit_videos, col, cfg, CONSECUTIVE)
        th, k = pick_operating_point(grid)
        ops[name] = {"threshold": th, "consecutive_windows": k}
        pv = video_alarm_summary(w[w["split"] == ev], ev_videos, col, alarm_cfg(cfg, k), th)
        per_video.append(pv.assign(score=name, fold=fold))
    info = {
        "fit_split": fit,
        "eval_split": ev,
        "candidate_selection_scores": scores,
        "chosen_candidate": chosen,
        "run": run,
        "operating_points": ops,
        "models": {
            "clip_weights_sha256_16": sha16(
                PROJECT_ROOT / "data" / "models" / "clip" / run / "clip_head.pt"
            ),
            "stacker_sha256_16": sha16(out / f"stacker_fold{fold}.json"),
        },
    }
    return info, pd.concat(per_video), w[w["split"] == ev].assign(fold=fold)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    cache = base.paths.resolve("data_processed") / "vision"
    results = base.paths.resolve("results") / "vision"
    out = results / "crossfit_v4"
    out.mkdir(parents=True, exist_ok=True)
    ann = pre_v5(pd.read_parquet(cache / "annotations.parquet"))

    folds, per_video, windows = {}, [], []
    for fold in FOLDS:
        info, pv, w = run_fold(fold, cache, results, ann, out)
        folds[fold] = info
        per_video.append(pv)
        windows.append(w)
        log.info("fold %s: chose %s (%s)", fold, info["chosen_candidate"], info["run"])
    pv_all, w_all = pd.concat(per_video), pd.concat(windows)
    pv_all.to_csv(out / "eval_per_video.csv", index=False, lineterminator="\n")

    summary_rows, auroc_rows = [], []
    within = w_all[w_all["label"].notna() & (w_all["video_label"] == "accident")]
    for name, col in SCORES.items():
        s = summarise_alarms(pv_all[pv_all["score"] == name])
        summary_rows.append({"score": name, **s})
        mean, (lo, hi), n = per_video_auroc(within, col)
        auroc_rows.append(
            {
                "score": name,
                "per_video_auroc_mean": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_videos": n,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "pooled_metrics.csv", index=False, lineterminator="\n")
    # Per fold: the two folds may choose different detectors, so the pooled numbers describe the
    # PROCEDURE; fold A's row is the deployed detector's own (smaller-sample) estimate.
    per_fold = pd.DataFrame(
        [
            {"fold": f, "score": name, **summarise_alarms(g)}
            for (f, name), g in pv_all.groupby(["fold", "score"])
        ]
    )
    per_fold.to_csv(out / "per_fold_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(auroc_rows).to_csv(
        out / "pooled_per_video_auroc.csv", index=False, lineterminator="\n"
    )

    sup = w_all[w_all["label"].notna()]
    calib = {}
    for label, sub in (("all_windows", sup), ("within_onset_videos", within)):
        rel = reliability(sub["label"].to_numpy(), sub["combined_p"].to_numpy())
        calib[label] = {"n": len(sub), "ece": rel.attrs.get("ece")}
    above = pd.concat(
        sup[
            (sup["fold"] == f)
            & (sup["combined_p"] >= folds[f]["operating_points"]["combined"]["threshold"])
        ]
        for f in FOLDS
    )
    calib["above_threshold"] = {
        "n": len(above),
        "mean_p": float(above["combined_p"].mean()) if len(above) else None,
        "observed": float(above["label"].mean()) if len(above) else None,
    }
    report = {
        "evaluation_version": 4,
        "protocol": "two-fold cross-fitting over val/test; all selection on the fit fold (D-026)",
        "detection_rule": f"alarm in [onset, offset + {DETECTION_GRACE_S:g} s]",
        "folds": folds,
        "pooled_combined": summary[summary["score"] == "combined"].iloc[0].to_dict(),
        "per_fold_combined": {
            f: per_fold[(per_fold["fold"] == f) & (per_fold["score"] == "combined")]
            .iloc[0]
            .drop(["fold", "score"])
            .to_dict()
            for f in FOLDS
        },
        "note": "Pooled = the cross-fitted PROCEDURE (folds may choose different detectors); "
        "per_fold_combined.A = the deployed detector (deployed_detector.json).",
        "calibration_combined_pooled": calib,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8", newline="\n"
    )
    export_sim_inputs(results / "sim_inputs.yaml", report)
    deploy = {
        "evaluation": "crossfit_v4 fold A (fitted on val)",
        "clip_run": folds["A"]["run"],
        "candidate": folds["A"]["chosen_candidate"],
        "stacker": (out / "stacker_foldA.json").relative_to(PROJECT_ROOT).as_posix(),
        "operating_point": folds["A"]["operating_points"]["combined"],
        "models": folds["A"]["models"],
    }
    (results / "deployed_detector.json").write_text(
        json.dumps(deploy, indent=2), encoding="utf-8", newline="\n"
    )
    cfgs = [CONFIG_DIR / "vision.yaml", *sorted((CONFIG_DIR / "vision_sweeps").glob("*.yaml"))]
    for f in (
        out / "report.json",
        out / "pooled_metrics.csv",
        out / "per_fold_metrics.csv",
        results / "sim_inputs.yaml",
        results / "deployed_detector.json",
    ):
        write_meta(f, configs=cfgs, synthetic=False, extra={"evaluation_version": 4})
    cols = [
        "score",
        "onset_detected",
        "onset_videos",
        "delay_median_s",
        "road_false_alarms",
        "road_hours",
        "nonroad_false_alarms",
        "nonroad_hours",
        "any_alarm_rate",
    ]
    print(summary[cols].to_string(index=False))
    print(
        json.dumps(
            {f: {k: folds[f][k] for k in ("chosen_candidate", "operating_points")} for f in FOLDS},
            indent=1,
        )
    )
    return 0


def _num(x: object) -> float | None:
    return None if x is None or pd.isna(x) else float(x)  # type: ignore[arg-type]


def export_sim_inputs(path: Path, report: dict) -> None:
    """Detector behaviour for the simulator: pooled cross-fitted estimates (timing-checked
    onset detection, delays, ROAD false-alarm rate) with intervals, plus the deployed detector's
    own fold-A estimates."""
    p = report["pooled_combined"]
    a = report["per_fold_combined"]["A"]
    doc = {
        "source": "measured: results/vision/crossfit_v4 (UCF-Crime val+test, 2-fold cross-fitting)",
        "evaluation_version": 4,
        "detection_rule": report["detection_rule"],
        "caveat": (
            "Public CCTV footage (UCF-Crime, 320x240), not Pune cameras. 23 onset-annotated "
            "videos, ~4 minutes of road footage before onsets: intervals are very wide. The "
            "test split "
            "was looked at during development (D-021..D-025); cross-fitting removes fit-on-eval "
            "bias but these remain development estimates. Vary in sensitivity analysis."
        ),
        "synthetic": False,
        "operating_points": {
            f: report["folds"][f]["operating_points"]["combined"] for f in report["folds"]
        },
        "onset_detected": int(p["onset_detected"]),
        "onset_videos": int(p["onset_videos"]),
        "onset_detection_rate": _num(p["onset_detection_rate"]),
        "onset_detection_rate_ci95": [
            _num(p["onset_detection_rate_ci95_lo"]),
            _num(p["onset_detection_rate_ci95_hi"]),
        ],
        "n_delays": int(p["n_delays"]),
        "delay_median_s": _num(p["delay_median_s"]),
        "road_false_alarms": int(p["road_false_alarms"]),
        "road_hours": _num(p["road_hours"]),
        "road_false_alarms_per_hour": _num(p["road_false_alarms_per_hour"]),
        "road_fa_poisson_ci95": [
            _num(p["road_fa_poisson_ci95_lo"]),
            _num(p["road_fa_poisson_ci95_hi"]),
        ],
        "road_fa_cluster_ci95": [
            _num(p["road_fa_cluster_ci95_lo"]),
            _num(p["road_fa_cluster_ci95_hi"]),
        ],
        "estimates_describe": (
            "the cross-fitted procedure (fold A and B may choose different detectors); the "
            "deployed live detector's own fold-A estimates are under deployed_detector_fold_A"
        ),
        "deployed_detector_fold_A": {
            "candidate": report["folds"]["A"]["chosen_candidate"],
            "onset_detected": int(a["onset_detected"]),
            "onset_videos": int(a["onset_videos"]),
            "road_false_alarms": int(a["road_false_alarms"]),
            "road_hours": _num(a["road_hours"]),
            "road_false_alarms_per_hour": _num(a["road_false_alarms_per_hour"]),
            "road_fa_poisson_ci95": [
                _num(a["road_fa_poisson_ci95_lo"]),
                _num(a["road_fa_poisson_ci95_hi"]),
            ],
        },
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
