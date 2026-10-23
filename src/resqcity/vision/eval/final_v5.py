"""Evaluation v5, FINAL run on the never-seen ACCIDENT test clips (pre-registered in
docs/plans/phase-3b.md; run once, after selection stages A, C and B).

    uv run python -m resqcity.vision.eval.final_v5

The deployed pipeline is fixed by ``results/vision/v5/stage_b.json``: classifier run, heuristics
detector, operating point, verifier and its threshold. The stacker is refitted on the
validation windows exactly as in selection (deterministic), the holdout windows are scored, and
only then are the sealed holdout labels read (``load_holdout_labels``, logged).

Endpoints (pre-registered): E1 detection rate (Wilson 95 % CI), E2 road false alarms per
camera-hour (Poisson and clip-clustered bootstrap CIs), E3 median detection delay; the
ACCIDENT temporal score T (sigma = 0.5, 1, 2 s; predicted impact = centre of the highest-scoring
window); secondary: US vs non-US clips, the UCF-Crime test split (v4 rule), and verifier calls.
Outputs ``results/vision/v5/`` (report.json, per_clip.csv, stacker.json) and re-exports
``results/vision/sim_inputs.yaml`` and ``results/vision/deployed_detector.json``.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.datasets.accident import load_holdout_labels
from resqcity.vision.eval.evaluate import add_scores, alarm_cfg, sha16
from resqcity.vision.eval.metrics import (
    ACCIDENT_PRE_TOLERANCE_S,
    DETECTION_GRACE_S,
    accident_alarm_summary,
    summarise_alarms,
)
from resqcity.vision.eval.select_v5 import (
    SWEEPS,
    CachedVerifier,
    accepts,
    per_video,
    run_config,
    validate,
)
from resqcity.vision.scoring import logit
from resqcity.vision.train_clip import score_videos

log = logging.getLogger(__name__)
SIGMAS = (0.5, 1.0, 2.0)
NON_US_REGIONS = {"World", "UAE", "Unknown"}


def temporal_score(pred: np.ndarray, truth: np.ndarray, sigma: float) -> float:
    """ACCIDENT's temporal metric: mean exp(-(pred - truth)^2 / (2 sigma^2))."""
    return float(np.mean(np.exp(-((pred - truth) ** 2) / (2 * sigma**2))))


def predicted_impact(w: pd.DataFrame, videos: pd.DataFrame, col: str) -> pd.Series:
    """Centre of the highest-scoring window per clip; clips without windows get the midpoint."""
    best = w.loc[w.groupby("video_id")[col].idxmax()]
    centre = dict(zip(best["video_id"], (best["start_s"] + best["end_s"]) / 2, strict=True))
    return videos.apply(
        lambda v: centre.get(v["video_id"], float(v["duration_s"]) / 2), axis=1
    ).rename("pred_s")


def _num(x: object) -> float | None:
    return None if x is None or pd.isna(x) else float(x)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    cache = base.paths.resolve("data_processed") / "vision"
    results = base.paths.resolve("results") / "vision"
    out = results / "v5"
    b = json.loads((out / "stage_b.json").read_text(encoding="utf-8"))
    run, tag = b["chosen_run"], b["detector_tag"]
    th, k = b["operating_point"]["threshold"], b["operating_point"]["consecutive_windows"]
    cfg = run_config(run)
    cc = cfg.clip_classifier
    ann = pd.read_parquet(cache / "annotations.parquet")

    # 1) stacker exactly as in selection (validation only), operating point unchanged
    summary, _, stacker = validate(run, cache, ann, cfg, tag)
    if (summary["threshold"], summary["consecutive_windows"]) != (th, k):
        raise RuntimeError("validation no longer reproduces the selected operating point")
    stacker.save(out / "stacker.json")

    # 2) score the holdout clips (no labels involved); runs trained with --no-holdout never saw
    #    them, so they are scored here from cached features with the saved head
    windows = pd.read_parquet(cache / f"windows_{run}.parquet")
    holdout = windows[windows["split"].eq("holdout")]
    if holdout.empty:
        holdout = score_videos(
            PROJECT_ROOT / "data" / "models" / "clip" / run,
            cfg,
            cache,
            ann[ann["split"].eq("holdout")].reset_index(drop=True),
        )
    w = add_scores(holdout, cache, tag)
    w["clf_logit"] = logit(w["clf_p"])
    w["combined_p"] = stacker.proba(w)

    # 3) only now: the sealed labels
    labels = load_holdout_labels(cache, f"final v5 evaluation, run {run}")
    videos = ann[ann["split"].eq("holdout")].drop(columns=["accident_onset_s"])
    videos = videos.merge(
        labels[["video_id", "accident_time", "region"]].rename(
            columns={"accident_time": "accident_onset_s"}
        ),
        on="video_id",
    )
    verify, scorer = None, None
    if b["verifier"] != "none":
        scorer = CachedVerifier(
            b["verifier"],
            cfg,
            videos,
            base.paths.resolve("data_raw") / "vision",
            cache / "verifier" / b["verifier"] / f"{run}_{tag or 'coco'}_holdout.parquet",
        )
        verify = accepts(scorer, b["tau"])
    pv = accident_alarm_summary(
        w, videos, "combined_p", alarm_cfg(cfg, k), th, cc.window_s, cc.stride_s, verify=verify
    )
    if scorer is not None:
        scorer.save()
    pv = pv.merge(videos[["video_id", "region", "accident_onset_s"]], on="video_id")
    pv["us"] = ~pv["region"].isin(NON_US_REGIONS)
    pv["pred_impact_s"] = predicted_impact(w, videos, "combined_p").to_numpy()
    pv.to_csv(out / "per_clip.csv", index=False, lineterminator="\n")

    primary = summarise_alarms(pv)
    truth, pred = pv["accident_onset_s"].to_numpy(float), pv["pred_impact_s"].to_numpy(float)
    ucf_test = ann[ann["source"].eq("ucf_crime") & ann["split"].eq("test")]
    wu = add_scores(windows[windows["video_id"].isin(ucf_test["video_id"])], cache, tag)
    wu["clf_logit"] = logit(wu["clf_p"])
    wu["combined_p"] = stacker.proba(wu)
    ucf = summarise_alarms(per_video(wu, ucf_test, "combined_p", cfg, th, k, verify=None))
    report = {
        "evaluation_version": 5,
        "protocol": "docs/plans/phase-3b.md (pre-registered); selection on validation only",
        "deployed": {
            "run": run,
            "detector": tag or "coco",
            "operating_point": {"threshold": th, "consecutive_windows": k},
            "verifier": b["verifier"],
            "tau": b["tau"],
        },
        "detection_rule": f"alarm in [t* - {ACCIDENT_PRE_TOLERANCE_S:g} s, t* + "
        f"{DETECTION_GRACE_S:g} s]; earlier alarms are road false alarms",
        "accident_holdout": primary,
        "accident_temporal_score": {f"sigma_{s:g}": temporal_score(pred, truth, s) for s in SIGMAS},
        "by_region": {
            "us": summarise_alarms(pv[pv["us"]]),
            "non_us": summarise_alarms(pv[~pv["us"]]),
        },
        "ucf_test_split_v4_rule_no_verifier": ucf,
        "verifier_calls": None if scorer is None else len(scorer.scores),
        "verifier_median_latency_s": None if scorer is None else scorer.median_latency(),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8", newline="\n"
    )
    export(results, out, report, run, tag)
    configs = [CONFIG_DIR / "vision.yaml", SWEEPS / f"{run}.yaml"]
    for f in (out / "report.json", out / "per_clip.csv", results / "sim_inputs.yaml"):
        write_meta(f, configs=configs, synthetic=False, extra={"evaluation_version": 5})
    write_meta(results / "deployed_detector.json", configs=configs, synthetic=False)
    print(json.dumps({k: report[k] for k in ("deployed", "accident_temporal_score")}, indent=1))
    print(json.dumps({k: primary[k] for k in list(primary)[:12]}, indent=1, default=float))
    return 0


def export(results, out, report: dict, run: str, tag: str) -> None:  # noqa: ANN001
    """``sim_inputs.yaml`` v5 (never-seen ACCIDENT estimates) and the deployed live detector."""
    p = report["accident_holdout"]
    doc = {
        "source": "measured: results/vision/v5 (ACCIDENT IID test clips, never used for "
        "training or selection; evaluation v5, pre-registered)",
        "evaluation_version": 5,
        "detection_rule": report["detection_rule"],
        "caveat": "Public CCTV clips (mostly US, ~20 % elsewhere), trimmed around a crash; road "
        "false alarms are measured on the footage BEFORE each crash. Not Pune cameras.",
        "synthetic": False,
        "deployed": report["deployed"],
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
    }
    (results / "sim_inputs.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=False), encoding="utf-8", newline="\n"
    )
    deploy = {
        "evaluation": "v5 (phase 3b): selected on validation, tested once on ACCIDENT holdout",
        "clip_run": run,
        "detector": tag or "coco",
        "stacker": (out / "stacker.json").relative_to(PROJECT_ROOT).as_posix(),
        "operating_point": report["deployed"]["operating_point"],
        "verifier": report["deployed"]["verifier"],
        "verifier_tau": report["deployed"]["tau"],
        "models": {
            "clip_weights_sha256_16": sha16(
                PROJECT_ROOT / "data" / "models" / "clip" / run / "clip_head.pt"
            ),
            "stacker_sha256_16": sha16(out / "stacker.json"),
        },
    }
    (results / "deployed_detector.json").write_text(
        json.dumps(deploy, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
