"""Exploratory analyses of the FINAL v5 run (not pre-registered; labelled as exploratory, D-027).

    uv run python -m resqcity.vision.eval.explore_v5

Uses ``results/vision/v5/per_clip.csv`` (labels written by the final run), the saved stacker and
the cached verifier scores; the sealed holdout label file is NOT read again. Every alarm of the
final run is reconstructed and checked against ``per_clip.csv`` / ``per_sequence_ua_detrac.csv``
first (the analysis stops if they disagree).
1. Lead time: detection on clips whose impact leaves the warm-up + 1 s, split by whether an alarm
   fired before t* - 1 s (the 60 s cooldown then hides a later detection by construction).
2. Timing of pre-impact alarms against a uniform-hazard null over each clip's armed interval.
3. False alarms in the simulator's units: rate while ARMED (warm-up and cooldown excluded) per
   source, and the implied long-run rate with the cooldown (audit, D-027).
4. ACCIDENT temporal score of a constant guess (median impact time of the train/val clips).
5. Detection interval clustered by source video; suggested phase-10 sensitivity ranges.
Output: ``results/vision/v5/exploratory.json``.
"""

import argparse
import json

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.vision.alarm import verified_alarm_times
from resqcity.vision.datasets.accident import source_id
from resqcity.vision.eval.evaluate import add_scores, alarm_cfg
from resqcity.vision.eval.final_v5 import SIGMAS, temporal_score
from resqcity.vision.eval.metrics import (
    ACCIDENT_PRE_TOLERANCE_S,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    DETECTION_GRACE_S,
    SECONDS_PER_HOUR,
    poisson_rate_interval,
    wilson_interval,
)
from resqcity.vision.eval.select_v5 import SWEEPS, CachedVerifier, run_config
from resqcity.vision.scoring import Stacker, logit
from resqcity.vision.train_clip import score_videos

V5 = PROJECT_ROOT / "results" / "vision" / "v5"
CACHE = PROJECT_ROOT / "data" / "processed" / "vision"


def reconstruct_alarms(pc: pd.DataFrame, nd: pd.DataFrame) -> tuple[dict, dict, object, dict, int]:
    """Verified alarm times of the final run for every holdout clip and UA-DETRAC sequence."""
    b = json.loads((V5 / "stage_b.json").read_text(encoding="utf-8"))
    run, tag, op = b["chosen_run"], b["detector_tag"], b["operating_point"]
    cfg = run_config(run)
    ann = pd.read_parquet(CACHE / "annotations.parquet")
    vids = ann[ann["video_id"].isin(set(pc["video_id"]) | set(nd["video_id"]))]
    w = add_scores(
        score_videos(PROJECT_ROOT / "data/models/clip" / run, cfg, CACHE, vids), CACHE, tag
    )
    w["clf_logit"] = logit(w["clf_p"])
    w["combined_p"] = Stacker.load(V5 / "stacker.json").proba(w)
    cached = pd.read_parquet(
        CACHE / "verifier" / b["verifier"] / f"{run}_{tag or 'coco'}_holdout.parquet"
    )
    score = {(r.video_id, r.t): r.p_yes for r in cached.itertuples()}
    misses: list[tuple[str, float]] = []
    detrac_ids = set(nd["video_id"])
    rescore_path = CACHE / "verifier" / b["verifier"] / f"{run}_{tag or 'coco'}_ua_detrac.parquet"
    if rescore_path.exists():  # earlier re-scores: a re-run needs no VLM
        score |= {(r.video_id, r.t): r.p_yes for r in pd.read_parquet(rescore_path).itertuples()}
    rescorer: list[CachedVerifier] = []  # created on first need (loads the VLM)

    def ok(vid: str, t: float) -> bool:
        key = (vid, round(float(t), 3))
        if key in score:
            return score[key] >= b["tau"]
        if vid not in detrac_ids:
            misses.append(key)
            return False
        # The final run saved its verifier cache before scoring UA-DETRAC, so those scores were
        # never written; re-score them (no labels: normal traffic only). The per-sequence alarm
        # counts are then checked against the final run.
        if not rescorer:
            rescorer.append(
                CachedVerifier(
                    b["verifier"],
                    cfg,
                    vids[vids["video_id"].isin(detrac_ids)],
                    PROJECT_ROOT / "data" / "raw" / "vision",
                    rescore_path,
                )
            )
        return rescorer[0](vid, t) >= b["tau"]

    alarms = {}
    for vid, g in w.groupby("video_id"):
        g = g.sort_values("end_s")
        alarms[vid] = verified_alarm_times(
            g["end_s"].tolist(),
            g["combined_p"].tolist(),
            alarm_cfg(cfg, op["consecutive_windows"]),
            op["threshold"],
            lambda t, v=vid: ok(v, t),
        )
    if misses:
        raise RuntimeError(f"{len(misses)} ACCIDENT verifier scores missing from the cache")
    if rescorer:
        rescorer[0].save()
    rescored = len(pd.read_parquet(rescore_path)) if rescore_path.exists() else 0
    durations = dict(zip(vids["video_id"], vids["duration_s"], strict=True))
    return alarms, durations, cfg, op, rescored


def check_reproduction(alarms: dict, pc: pd.DataFrame, nd: pd.DataFrame) -> None:
    for r in pc.itertuples():
        a = alarms.get(r.video_id, [])
        pre = sum(t < r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S for t in a)
        hit = any(
            r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S
            <= t
            <= r.accident_onset_s + DETECTION_GRACE_S
            for t in a
        )
        if pre != r.road_false_alarms or hit != bool(r.detected):
            raise RuntimeError(f"reconstruction differs from the final run for {r.video_id}")
    for r in nd.itertuples():
        if len(alarms.get(r.video_id, [])) != r.road_false_alarms:
            raise RuntimeError(f"reconstruction differs from the final run for {r.video_id}")


def lead_time_breakdown(pc: pd.DataFrame, warm_up: float) -> dict[str, object]:
    lead = pc["accident_onset_s"] - ACCIDENT_PRE_TOLERANCE_S >= warm_up + 1.0
    out = {}
    for name, g in (("with_lead_time", pc[lead]), ("short_lead_time", pc[~lead])):
        early = g["road_false_alarms"] > 0
        out[name] = {
            "clips": len(g),
            "detected": int(g["detected"].sum()),
            "clips_with_pre_impact_alarm": int(early.sum()),
            "detected_among_them": int(g[early]["detected"].sum()),
            "note_detected_among_them": "0 by construction: the 60 s cooldown follows the alarm",
            "clips_without_pre_impact_alarm": int((~early).sum()),
            "detected_without_pre_impact_alarm": int(g[~early]["detected"].sum()),
        }
    return out


def timing_vs_uniform_null(pc: pd.DataFrame, alarms: dict, warm_up: float) -> dict[str, object]:
    """First pre-impact alarm per clip (later ones are blocked by the cooldown) vs a uniform hazard
    over the armed interval [warm-up, t* - 1 s]."""
    obs, exp3, exp5, before = [], [], [], []
    for r in pc.itertuples():
        end = r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S
        pre = [t for t in alarms.get(r.video_id, []) if t < end]
        if not pre:
            continue
        length = max(end - warm_up, 1e-9)
        before.append(r.accident_onset_s - pre[0])
        exp3.append(min(1.0, (3.0 - ACCIDENT_PRE_TOLERANCE_S) / length))
        exp5.append(min(1.0, (5.0 - ACCIDENT_PRE_TOLERANCE_S) / length))
        obs.append(pre[0])
    s = np.asarray(before)
    return {
        "clips_with_pre_impact_alarm": len(s),
        "seconds_before_impact_median": float(np.median(s)),
        "share_within_3s": float(np.mean(s <= 3)),
        "share_within_5s": float(np.mean(s <= 5)),
        "uniform_null_expected_share_within_3s": float(np.mean(exp3)),
        "uniform_null_expected_share_within_5s": float(np.mean(exp5)),
        "reading": "observed shares above the null would indicate alarms concentrating just "
        "before impact; shares near the null mean the short armed interval explains the timing",
    }


def armed_rates(
    pc: pd.DataFrame,
    nd: pd.DataFrame,
    alarms: dict,
    durations: dict,
    warm_up: float,
    cooldown: float,
) -> dict[str, object]:
    """Alarms per armed hour: warm-up excluded and, after each alarm, the cooldown."""

    def armed_time(alarm_list: list[float], start: float, end: float) -> tuple[int, float]:
        n, t, armed = 0, start, 0.0
        for a in sorted(x for x in alarm_list if start <= x < end):
            if a < t:
                continue  # inside a cooldown (cannot happen for real alarms; defensive)
            armed += a - t
            n += 1
            t = a + cooldown
        return n, armed + max(0.0, end - t)

    acc_n = acc_h = 0.0
    for r in pc.itertuples():
        n, s = armed_time(
            alarms.get(r.video_id, []), warm_up, r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S
        )
        acc_n, acc_h = acc_n + n, acc_h + s / SECONDS_PER_HOUR
    det_n = det_h = 0.0
    for r in nd.itertuples():
        n, s = armed_time(alarms.get(r.video_id, []), warm_up, float(durations[r.video_id]))
        det_n, det_h = det_n + n, det_h + s / SECONDS_PER_HOUR

    def block(n: float, h: float) -> dict[str, float]:
        rate = n / h if h > 0 else float("nan")
        lo, hi = poisson_rate_interval(int(n), h)
        long_run = rate / (1 + rate * cooldown / SECONDS_PER_HOUR)
        return {
            "alarms": int(n),
            "armed_hours": h,
            "rate_per_armed_hour": rate,
            "poisson_ci95": [lo, hi],
            "long_run_per_camera_hour_with_cooldown": long_run,
        }

    return {
        "definition": "Poisson alarms at rate lambda while armed; after each alarm the camera is "
        f"silent for {cooldown:g} s; long-run rate = lambda / (1 + lambda * cooldown)",
        "accident_pre_impact": block(acc_n, acc_h),
        "ua_detrac_normal_traffic": block(det_n, det_h),
        "pooled": block(acc_n + det_n, acc_h + det_h),
    }


def constant_guess_score(pc: pd.DataFrame) -> dict[str, object]:
    ann = pd.read_parquet(CACHE / "annotations.parquet")
    trval = ann[ann["source"].eq("accident") & ann["split"].isin(["train", "val"])]
    c = float(trval["accident_onset_s"].median())
    truth = pc["accident_onset_s"].to_numpy(float)
    return {
        "constant_impact_time_s": c,
        "source": "median impact time of ACCIDENT train/val clips (no holdout information)",
        "temporal_score": {
            f"sigma_{s:g}": temporal_score(np.full_like(truth, c), truth, s) for s in SIGMAS
        },
    }


def clustered_detection(pc: pd.DataFrame) -> dict[str, object]:
    src = pc["video_id"].str.removeprefix("acc_").map(source_id)
    g = pc.assign(src=src).groupby("src")["detected"].agg(["sum", "count"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(g), size=(BOOTSTRAP_REPS, len(g)))
    rates = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return {
        "source_videos": len(g),
        "detection_rate": float(pc["detected"].mean()),
        "cluster_bootstrap_ci95": [
            float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    pc = pd.read_csv(V5 / "per_clip.csv")
    nd = pd.read_csv(V5 / "per_sequence_ua_detrac.csv")
    alarms, durations, cfg, op, rescored = reconstruct_alarms(pc, nd)
    check_reproduction(alarms, pc, nd)
    cc = cfg.clip_classifier
    warm_up = cc.window_s + (op["consecutive_windows"] - 1) * cc.stride_s
    rates = armed_rates(pc, nd, alarms, durations, warm_up, cfg.alarm.cooldown_s)
    det = clustered_detection(pc)
    non_us = pc[~pc["us"].astype(bool)]
    nu_lo, nu_hi = wilson_interval(int(non_us["detected"].sum()), len(non_us))
    all_lo, all_hi = wilson_interval(int(pc["detected"].sum()), len(pc))
    report = {
        "status": "exploratory (not pre-registered), analyses of the final v5 run",
        "reconstruction_matches_final_run": True,
        "ua_detrac_verifier_calls_rescored": rescored,
        "rescore_reason": "the final run saved its verifier cache before scoring UA-DETRAC; "
        "those calls were re-scored with the same verifier (no labels) and every per-sequence "
        "alarm count matched the final run",
        "warm_up_s": warm_up,
        "lead_time": lead_time_breakdown(pc, warm_up),
        "pre_impact_alarm_timing": timing_vs_uniform_null(pc, alarms, warm_up),
        "false_alarms_armed": rates,
        "temporal_constant_guess": constant_guess_score(pc),
        "detection_clustered": det,
        "suggested_phase10_ranges": {
            "detection_rate": [nu_lo, all_hi],
            "detection_rate_basis": "non-US Wilson lower bound to overall Wilson upper bound; "
            f"overall {pc['detected'].mean():.3f} ({all_lo:.3f}-{all_hi:.3f}), non-US "
            f"{non_us['detected'].mean():.3f} ({nu_lo:.3f}-{nu_hi:.3f})",
            "false_alarm_rate_per_armed_hour": [
                rates["ua_detrac_normal_traffic"]["poisson_ci95"][0],
                rates["accident_pre_impact"]["poisson_ci95"][1],
            ],
            "false_alarm_basis": "UA-DETRAC lower bound to ACCIDENT pre-impact upper bound, both "
            "per armed hour; the simulator applies the cooldown",
        },
    }
    out = V5 / "exploratory.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    b = json.loads((V5 / "stage_b.json").read_text(encoding="utf-8"))
    write_meta(
        out,
        configs=[CONFIG_DIR / "vision.yaml", SWEEPS / f"{b['chosen_run']}.yaml"],
        synthetic=False,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
