"""Exploratory analyses of the FINAL v5 run (not pre-registered; labelled as exploratory, D-027).

    uv run python -m resqcity.vision.eval.explore_v5

Uses ``results/vision/v5/per_clip.csv`` (labels written by the final run), the saved stacker and
the cached verifier scores; the sealed holdout label file is NOT read again. Every alarm of the
final run is reconstructed and checked against ``per_clip.csv`` (counts, detection and detection
time) and ``per_sequence_ua_detrac.csv`` (counts) first; the analysis stops if they disagree.
1. Lead time: detection on clips whose impact leaves the warm-up + 1 s, split by whether an alarm
   fired before t* - 1 s (the 60 s cooldown then hides a later detection by construction).
2. Clip-start firing (re-audit): detections at the first instant an alarm can fire, a
   content-free baseline that alarms at that instant on every clip, detection by impact time,
   and detection when the camera is armed at the impact (no earlier alarm).
3. Timing of pre-impact alarms against a uniform alarm-time null over each clip's armed interval.
4. False alarms in the simulator's units: rate while ARMED (warm-up and cooldown excluded) per
   source, with Poisson and unit-clustered bootstrap intervals, and the implied long-run rate
   with the cooldown (audit, D-027).
5. ACCIDENT temporal score of a constant guess (median impact time of the train/val clips).
6. Detection interval clustered by source video; simulator defaults and phase-10 ranges.
Outputs ``results/vision/v5/``: exploratory.json, alarms_final.csv (every reconstructed alarm),
armed_time_per_unit.csv.
"""

import argparse
import json
import math

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.vision.alarm import verified_alarm_times
from resqcity.vision.datasets.accident import source_id
from resqcity.vision.eval.evaluate import add_scores, alarm_cfg
from resqcity.vision.eval.final_v5 import NON_US_REGIONS, SIGMAS, temporal_score
from resqcity.vision.eval.metrics import (
    ACCIDENT_PRE_TOLERANCE_S,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    DETECTION_GRACE_S,
    SECONDS_PER_HOUR,
    cluster_bootstrap_rate,
    poisson_rate_interval,
    wilson_interval,
)
from resqcity.vision.eval.select_v5 import SWEEPS, CachedVerifier, run_config
from resqcity.vision.scoring import Stacker, logit
from resqcity.vision.train_clip import score_videos

V5 = PROJECT_ROOT / "results" / "vision" / "v5"
CACHE = PROJECT_ROOT / "data" / "processed" / "vision"
IMPACT_BANDS_S = (0.0, 3.0, 5.0, 7.0, 9.0, 15.0, math.inf)
TIME_TOL_S = 1e-6


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


def detection_time(alarm_list: list[float], onset_s: float) -> float | None:
    """First alarm inside the detection interval [t* - 1 s, t* + grace], else None."""
    lo, hi = onset_s - ACCIDENT_PRE_TOLERANCE_S, onset_s + DETECTION_GRACE_S
    return next((t for t in sorted(alarm_list) if lo <= t <= hi), None)


def check_reproduction(alarms: dict, pc: pd.DataFrame, nd: pd.DataFrame) -> None:
    """Counts, detection and detection time per clip; counts per UA-DETRAC sequence (the final
    run stored no alarm times for UA-DETRAC)."""
    for r in pc.itertuples():
        a = alarms.get(r.video_id, [])
        pre = sum(t < r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S for t in a)
        hit = detection_time(a, r.accident_onset_s)
        same_time = hit is None or abs(hit - (r.accident_onset_s + r.delay_s)) < 1e-3
        if pre != r.road_false_alarms or (hit is not None) != bool(r.detected) or not same_time:
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


def content_free_detected(
    onset_s: np.ndarray, duration_s: np.ndarray, warm_up: float
) -> np.ndarray:
    """Baseline that ignores the video: one alarm at the first possible instant (the warm-up) on
    every clip long enough to have one. Detected by the pre-registered rule if that instant falls
    in [t* - 1 s, t* + grace]."""
    fires = duration_s >= warm_up - TIME_TOL_S
    lo, hi = onset_s - ACCIDENT_PRE_TOLERANCE_S, onset_s + DETECTION_GRACE_S
    return fires & (lo <= warm_up) & (warm_up <= hi)


def _rate(k: int, n: int) -> dict[str, object]:
    lo, hi = wilson_interval(k, n)
    return {
        "detected": k,
        "clips": n,
        "rate": k / n if n else float("nan"),
        "wilson_ci95": [lo, hi],
    }


def clip_start_analysis(
    pc: pd.DataFrame, alarms: dict, durations: dict, warm_up: float
) -> dict[str, object]:
    onset = pc["accident_onset_s"].to_numpy(float)
    dur = pc["video_id"].map(durations).to_numpy(float)
    hits = [
        detection_time(alarms.get(v, []), t) for v, t in zip(pc["video_id"], onset, strict=True)
    ]
    hit_t = np.array([np.nan if h is None else h for h in hits])
    at_first = np.isclose(hit_t, warm_up, atol=TIME_TOL_S)
    base = content_free_detected(onset, dur, warm_up)
    us = pc["us"].astype(bool).to_numpy()
    det = pc["detected"].astype(bool).to_numpy()
    band = pd.cut(pc["accident_onset_s"], IMPACT_BANDS_S, right=True)
    by_band = {
        str(k): {
            "clips": len(g),
            "detection_rate": float(g["detected"].mean()),
            "content_free_rate": float(base[g.index].mean()),
        }
        for k, g in pc.groupby(band, observed=True)
    }
    # pre-impact alarms: when do they fire after the warm-up?
    has_pre_footage = onset - ACCIDENT_PRE_TOLERANCE_S > warm_up
    first_pre = [
        min(t for t in alarms[v] if t < o - ACCIDENT_PRE_TOLERANCE_S) - warm_up
        for v, o, n in zip(pc["video_id"], onset, pc["road_false_alarms"], strict=True)
        if n > 0
    ]
    fp = np.asarray(first_pre)
    # camera armed at the impact: enough lead time and no earlier alarm (no cooldown running)
    lead = onset - ACCIDENT_PRE_TOLERANCE_S >= warm_up + 1.0
    armed = lead & (pc["road_false_alarms"].to_numpy() == 0)
    delays = pc.loc[armed & det, "delay_s"].astype(float)
    return {
        "first_possible_alarm_s": warm_up,
        "detections_at_first_possible_instant": int(at_first.sum()),
        "detections": int(det.sum()),
        "content_free_baseline": {
            "rule": "one alarm at the warm-up end on every clip at least that long",
            "overall": _rate(int(base.sum()), len(pc)),
            "us": _rate(int(base[us].sum()), int(us.sum())),
            "non_us": _rate(int(base[~us].sum()), int((~us).sum())),
            "clips_shorter_than_warm_up": int((dur < warm_up - TIME_TOL_S).sum()),
        },
        "detector_overall": _rate(int(det.sum()), len(pc)),
        "detection_by_impact_time_s": by_band,
        "pre_impact": {
            "clips_with_pre_impact_footage": int(has_pre_footage.sum()),
            "clips_with_pre_impact_alarm": len(fp),
            "first_alarm_s_after_warm_up_mean": float(fp.mean()) if len(fp) else None,
            "first_alarm_s_after_warm_up_median": float(np.median(fp)) if len(fp) else None,
            "share_at_first_possible_instant": float(np.mean(fp < TIME_TOL_S)) if len(fp) else None,
        },
        "detection_when_armed": {
            "definition": "clips with impact >= warm-up + 2 s and no alarm before t* - 1 s, so "
            "the camera is armed (no cooldown) when the crash happens and the first window "
            "cannot contain the impact",
            **_rate(int((armed & det).sum()), int(armed.sum())),
            "non_us": _rate(int((armed & det & ~us).sum()), int((armed & ~us).sum())),
            "delay_median_s": float(delays.median()) if len(delays) else None,
        },
        "detection_with_lead_time_incl_masking": _rate(int((lead & det).sum()), int(lead.sum())),
    }


def uniform_share_within(seconds_before: float, armed_s: float) -> float:
    """P(an alarm uniform on [warm-up, t* - 1 s] lies within ``seconds_before`` of t*)."""
    span = seconds_before - ACCIDENT_PRE_TOLERANCE_S
    return min(1.0, max(0.0, span) / max(armed_s, 1e-9))


def timing_vs_uniform_null(pc: pd.DataFrame, alarms: dict, warm_up: float) -> dict[str, object]:
    """First pre-impact alarm per clip (later ones are blocked by the cooldown) vs a uniform
    alarm-time null over the armed interval [warm-up, t* - 1 s]."""
    exp3, exp5, before = [], [], []
    for r in pc.itertuples():
        end = r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S
        pre = [t for t in alarms.get(r.video_id, []) if t < end]
        if not pre:
            continue
        before.append(r.accident_onset_s - min(pre))
        exp3.append(uniform_share_within(3.0, end - warm_up))
        exp5.append(uniform_share_within(5.0, end - warm_up))
    s = np.asarray(before)
    return {
        "clips_with_pre_impact_alarm": len(s),
        "seconds_before_impact_median": float(np.median(s)),
        "share_within_3s": float(np.mean(s <= 3)),
        "share_within_5s": float(np.mean(s <= 5)),
        "uniform_null_expected_share_within_3s": float(np.mean(exp3)),
        "uniform_null_expected_share_within_5s": float(np.mean(exp5)),
        "reading": "shares below the null mean the alarms come EARLY in the armed interval (see "
        "clip_start.pre_impact: most fire at the first possible instant), not near the impact",
    }


def armed_time(
    alarm_list: list[float], start: float, end: float, cooldown: float
) -> tuple[int, float]:
    """Alarms in [start, end) and the seconds the camera was ARMED (not in a cooldown)."""
    n, t, armed = 0, start, 0.0
    for a in sorted(x for x in alarm_list if start <= x < end):
        if a < t:
            continue  # inside a cooldown (cannot happen for real alarms; defensive)
        armed += a - t
        n += 1
        t = a + cooldown
    return n, armed + max(0.0, end - t)


def _at(alarm_list: list[float], warm_up: float, end: float) -> bool:
    return warm_up < end and any(abs(t - warm_up) < TIME_TOL_S for t in alarm_list)


def armed_units(
    pc: pd.DataFrame, nd: pd.DataFrame, alarms: dict, durations: dict, warm_up: float, cd: float
) -> pd.DataFrame:
    rows = []
    for r in pc.itertuples():
        end = r.accident_onset_s - ACCIDENT_PRE_TOLERANCE_S
        a = alarms.get(r.video_id, [])
        n, s = armed_time(a, warm_up, end, cd)
        rows.append({"source": "accident_pre_impact", "video_id": r.video_id, "alarms": n,
                     "armed_s": s, "alarm_at_warm_up": _at(a, warm_up, end)})  # fmt: skip
    for r in nd.itertuples():
        a, end = alarms.get(r.video_id, []), float(durations[r.video_id])
        n, s = armed_time(a, warm_up, end, cd)
        rows.append({"source": "ua_detrac_normal_traffic", "video_id": r.video_id, "alarms": n,
                     "armed_s": s, "alarm_at_warm_up": _at(a, warm_up, end)})  # fmt: skip
    return pd.DataFrame(rows)


def rate_summary(units: pd.DataFrame, cooldown: float) -> dict[str, object]:
    n, h = int(units["alarms"].sum()), float(units["armed_s"].sum()) / SECONDS_PER_HOUR
    rate = n / h if h > 0 else float("nan")
    return {
        "units": len(units),
        "units_with_alarm": int((units["alarms"] > 0).sum()),
        "units_with_2plus_alarms": int((units["alarms"] > 1).sum()),
        "units_alarming_at_first_possible_instant": int(units["alarm_at_warm_up"].sum()),
        "alarms": n,
        "armed_hours": h,
        "rate_per_armed_hour": rate,
        "poisson_ci95": list(poisson_rate_interval(n, h)),
        "cluster_bootstrap_ci95": list(
            cluster_bootstrap_rate(
                units["alarms"].to_numpy(), units["armed_s"].to_numpy() / SECONDS_PER_HOUR
            )
        ),
        "long_run_per_camera_hour_with_cooldown": rate / (1 + rate * cooldown / SECONDS_PER_HOUR),
    }


def armed_rates(units: pd.DataFrame, cooldown: float) -> dict[str, object]:
    return {
        "definition": "Poisson alarms at rate lambda while armed; after each alarm the camera is "
        f"silent for {cooldown:g} s; long-run rate = lambda / (1 + lambda * cooldown). Cluster "
        "intervals resample clips / sequences (UA-DETRAC recording sites are not identified).",
        "accident_pre_impact": rate_summary(
            units[units["source"].eq("accident_pre_impact")], cooldown
        ),
        "ua_detrac_normal_traffic": rate_summary(
            units[units["source"].eq("ua_detrac_normal_traffic")], cooldown
        ),
        "always_alarm_ceiling_per_camera_hour": SECONDS_PER_HOUR / cooldown,
        "accident_pre_impact_reading": "not a hazard rate: about half of the clips with "
        "pre-impact footage alarm at the first possible instant, so this mostly measures "
        "clip-start firing on short trimmed clips (clip_start.pre_impact)",
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


def chance_detection(rate_per_hour: float) -> float:
    """P(an armed camera alarming as a Poisson process at ``rate_per_hour`` fires at least once
    in the detection interval [t* - 1 s, t* + grace]) - detection by chance alone."""
    window_s = ACCIDENT_PRE_TOLERANCE_S + DETECTION_GRACE_S
    return 1.0 - math.exp(-rate_per_hour * window_s / SECONDS_PER_HOUR)


def chance_levels(units: pd.DataFrame, rates: dict) -> dict[str, object]:
    """Chance detection for the armed-camera estimate (closing audit, D-027)."""
    ua = rates["ua_detrac_normal_traffic"]
    acc = units[units["source"].eq("accident_pre_impact")]
    later = int(acc["alarms"].sum() - acc["alarm_at_warm_up"].sum())
    later_rate = later / (float(acc["armed_s"].sum()) / SECONDS_PER_HOUR)
    return {
        "definition": "P(>= 1 Poisson alarm in the "
        f"{ACCIDENT_PRE_TOLERANCE_S + DETECTION_GRACE_S:g} s detection interval)",
        "at_ua_detrac_rate": chance_detection(ua["rate_per_armed_hour"]),
        "at_ua_detrac_cluster_upper": chance_detection(ua["cluster_bootstrap_ci95"][1]),
        "accident_pre_impact_rate_after_first_instant_per_hour": later_rate,
        "at_accident_pre_impact_rate_after_first_instant": chance_detection(later_rate),
    }


def simulator_defaults(clip: dict, rates: dict, cooldown: float) -> dict[str, object]:
    armed = clip["detection_when_armed"]
    lead = clip["detection_with_lead_time_incl_masking"]
    ua = rates["ua_detrac_normal_traffic"]
    low = min(lead["wilson_ci95"][0], armed["non_us"]["wilson_ci95"][0])
    return {
        "status": "exploratory (not pre-registered); replaces the pre-registered ACCIDENT "
        "detection rate, which mostly reflects clip trimming (content-free baseline above it)",
        "detection_probability_when_armed": armed["rate"],
        "detection_probability_ci95": armed["wilson_ci95"],
        "detection_basis": "ACCIDENT holdout clips with lead time and no earlier alarm "
        f"({armed['detected']}/{armed['clips']}); the simulator models masking by the cooldown "
        "itself through the false-alarm process",
        "delay_median_s": armed["delay_median_s"],
        "false_alarm_source": "ua_detrac_normal_traffic",
        "false_alarm_rate_per_armed_hour": ua["rate_per_armed_hour"],
        "false_alarm_rate_ci95": ua["cluster_bootstrap_ci95"],
        "cooldown_s": cooldown,
        "phase10_ranges": {
            "detection_probability": [low, armed["wilson_ci95"][1]],
            "detection_basis": "lower: the lower of the Wilson low of lead-time clips counting "
            f"masked crashes as misses ({lead['detected']}/{lead['clips']}) and of non-US armed "
            f"clips ({armed['non_us']['detected']}/{armed['non_us']['clips']}); upper: armed "
            "clips (Wilson high)",
            "false_alarm_rate_per_armed_hour": ua["cluster_bootstrap_ci95"],
            "false_alarm_basis": "UA-DETRAC sequence-clustered bootstrap interval",
            "false_alarm_stress_per_camera_hour": rates["always_alarm_ceiling_per_camera_hour"],
            "stress_basis": "a camera that alarms at every opportunity (one alarm per cooldown)",
        },
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    pc = pd.read_csv(V5 / "per_clip.csv")
    nd = pd.read_csv(V5 / "per_sequence_ua_detrac.csv")
    if set(pc.loc[~pc["us"].astype(bool), "region"]) - NON_US_REGIONS:
        raise RuntimeError("per_clip.csv regions do not match NON_US_REGIONS")
    alarms, durations, cfg, op, rescored = reconstruct_alarms(pc, nd)
    check_reproduction(alarms, pc, nd)
    cc = cfg.clip_classifier
    cooldown = cfg.alarm.cooldown_s
    warm_up = cc.window_s + (op["consecutive_windows"] - 1) * cc.stride_s
    units = armed_units(pc, nd, alarms, durations, warm_up, cooldown)
    rates = armed_rates(units, cooldown)
    clip = clip_start_analysis(pc, alarms, durations, warm_up)
    report = {
        "status": "exploratory (not pre-registered), analyses of the final v5 run",
        "reconstruction_matches_final_run": True,
        "reconstruction_checks": "ACCIDENT: false-alarm count, detection and detection time per "
        "clip; UA-DETRAC: alarm count per sequence (no times were stored by the final run)",
        "ua_detrac_verifier_calls_rescored": rescored,
        "rescore_reason": "the final run saved its verifier cache before scoring UA-DETRAC; "
        "those calls were re-scored with the same verifier (no labels) and every per-sequence "
        "alarm count matched the final run",
        "warm_up_s": warm_up,
        "lead_time": lead_time_breakdown(pc, warm_up),
        "clip_start": clip,
        "chance_detection_when_armed": chance_levels(units, rates),
        "pre_impact_alarm_timing": timing_vs_uniform_null(pc, alarms, warm_up),
        "false_alarms_armed": rates,
        "temporal_constant_guess": constant_guess_score(pc),
        "detection_clustered": clustered_detection(pc),
        "simulator_defaults": simulator_defaults(clip, rates, cooldown),
    }
    b = json.loads((V5 / "stage_b.json").read_text(encoding="utf-8"))
    configs = [CONFIG_DIR / "vision.yaml", SWEEPS / f"{b['chosen_run']}.yaml"]
    seeds = {"bootstrap": BOOTSTRAP_SEED}
    out = V5 / "exploratory.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    alarm_rows = [
        {
            "video_id": v,
            "source": "ua_detrac" if v in set(nd["video_id"]) else "accident",
            "alarm_s": t,
        }
        for v in sorted(alarms)
        for t in alarms[v]
    ]
    pd.DataFrame(alarm_rows).to_csv(V5 / "alarms_final.csv", index=False, lineterminator="\n")
    units.to_csv(V5 / "armed_time_per_unit.csv", index=False, lineterminator="\n")
    for f in (out, V5 / "alarms_final.csv", V5 / "armed_time_per_unit.csv"):
        write_meta(f, configs=configs, synthetic=False, seeds=seeds)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
