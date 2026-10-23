"""Exploratory analyses of the FINAL v5 run (not pre-registered; labelled as exploratory, D-027).

    uv run python -m resqcity.vision.eval.explore_v5

Uses only ``results/vision/v5/per_clip.csv`` (written by the final run), the saved stacker and the
cached verifier scores; the sealed holdout label file is not read again.
1. Detection by lead time: clips whose impact leaves at least the warm-up + 1 s, split by whether
   an alarm fired before t* - 1 s (the 60 s cooldown then hides a later detection).
2. Timing of the pre-impact alarms: seconds before the impact.
Output: ``results/vision/v5/exploratory.json``.
"""

import argparse
import json

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.vision.alarm import verified_alarm_times
from resqcity.vision.eval.evaluate import add_scores, alarm_cfg
from resqcity.vision.eval.metrics import ACCIDENT_PRE_TOLERANCE_S
from resqcity.vision.eval.select_v5 import SWEEPS, run_config
from resqcity.vision.scoring import Stacker, logit
from resqcity.vision.train_clip import score_videos

V5 = PROJECT_ROOT / "results" / "vision" / "v5"


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
            "clips_without_pre_impact_alarm": int((~early).sum()),
            "detected_without_pre_impact_alarm": int(g[~early]["detected"].sum()),
        }
    return out


def pre_impact_alarm_timing(pc: pd.DataFrame) -> dict[str, object]:
    b = json.loads((V5 / "stage_b.json").read_text(encoding="utf-8"))
    run, tag, op = b["chosen_run"], b["detector_tag"], b["operating_point"]
    cfg = run_config(run)
    cache = PROJECT_ROOT / "data" / "processed" / "vision"
    ann = pd.read_parquet(cache / "annotations.parquet")
    stacker = Stacker.load(V5 / "stacker.json")
    vids = ann[ann["video_id"].isin(pc[pc["road_false_alarms"] > 0]["video_id"])]
    w = score_videos(PROJECT_ROOT / "data" / "models" / "clip" / run, cfg, cache, vids)
    w = add_scores(w, cache, tag)
    w["clf_logit"] = logit(w["clf_p"])
    w["combined_p"] = stacker.proba(w)
    cached = pd.read_parquet(
        cache / "verifier" / b["verifier"] / f"{run}_{tag or 'coco'}_holdout.parquet"
    )
    score = {(r.video_id, r.t): r.p_yes for r in cached.itertuples()}
    onset = dict(zip(pc["video_id"], pc["accident_onset_s"], strict=True))
    before = []
    for vid, g in w.groupby("video_id"):
        g = g.sort_values("end_s")
        alarms = verified_alarm_times(
            g["end_s"].tolist(),
            g["combined_p"].tolist(),
            alarm_cfg(cfg, op["consecutive_windows"]),
            op["threshold"],
            lambda t, v=vid: score.get((v, round(float(t), 3)), 0.0) >= b["tau"],
        )
        before += [onset[vid] - t for t in alarms if t < onset[vid] - ACCIDENT_PRE_TOLERANCE_S]
    s = np.asarray(before)
    return {
        "pre_impact_alarms": len(s),
        "seconds_before_impact_median": float(np.median(s)) if len(s) else None,
        "share_within_3s": float(np.mean(s <= 3)) if len(s) else None,
        "share_within_5s": float(np.mean(s <= 5)) if len(s) else None,
        "share_over_10s": float(np.mean(s > 10)) if len(s) else None,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    pc = pd.read_csv(V5 / "per_clip.csv")
    b = json.loads((V5 / "stage_b.json").read_text(encoding="utf-8"))
    cc = run_config(b["chosen_run"]).clip_classifier
    warm_up = cc.window_s + (b["operating_point"]["consecutive_windows"] - 1) * cc.stride_s
    report = {
        "status": "exploratory (not pre-registered), analyses of the final v5 run",
        "warm_up_s": warm_up,
        "lead_time": lead_time_breakdown(pc, warm_up),
        "pre_impact_alarm_timing": pre_impact_alarm_timing(pc),
    }
    out = V5 / "exploratory.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[CONFIG_DIR / "vision.yaml", SWEEPS / f"{b['chosen_run']}.yaml"],
               synthetic=False)  # fmt: skip
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
