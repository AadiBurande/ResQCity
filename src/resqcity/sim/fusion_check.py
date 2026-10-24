"""Synthetic-week model check of fusion against the simulator's ground truth (phase 4; SYNTHETIC).

    uv run python -m resqcity.sim.fusion_check [--scenario normal] [--reps 1]

Generator -> calls (sim.calls) + camera events (sim.detection) -> rule triage -> fusion engine,
on the Pune network with the 40 camera sites, using the TUNING seed base (evaluation seeds stay
untouched for the confirmatory experiments). Reports, per replication and pooled:
- association: pairwise precision / recall of "same incident" over evidence items;
- calibration: final posterior of each candidate vs whether it holds genuine evidence;
- time from onset to posterior >= 0.5 for real incidents, by evidence path;
- false candidates per day, and how many of them reach 0.5.
These check the model and its inputs; they are not research results (D-028).
Output: results/fusion/week_check_synthetic.json.
"""

import argparse
import itertools
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.requirements import load_requirements_config
from resqcity.fusion.config import (
    load_calls_config,
    load_detector_model,
    load_fusion_config,
    load_templates,
    load_triage_config,
)
from resqcity.fusion.engine import FusionEngine
from resqcity.fusion.triage import RuleTriager
from resqcity.geo.cameras import camera_coverage, load_cameras
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.geo.scenario import ScenarioState, load_scenario
from resqcity.sim.background import build_background
from resqcity.sim.calls import CallSimulator
from resqcity.sim.detection import DetectionProcess
from resqcity.sim.generator import IncidentGenerator, load_incidents_config

OUT = PROJECT_ROOT / "results" / "fusion" / "week_check_synthetic.json"
THRESHOLD = 0.5
CAL_EDGES = [0.0, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0]
EXPIRY_TICK = timedelta(minutes=5)
log = logging.getLogger(__name__)


def pair_scores(groups: dict[str, list[str | None]]) -> dict[str, float]:
    """Association quality over evidence items (truth label = incident id, None = false evidence;
    prediction = candidate). Recall: share of same-incident pairs that share a candidate.
    Precision: among pairs in one candidate that involve genuine evidence, the share from the
    same incident. Recurring false alarms of one camera sharing a candidate are by design and
    are not counted; false evidence absorbed into real candidates is reported separately."""
    tp = genuine_pairs = 0
    truth_pairs: Counter = Counter()
    false_in_real = 0
    for labels in groups.values():
        counts = Counter(lab for lab in labels if lab is not None)
        n_real, n_false = sum(counts.values()), labels.count(None)
        tp += sum(c * (c - 1) // 2 for c in counts.values())
        genuine_pairs += n_real * (n_real - 1) // 2 + n_real * n_false
        truth_pairs.update(counts)
        false_in_real += n_false if n_real else 0
    total_truth = sum(c * (c - 1) // 2 for c in truth_pairs.values())
    return {
        "pair_precision": tp / genuine_pairs if genuine_pairs else 1.0,
        "pair_recall": tp / total_truth if total_truth else 1.0,
        "same_incident_pairs": total_truth,
        "false_items_in_real_candidates": false_in_real,
    }


def run(rep: int, scenario: str, network: RoadNetwork, seed_base: int) -> dict:
    inc_cfg, req = load_incidents_config(), load_requirements_config()
    calls_cfg, fusion = load_calls_config(), load_fusion_config()
    detector = load_detector_model(PROJECT_ROOT / fusion.detector_inputs)
    sc = load_scenario(scenario)
    state = ScenarioState(sc, network)
    gen = IncidentGenerator(network, state, inc_cfg, req)
    incidents = gen.generate(seed_base, rep)
    start, hours = sc.start, sc.days * 24
    cams = load_cameras(CONFIG_DIR / "cameras.yaml")
    cover = camera_coverage(network, cams, fusion.camera.coverage_radius_m.value)
    background = build_background(network, inc_cfg, calls_cfg, sc.incident_rate_multiplier)
    mix = background.type_share
    caller = CallSimulator(network, calls_cfg, load_templates(), req, gen.node_share, mix)
    calls = caller.generate(incidents, start, hours, seed_base, rep)
    events = DetectionProcess(network, cams, detector, fusion, req).generate(
        incidents, start, hours, seed_base, rep
    )
    engine = FusionEngine(network, cams, detector, calls_cfg, fusion, req, background, cover)
    triager = RuleTriager(load_triage_config(), req)
    stream = sorted(
        [(e.event.ts, 0, e.event.id, e) for e in events]
        + [(c.call.ts, 1, c.call.id, c) for c in calls]
    )
    truth: dict[str, str | None] = {}
    first_reach: dict[str, object] = {}  # candidate -> first time posterior >= threshold
    path: dict[str, list[str]] = defaultdict(list)
    next_expiry = start + EXPIRY_TICK
    t0 = time.perf_counter()
    for ts, kind, eid, item in stream:
        while ts >= next_expiry:
            engine.tick(next_expiry, refresh=False)
            next_expiry += EXPIRY_TICK
        if kind == 0:
            out = engine.on_event(item.event)
        else:
            out = engine.on_call(item.call, triager.triage(item.call))
        truth[eid] = item.incident_id
        for inc in out:
            path[inc.id].append("camera" if kind == 0 else "call")
            if inc.posterior >= THRESHOLD and inc.id not in first_reach:
                first_reach[inc.id] = ts
    elapsed = time.perf_counter() - t0
    end = start + timedelta(hours=hours)
    onset = {i.id: i.onset_ts for i in incidents}
    covered_ids = {
        i.id for i in incidents if i.type in fusion.camera.detected_types and any(
            network.index_of(i.node) in set(cover[c.id].tolist()) for c in cams
        )
    }  # fmt: skip

    groups: dict[str, list[str | None]] = {}
    final: list[tuple[float, bool]] = []
    reach_by_path: dict[str, list[float]] = defaultdict(list)
    n_real_cands = Counter()
    false_cands = false_reach = 0
    for cand in engine.candidates.values():
        ids = [e.id for e in cand.events] + [c.id for c, _ in cand.calls]
        labels = [truth[i] for i in ids]
        groups[cand.id] = labels
        real = Counter(lab for lab in labels if lab is not None)
        posterior = engine.model.posterior(engine.evidence(cand, min(cand.updated, end)))[0]
        final.append((posterior, bool(real)))
        if not real:
            false_cands += 1
            false_reach += cand.id in first_reach
            continue
        inc_id = real.most_common(1)[0][0]
        kinds = path[cand.id]
        p = "call only" if "camera" not in kinds else (
            "camera only" if "call" not in kinds else (
                "camera then call" if kinds.index("camera") < kinds.index("call") else
                "call then camera"))  # fmt: skip
        n_real_cands[p] += 1
        if cand.id in first_reach:
            reach_by_path[p].append((first_reach[cand.id] - onset[inc_id]).total_seconds())
    cal = []
    post = np.array([p for p, _ in final])
    real_flag = np.array([r for _, r in final])
    for lo, hi in itertools.pairwise(CAL_EDGES):
        sel = (post >= lo) & (post < hi if hi < 1.0 else post <= hi)
        cal.append({
            "bin": [lo, hi], "candidates": int(sel.sum()),
            "mean_posterior": float(post[sel].mean()) if sel.any() else None,
            "share_real": float(real_flag[sel].mean()) if sel.any() else None,
        })  # fmt: skip
    days = hours / 24.0
    return {
        "replication": rep,
        "incidents": len(incidents),
        "incidents_in_camera_view": len(covered_ids),
        "calls": len(calls),
        "false_calls": sum(c.incident_id is None for c in calls),
        "camera_events": len(events),
        "true_camera_events": sum(e.incident_id is not None for e in events),
        "candidates": len(engine.candidates),
        "association": pair_scores(groups),
        "calibration": cal,
        "real_candidates_by_path": dict(n_real_cands),
        "onset_to_threshold_s": {
            p: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
            for p, v in reach_by_path.items()
        },
        "real_incidents_reaching_threshold": len(
            {
                Counter(lab for lab in groups[c] if lab).most_common(1)[0][0]
                for c in first_reach
                if any(groups[c])
            }
        ),
        "false_candidates_per_day": false_cands / days,
        "false_candidates_reaching_threshold_per_day": false_reach / days,
        "fusion_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    seed_base = base.random.tuning_seed_base
    reps = []
    for r in range(args.reps):
        reps.append(run(r, args.scenario, network, seed_base))
        log.info("replication %d: %s", r, {k: v for k, v in reps[-1].items() if k in (
            "incidents", "calls", "camera_events", "candidates", "association")})  # fmt: skip
    report = {
        "status": "synthetic model check (UNVERIFIED call and review parameters; measured "
        "detector); not a research result",
        "scenario": args.scenario,
        "seed_base": "tuning_seed_base",
        "threshold": THRESHOLD,
        "replications": reps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    configs = [
        CONFIG_DIR / n
        for n in ("fusion.yaml", "calls.yaml", "call_templates.yaml", "triage.yaml",
                  "incidents.yaml", "requirements.yaml", "cameras.yaml",
                  f"scenarios/{args.scenario}.yaml")
    ]  # fmt: skip
    write_meta(OUT, configs=configs, synthetic=True, seeds={"tuning_seed_base": seed_base})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
