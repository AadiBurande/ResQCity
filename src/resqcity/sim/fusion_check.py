"""Synthetic-week model check of fusion against the simulator's ground truth (phase 4; SYNTHETIC).

    uv run python -m resqcity.sim.fusion_check [--scenario normal] [--reps 1]

Generator -> calls (sim.calls) + camera events (sim.detection) -> rule triage -> fusion engine,
on the Pune network with the 40 camera sites, using the TUNING seed base (evaluation seeds stay
untouched for the confirmatory experiments). Reports, per replication and pooled:
- association: pairwise precision / recall of "same incident" over evidence items;
- calibration: final posterior of each candidate vs whether it holds genuine evidence;
- time from onset to posterior >= 0.5 for real incidents, by the TRUE evidence path (which of
  the candidate's camera events and calls really belong to the incident);
- how often a genuine call lands in a candidate seeded by a camera false alarm;
- whether the FUSED type and required services match the incident (they drive dispatch from
  phase 6 on), and how often one incident ends up as two dispatch-level candidates;
- false candidates per day, and how many of them reach 0.5;
- the detector's self-masking block (``detector_masking``), so D-029's headline is traceable.
Camera coverage is thin (about 2 % of the incident location mass within 50 m road-graph of a
camera junction), so a single week holds well under one incident in camera view: use enough
replications before reading anything about the camera path.
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
FUSED_KEYS = (
    "n", "type_correct", "requirements_exact", "requirements_under", "requirements_over",
)  # fmt: skip
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
        "pairs_together": tp,
        "pairs_in_candidates": genuine_pairs,
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
    kind_of: dict[str, int] = {}
    first_reach: dict[str, object] = {}  # candidate -> first time posterior >= threshold
    seeded_by: dict[str, str] = {}  # candidate -> "true"/"false" evidence that opened it
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
        kind_of[eid] = kind
        for inc in out:
            seeded_by.setdefault(inc.id, "true" if item.incident_id else "false")
            if inc.posterior >= THRESHOLD and inc.id not in first_reach:
                first_reach[inc.id] = ts
    elapsed = time.perf_counter() - t0
    end = start + timedelta(hours=hours)
    onset = {i.id: i.onset_ts for i in incidents}
    covered = {int(v) for a in cover.values() for v in a.tolist()}
    by_id = {i.id: i for i in incidents}
    detectable = [i for i in incidents if i.type in fusion.camera.detected_types]
    covered_ids = {i.id for i in detectable if network.index_of(i.node) in covered}
    coverage_mass = float(np.sum(gen.node_share[sorted(covered)]))

    groups: dict[str, list[str | None]] = {}
    final: list[tuple[float, bool]] = []
    reach_by_path: dict[str, list[float]] = defaultdict(list)
    n_real_cands = Counter()
    fused = Counter()
    reached_incidents: Counter = Counter()
    false_cands = false_reach = seeded_false = 0
    for cand in engine.candidates.values():
        ids = [e.id for e in cand.events] + [c.id for c, _ in cand.calls]
        labels = [truth[i] for i in ids]
        groups[cand.id] = labels
        real = Counter(lab for lab in labels if lab is not None)
        at = min(cand.updated, end)  # the moment of its last evidence; see posterior_evaluated_at
        posterior = engine.model.posterior(engine.evidence(cand, at))[0]
        final.append((posterior, bool(real)))
        if not real:
            false_cands += 1
            false_reach += cand.id in first_reach
            continue
        inc_id = real.most_common(1)[0][0]
        true_kinds = {kind_of[i] for i in ids if truth[i] == inc_id}
        p = {
            frozenset({0}): "true camera only",
            frozenset({1}): "call only",
            frozenset({0, 1}): "true camera + call",
        }[frozenset(true_kinds)]
        n_real_cands[p] += 1
        seeded_false += seeded_by.get(cand.id) == "false"
        truth_inc = by_id[inc_id]
        got = engine.incident(cand, at)
        fused["n"] += 1
        fused["type_correct"] += got.type == truth_inc.type
        want, have = truth_inc.requirements, got.requirements
        under = any(getattr(have, a) < getattr(want, a) for a in ("ambulance", "police", "fire"))
        over = any(getattr(have, a) > getattr(want, a) for a in ("ambulance", "police", "fire"))
        fused["requirements_exact"] += not under and not over
        fused["requirements_under"] += under
        fused["requirements_over"] += over and not under
        if cand.id in first_reach:
            reach_by_path[p].append((first_reach[cand.id] - onset[inc_id]).total_seconds())
            reached_incidents[inc_id] += 1
    cal = []
    post = np.array([p for p, _ in final])
    real_flag = np.array([r for _, r in final])
    for lo, hi in itertools.pairwise(CAL_EDGES):
        sel = (post >= lo) & (post < hi if hi < 1.0 else post <= hi)
        cal.append({
            "bin": [lo, hi], "candidates": int(sel.sum()), "real": int(real_flag[sel].sum()),
            "posterior_sum": float(post[sel].sum()),
            "mean_posterior": float(post[sel].mean()) if sel.any() else None,
            "share_real": float(real_flag[sel].mean()) if sel.any() else None,
        })  # fmt: skip
    days = hours / 24.0
    return {
        "replication": rep,
        "incidents": len(incidents),
        "detectable_incidents": len(detectable),
        "camera_coverage_mass": coverage_mass,
        "incidents_in_camera_view": len(covered_ids),
        "expected_incidents_in_camera_view": len(detectable) * coverage_mass,
        "calls": len(calls),
        "false_calls": sum(c.incident_id is None for c in calls),
        "camera_events": len(events),
        "true_camera_events": sum(e.incident_id is not None for e in events),
        "candidates": len(engine.candidates),
        "association": pair_scores(groups),
        "calibration": cal,
        "real_candidates_by_path": dict(n_real_cands),
        "real_candidates_seeded_by_a_false_alarm": seeded_false,
        "fused_estimate": dict(fused),
        "real_candidates_reaching_threshold": int(sum(reached_incidents.values())),
        "duplicate_dispatch_candidates": int(
            sum(reached_incidents.values()) - len(reached_incidents)
        ),
        "onset_to_threshold_s": {p: sorted(round(x) for x in v) for p, v in reach_by_path.items()},
        "days": days,
        "real_incidents_reaching_threshold": len(reached_incidents),
        "false_candidates_per_day": false_cands / days,
        "false_candidates_reaching_threshold_per_day": false_reach / days,
        "fusion_seconds": elapsed,
    }


def quantiles(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def pool(reps: list[dict]) -> dict:
    """Totals over replications: a single week holds ~0.8 incidents in camera view."""
    total = {
        k: sum(r[k] for r in reps)
        for k in (
            "incidents", "detectable_incidents", "incidents_in_camera_view",
            "expected_incidents_in_camera_view", "calls", "false_calls", "camera_events",
            "true_camera_events", "candidates", "real_candidates_seeded_by_a_false_alarm",
            "real_candidates_reaching_threshold", "duplicate_dispatch_candidates", "days",
        )
    }  # fmt: skip
    total["camera_coverage_mass"] = reps[0]["camera_coverage_mass"]  # a property of the map
    fused = {k: sum(r["fused_estimate"].get(k, 0) for r in reps) for k in FUSED_KEYS}
    paths: dict[str, list[float]] = defaultdict(list)
    by_path = Counter()
    for r in reps:
        for name, values in r["onset_to_threshold_s"].items():
            paths[name] += values
        by_path.update(r["real_candidates_by_path"])
    pairs = {k: sum(r["association"][k] for r in reps)
             for k in ("pairs_together", "pairs_in_candidates", "same_incident_pairs",
                       "false_items_in_real_candidates")}  # fmt: skip
    cal = []
    for j, (lo, hi) in enumerate(itertools.pairwise(CAL_EDGES)):
        n = sum(r["calibration"][j]["candidates"] for r in reps)
        real = sum(r["calibration"][j]["real"] for r in reps)
        psum = sum(r["calibration"][j]["posterior_sum"] for r in reps)
        cal.append({"bin": [lo, hi], "candidates": n, "real": real,
                    "mean_posterior": psum / n if n else None,
                    "share_real": real / n if n else None})  # fmt: skip
    return {
        "replications": len(reps),
        **total,
        "association": {
            "pair_precision": pairs["pairs_together"] / max(pairs["pairs_in_candidates"], 1),
            "pair_recall": pairs["pairs_together"] / max(pairs["same_incident_pairs"], 1),
            **pairs,
        },
        "real_candidates_by_path": dict(by_path),
        "fused_estimate": fused,
        "onset_to_threshold_s": {k: quantiles(v) for k, v in sorted(paths.items())},
        "real_incidents_reaching_threshold": sum(
            r["real_incidents_reaching_threshold"] for r in reps
        ),
        "false_candidates_per_day": sum(r["false_candidates_per_day"] * r["days"] for r in reps)
        / total["days"],
        "false_candidates_reaching_threshold_per_day": sum(
            r["false_candidates_reaching_threshold_per_day"] * r["days"] for r in reps
        )
        / total["days"],
        "calibration": cal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--repool",
        action="store_true",
        help="recompute the pooled summary from the replications already in the output file "
        "(a pure function of them) instead of simulating again",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    seed_base = base.random.tuning_seed_base
    if args.repool:
        reps = json.loads(OUT.read_text(encoding="utf-8"))["replications"]
        log.info("re-pooling %d stored replications; nothing is simulated", len(reps))
    else:
        network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
        reps = []
        for r in range(args.reps):
            reps.append(run(r, args.scenario, network, seed_base))
            log.info("replication %d: %s", r, {k: v for k, v in reps[-1].items() if k in (
                "incidents", "incidents_in_camera_view", "calls", "camera_events",
                "candidates")})  # fmt: skip
    detector = load_detector_model(PROJECT_ROOT / load_fusion_config().detector_inputs)
    masking = detector.masking()
    pooled = pool(reps)
    masking["expected_true_detections"] = (
        pooled["incidents_in_camera_view"] * detector.effective_p_detect
    )
    masking["observed_true_detections"] = pooled["true_camera_events"]
    report = {
        "status": "synthetic model check (UNVERIFIED call and review parameters; measured "
        "detector); not a research result",
        "scenario": args.scenario,
        "seed_base": "tuning_seed_base",
        "threshold": THRESHOLD,
        "posterior_evaluated_at": "the time of each candidate's last evidence (not the end of "
        "the horizon), so the calibration table excludes the decay a candidate suffers while "
        "nothing further confirms it",
        "calibration_label": "a candidate counts as real if it holds any genuine evidence, which "
        "is stricter than the hypothesis H1 (an incident occurred anywhere in its region during "
        "its window)",
        "detector_masking": masking,
        "pooled": pooled,
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
    print(json.dumps({k: v for k, v in report.items() if k != "replications"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
