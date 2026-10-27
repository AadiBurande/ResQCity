"""Run one replication of one policy on one scenario, and write the logs (phase 7 slices 2+6).

    uv run python -m resqcity.sim.runner --scenario normal --policy B0 --reps 1

This is the module that wires the pieces together. Nothing here invents a process: incidents come
from ``sim/generator.py``, camera events from ``sim/detection.py`` (the measured detector),
calls from ``sim/calls.py``, the posterior from ``fusion/engine.py``, the assignment from
``sim/policies.py`` and the clock and fleet from ``sim/engine.py``.

**Evidence is pre-generated, then replayed.** The whole week's incidents, alarms and calls are
drawn before the clock starts, exactly as ``sim/fusion_check.py`` does. That is what makes common
random numbers across policies *exact*: B0 and FULL see the identical week, so a difference
between them is the policy and nothing else. A test asserts it rather than the docstring claiming
it.

**Ground truth stays here.** The engine needs to know whether a dispatched-to candidate is real,
because a false alarm costs a short look rather than a full response. That knowledge lives in
this module and in the engine, never in a policy: policies are handed fused ``Incident`` objects
only, and ``tests/test_architecture.py`` enforces the import boundary.

**The warm-up is simulated and then discarded.** The fleet starts idle at every base, which
flatters the first incidents, so incidents whose onset falls inside ``run.warmup_hours`` are
excluded from the metrics but still consume units.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.requirements import load_requirements_config
from resqcity.common.rng import stream
from resqcity.common.schemas import Agency, Incident, ServiceCounts, SourceKind
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
from resqcity.geo.facilities import load_facilities
from resqcity.geo.network import RoadNetwork
from resqcity.geo.scenario import ScenarioState, load_scenario
from resqcity.optimization.config import load_fleet_config, load_optimization_config
from resqcity.optimization.fleet import build_fleet
from resqcity.optimization.routing import EtaTable
from resqcity.sim.background import build_background
from resqcity.sim.background_load import BackgroundLoad
from resqcity.sim.calls import CallSimulator
from resqcity.sim.config import load_sim_config, load_survival_config
from resqcity.sim.detection import DetectionProcess
from resqcity.sim.engine import Engine
from resqcity.sim.generator import IncidentGenerator, hotspot_weights, load_incidents_config
from resqcity.sim.hospitals import HospitalNetwork
from resqcity.sim.optimize_run import ward_demand
from resqcity.sim.policies import build as build_policy
from resqcity.sim.units import Fleet

OUT = PROJECT_ROOT / "results" / "sim"
#: How often expired candidates are swept out of the fusion engine (as in fusion_check).
EXPIRY_TICK = timedelta(minutes=5)
log = logging.getLogger(__name__)


@dataclass
class IncidentRow:
    """One row of the per-incident log. Column names match configs/sim.yaml -> logs."""

    incident_id: str
    policy: str
    replication: int
    onset_ts: datetime | None = None
    first_detection_ts: datetime | None = None
    first_call_ts: datetime | None = None
    decision_ts: datetime | None = None
    posterior_at_decision: float | None = None
    tier: str | None = None
    type: str | None = None
    severity_class: str | None = None
    ward_id: str | None = None
    node: int | None = None
    required_ambulance: int = 0
    required_police: int = 0
    required_fire: int = 0
    estimated_ambulance: int = 0
    estimated_police: int = 0
    estimated_fire: int = 0
    arrival_ambulance_ts: datetime | None = None
    arrival_police_ts: datetime | None = None
    arrival_fire_ts: datetime | None = None
    last_required_arrival_ts: datetime | None = None
    second_wave_ts: datetime | None = None
    hospital_arrival_ts: datetime | None = None
    handover_ts: datetime | None = None
    shortfall_units: int = 0
    was_camera_covered: bool = False
    is_false_alarm: bool = False
    in_warmup: bool = False


@dataclass
class Replication:
    """Everything one (policy, replication) run produced."""

    policy: str
    replication: int
    rows: list[IncidentRow] = field(default_factory=list)
    background: dict[str, object] = field(default_factory=dict)
    hospitals: dict[str, object] = field(default_factory=dict)
    utilisation: dict[str, float] = field(default_factory=dict)
    evidence_digest: str = ""
    wall_s: float = 0.0
    solver_calls: int = 0
    solver_s: float = 0.0


def _streams(seed_base: int, rep: int) -> "callable":
    """One persistent RNG per stream name (common random numbers, D-030 pattern).

    The cache matters: handing back a fresh generator for the same name would replay its first
    draw for ever, which is the bug slice 4 found in the background workload.
    """
    cache: dict[str, np.random.Generator] = {}

    def rng(name: str) -> np.random.Generator:
        # the whole name is the stream key (crc32 of it), so "on_scene/I1/P1" and
        # "background_service/police" are independent streams without needing numeric paths
        if name not in cache:
            cache[name] = stream(seed_base, rep, name)
        return cache[name]

    return rng


class Week:
    """The pre-generated evidence for one replication: identical across policies."""

    def __init__(self, network: RoadNetwork, scenario: str, seed_base: int, rep: int) -> None:
        self.inc_cfg = load_incidents_config()
        self.req = load_requirements_config()
        self.calls_cfg = load_calls_config()
        self.fusion_cfg = load_fusion_config()
        self.detector = load_detector_model(PROJECT_ROOT / self.fusion_cfg.detector_inputs)
        self.sc = load_scenario(scenario)
        self.state = ScenarioState(self.sc, network)
        self.generator = IncidentGenerator(network, self.state, self.inc_cfg, self.req)
        self.incidents = self.generator.generate(seed_base, rep)
        self.start = self.sc.start
        hours = self.sc.days * 24
        self.cameras = load_cameras(CONFIG_DIR / "cameras.yaml")
        self.coverage = camera_coverage(
            network, self.cameras, self.fusion_cfg.camera.coverage_radius_m.value
        )
        self.background = build_background(
            network, self.inc_cfg, self.calls_cfg, self.sc.incident_rate_multiplier
        )
        caller = CallSimulator(
            network,
            self.calls_cfg,
            load_templates(),
            self.req,
            self.generator.node_share,
            self.background.type_share,
        )
        self.calls = caller.generate(self.incidents, self.start, hours, seed_base, rep)
        self.events = DetectionProcess(
            network, self.cameras, self.detector, self.fusion_cfg, self.req
        ).generate(self.incidents, self.start, hours, seed_base, rep)
        # camera_coverage returns internal node INDICES; incident nodes are OSM node IDS, so
        # they have to be translated or the two sets never intersect (which silently reported
        # "no incident was ever in camera range" while the detector was detecting some).
        covered_indices = {int(v) for nodes in self.coverage.values() for v in nodes.tolist()}
        node_ids = network.nodes["node_id"].to_numpy()
        self.covered_nodes = {int(node_ids[i]) for i in covered_indices}

    def digest(self) -> str:
        """A fingerprint of the week, so two policies can be proved to have seen the same one."""
        import hashlib

        h = hashlib.sha256()
        for inc in self.incidents:
            h.update(f"{inc.id}|{inc.onset_ts.isoformat()}|{inc.node}|".encode())
        for e in self.events:
            h.update(f"{e.event.id}|{e.event.ts.isoformat()}|{e.incident_id}|".encode())
        for c in self.calls:
            h.update(f"{c.call.id}|{c.call.ts.isoformat()}|{c.incident_id}|".encode())
        return h.hexdigest()

    def stream(self) -> list[tuple[datetime, int, str, object]]:
        """Camera events and calls in time order, as the control room would receive them."""
        return sorted(
            [(e.event.ts, 0, e.event.id, e) for e in self.events]
            + [(c.call.ts, 1, c.call.id, c) for c in self.calls]
        )


def run_replication(
    network: RoadNetwork, scenario: str, policy_name: str, rep: int, seed_base: int
) -> Replication:
    started = time.perf_counter()
    sim_cfg, opt_cfg = load_sim_config(), load_optimization_config()
    week = Week(network, scenario, seed_base, rep)
    rng = _streams(seed_base, rep)

    base = load_base_config()
    facilities = load_facilities(base.paths.resolve("data_processed") / "facilities.json")
    demand = ward_demand(network, hotspot_weights(network, week.inc_cfg.hotspot))
    units = build_fleet(facilities.stations, facilities.hospitals, demand, load_fleet_config())

    hospitals = HospitalNetwork(list(facilities.hospitals))
    engine = Engine(
        network,
        week.state,
        Fleet.from_units(units),
        sim_cfg,
        opt_cfg,
        week.start,
        rng,
        hospitals=hospitals,
    )
    load = BackgroundLoad(engine, sim_cfg, opt_cfg)
    load.start()

    def _retry_worth_it() -> bool:
        """Is any short response now fillable?

        Only a REFUSED ask counts. A response can be short because the policy deliberately
        held units back -- which is exactly what a confidence tier does -- and retrying that
        would both fight the policy and, since the answer never changes, rebuild an ETA table
        (a Dijkstra over the Pune graph) every interval for the rest of the week. Watching for
        any unit to free up is no good either: with the background workload running, units free
        constantly. The condition is a refused ask plus a claimable unit of that agency, which
        costs a scan of the fleet rather than a shortest-path search.
        """
        for state in engine.responses.values():
            for agency in state.was_refused():
                if engine.fleet.claimable(agency):
                    return True
        return False

    fusion = FusionEngine(
        network,
        week.cameras,
        week.detector,
        week.calls_cfg,
        week.fusion_cfg,
        week.req,
        week.background,
        week.coverage,
    )
    triager = RuleTriager(load_triage_config(), week.req)
    policy = build_policy(policy_name, sim_cfg, opt_cfg)

    def _counts(outstanding: dict[Agency, int]) -> ServiceCounts:
        return ServiceCounts(**{a.value: outstanding.get(a, 0) for a in Agency})

    def _human_evidence(candidate: Incident) -> bool:
        return any(
            s.kind in (SourceKind.CALL, SourceKind.OPERATOR_REVIEW) for s in candidate.sources
        )

    truth_of: dict[str, str | None] = {}  # fused candidate -> ground-truth id, or None if false
    latest: dict[str, Incident] = {}  # most recent fused view, for re-offering a short response
    rows: dict[str, IncidentRow] = {}
    decided: set[str] = set()
    result = Replication(policy=policy_name, replication=rep, evidence_digest=week.digest())

    def record_evidence(candidate: Incident, truth_id: str | None, kind: int, ts: datetime) -> None:
        if candidate.id not in truth_of or truth_of[candidate.id] is None:
            truth_of[candidate.id] = truth_id
        latest[candidate.id] = candidate
        row = rows.setdefault(candidate.id, IncidentRow(candidate.id, policy_name, rep))
        if kind == 0 and row.first_detection_ts is None:
            row.first_detection_ts = ts
        if kind == 1 and row.first_call_ts is None:
            row.first_call_ts = ts

    def dispatch(now: datetime, pending: dict[str, Incident]) -> None:
        """Ask the policy about the candidates that changed, then execute what it decides.

        Only candidates whose evidence moved since the last batch are offered. Re-offering every
        open candidate on a timer would rebuild an ETA table -- a Dijkstra over the Pune graph --
        on each of the ~20,000 ticks in a week for nothing: a candidate with no new evidence has
        the same posterior and would get the same answer.
        """
        offered = dict(pending)
        # Top up any response that is still short of units. A first batch that could only find
        # a police unit must not leave the incident permanently incomplete: without this the
        # primary outcome (time to the last REQUIRED arrival) would almost never be defined.
        for candidate_id, state in engine.responses.items():
            refused = state.was_refused()
            if not refused:
                continue
            known = latest.get(candidate_id)
            if known is None:
                continue
            # re-offer only what the policy asked for and could not get
            offered[candidate_id] = known.model_copy(update={"requirements": _counts(refused)})
        open_candidates = [
            inc
            for inc in offered.values()
            if sum(getattr(inc.requirements, a.value) for a in Agency) > 0
        ]
        if policy_name == "B0":
            open_candidates = [c for c in open_candidates if _human_evidence(c)]
        if policy_name in {"B0", "B1", "B2"}:
            # the untiered policies will not act below the credibility gate, so do not pay for
            # an ETA table to be told that (B3 and FULL tier instead and must see everything)
            gate = sim_cfg.policies.min_dispatch_posterior.value
            open_candidates = [c for c in open_candidates if c.posterior >= gate]
        if not open_candidates:
            return
        # Only candidates that could actually receive a unit are worth a shortest-path search.
        # A tiered policy still has to SEE the rest -- a camera-only alarm gets a REVIEW
        # decision, and that review workload is itself a result -- so they stay in the batch
        # with unreachable ETAs, which no assignment can come from. At ~193,000 camera alarms a
        # week this is the difference between one Dijkstra per batch and thousands.
        floor = (
            opt_cfg.policy.tau_low.value
            if policy_name in {"B3", "FULL"}
            else sim_cfg.policies.min_dispatch_posterior.value
        )
        actionable = [k for k, c in enumerate(open_candidates) if c.posterior >= floor]
        nodes = [int(open_candidates[k].node) for k in actionable]
        seconds = np.full((len(engine.fleet), len(open_candidates)), np.inf)
        if nodes:
            partial = engine.eta_table(nodes)
            seconds[:, actionable] = partial.seconds
            # Mask the units that are not available. Policies are handed Unit RECORDS, whose
            # state field is the roster's, not the live one, so without this every policy would
            # choose among all 91 units as though none were busy -- naming a committed unit,
            # having the engine refuse it, and being asked again. It would also have quietly
            # undone the background workload: capacity cannot bite if the dispatcher cannot
            # see it.
            busy = [i for i, u in enumerate(engine.fleet.units) if not u.claimable]
            if busy:
                seconds[busy, :] = np.inf
        table = EtaTable(engine.fleet.nodes, [int(c.node) for c in open_candidates], seconds)
        plan = policy.decide(open_candidates, units, table, now)
        result.solver_calls += 1
        result.solver_s += plan.solve_s
        by_id = {d.incident_id: d for d in plan.decisions}
        for candidate in open_candidates:
            decision = by_id.get(candidate.id)
            if decision is None:
                continue
            truth_id = truth_of.get(candidate.id)
            required = {a: int(getattr(candidate.requirements, a.value)) for a in Agency}
            engine.register(
                candidate.id,
                int(candidate.node),
                required,
                is_false_alarm=truth_id is None,
                event_type=candidate.type,
            )
            row = rows.setdefault(candidate.id, IncidentRow(candidate.id, policy_name, rep))
            row.decision_ts = now
            row.posterior_at_decision = candidate.posterior
            row.tier = decision.tier.value
            row.type = candidate.type.value
            row.ward_id = candidate.ward_id
            row.node = int(candidate.node)
            row.is_false_alarm = truth_id is None
            row.was_camera_covered = int(candidate.node) in week.covered_nodes
            row.estimated_ambulance = required[Agency.AMBULANCE]
            row.estimated_police = required[Agency.POLICE]
            row.estimated_fire = required[Agency.FIRE]
            if decision.assignments:
                decided.add(candidate.id)  # kept for the log: this candidate was acted on
        engine.execute(plan.decisions, plan.delays)

    # ---- the clock -------------------------------------------------------------------------
    # Evidence is replayed in time order; the clock advances to each item, and a dispatch batch
    # is formed from the accumulated evidence at most every dispatch_interval_s.
    horizon_s = min(sim_cfg.run.horizon_s, week.sc.days * 86400.0)
    next_expiry = week.start + EXPIRY_TICK
    pending: dict[str, Incident] = {}
    last_dispatch_s = -sim_cfg.run.dispatch_interval_s
    for ts, kind, _eid, item in week.stream():
        at_s = (ts - week.start).total_seconds()
        if at_s > horizon_s:
            break
        engine.env.run(until=max(at_s, engine.now_s))
        while ts >= next_expiry:
            fusion.tick(next_expiry, refresh=False)
            next_expiry += EXPIRY_TICK
        produced = (
            fusion.on_event(item.event)
            if kind == 0
            else fusion.on_call(item.call, triager.triage(item.call))
        )
        for candidate in produced:
            record_evidence(candidate, item.incident_id, kind, ts)
            pending[candidate.id] = candidate
        due = engine.now_s - last_dispatch_s >= sim_cfg.run.dispatch_interval_s
        if due and (pending or _retry_worth_it()):
            dispatch(engine.now, pending)
            pending.clear()
            last_dispatch_s = engine.now_s
    if pending:
        dispatch(engine.now, pending)
    engine.run(until_s=horizon_s)

    # ---- outcomes --------------------------------------------------------------------------
    onset_of = {i.id: i.onset_ts for i in week.incidents}
    truth_required = {
        i.id: {a: int(getattr(i.requirements, a.value)) for a in Agency} for i in week.incidents
    }
    for candidate_id, row in rows.items():
        truth_id = truth_of.get(candidate_id)
        if truth_id is not None and truth_id in onset_of:
            row.onset_ts = onset_of[truth_id]
            need = truth_required[truth_id]
            row.required_ambulance = need[Agency.AMBULANCE]
            row.required_police = need[Agency.POLICE]
            row.required_fire = need[Agency.FIRE]
        state = engine.responses.get(candidate_id)
        if state is not None:
            first: dict[Agency, float] = {}
            for arrival in state.arrivals:
                first.setdefault(arrival.agency, arrival.arrived_s)
            for agency, seconds in first.items():
                setattr(
                    row,
                    f"arrival_{agency.value}_ts",
                    week.start + timedelta(seconds=seconds),
                )
            last = state.last_required_arrival_s()
            if last is not None:
                row.last_required_arrival_ts = week.start + timedelta(seconds=last)
            row.shortfall_units = sum(
                max(0, n - state.arrived_counts()[a]) for a, n in state.required.items()
            )
            if state.hospital_arrival_s is not None:
                row.hospital_arrival_ts = week.start + timedelta(seconds=state.hospital_arrival_s)
            if state.handover_s is not None:
                row.handover_ts = week.start + timedelta(seconds=state.handover_s)
        anchor = row.onset_ts or row.first_detection_ts or row.first_call_ts
        if anchor is not None:
            row.in_warmup = (anchor - week.start).total_seconds() < sim_cfg.run.warmup_s
    result.rows = list(rows.values())
    result.background = load.report()
    result.hospitals = hospitals.report()
    result.utilisation = {
        a.value: engine.fleet.utilisation(a, sim_cfg.run.warmup_s, horizon_s) for a in Agency
    }
    result.wall_s = time.perf_counter() - started
    return result


def to_frame(reps: list[Replication]) -> pd.DataFrame:
    rows = [{**row.__dict__} for rep in reps for row in rep.rows]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--policy", default="B0")
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    sim_cfg = load_sim_config()
    survival = load_survival_config()
    seed_base = base.random.seed_base
    reps = [
        run_replication(network, args.scenario, args.policy, rep, seed_base)
        for rep in range(args.reps)
    ]
    frame = to_frame(reps)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_{args.policy}_synthetic"
    if sim_cfg.logs.write_parquet:
        frame.to_parquet(OUT / f"incidents_{stem}.parquet", index=False)
    summary = {
        "status": "SYNTHETIC. Primary outcome is "
        f"{survival.primary_outcome}; expected survival is secondary and its curves are "
        "ILLUSTRATIVE (configs/survival.yaml).",
        "scenario": args.scenario,
        "policy": args.policy,
        "replications": args.reps,
        "seed_base": seed_base,
        "warmup_hours": sim_cfg.run.warmup_hours,
        "rows": len(frame),
        "evidence_digests": [r.evidence_digest for r in reps],
        "wall_seconds": [r.wall_s for r in reps],
        "solver_calls": [r.solver_calls for r in reps],
        "utilisation_after_warmup": [r.utilisation for r in reps],
        "background": [r.background for r in reps],
        "hospitals": [r.hospitals for r in reps],
    }
    (OUT / f"summary_{stem}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    for path in (OUT / f"summary_{stem}.json",):
        write_meta(
            path,
            configs=[CONFIG_DIR / "sim.yaml", CONFIG_DIR / "survival.yaml"],
            synthetic=True,
            seeds={"seed_base": seed_base},
        )
    log.info(
        "%s %s: %d rows, %.1f s/rep, utilisation %s",
        args.scenario,
        args.policy,
        len(frame),
        sum(r.wall_s for r in reps) / max(1, len(reps)),
        {k: round(v, 4) for k, v in reps[0].utilisation.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
