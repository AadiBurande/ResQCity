"""The five dispatch policies compared by the study (phase 7 slice 3, D-032).

| policy | evidence | who is sent | extras |
|---|---|---|---|
| ``B0`` | calls only | nearest unit per agency, **agency by agency** | sequential hand-off delay |
| ``B1`` | calls + CCTV | nearest unit per agency, all at once | - |
| ``B2`` | calls + CCTV | joint MILP (C1) | - |
| ``B3`` | calls + CCTV | joint MILP | confidence tiers (C2) |
| ``FULL`` | calls + CCTV | joint MILP | tiers + green corridor + pre-positioning (C3) |

The ladder is designed so that each step isolates one mechanism: B0 to B1 is *evidence*, B1 to B2
is *joint assignment*, B2 to B3 is *confidence*, B3 to FULL is *anticipation*. A policy that wins
should be attributable to the step it won at.

**What a policy may see.** Fused ``Incident`` objects, the ``Unit`` records and an ETA table.
Never a ``GroundTruthIncident``: ``tests/test_architecture.py`` enforces that only ``sim`` reads
ground truth, and the engine hands policies nothing else. In particular the requirement a policy
acts on is the ESTIMATE from camera cues or call triage, which can be wrong -- without that,
CCTV-led policies would be flattered (master plan section 6 item 7).

**B0's hand-off is the honest weak point.** It is the delay joint dispatch removes, so it drives
RQ2 and it is UNVERIFIED. See ``configs/sim.yaml`` -> ``policies.b0``; phase 10 sweeps it
including zero.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DispatchDecision,
    DispatchTier,
    Explanation,
    Factor,
    Incident,
    SourceKind,
    Unit,
)
from resqcity.optimization import policy as c2
from resqcity.optimization.config import OptimizationConfig
from resqcity.optimization.dispatch import DispatchIncident
from resqcity.optimization.dispatch import solve as c1_solve
from resqcity.optimization.routing import EtaTable, green_corridor_etas, nearest_unit
from resqcity.sim.config import SimConfig

log = logging.getLogger(__name__)

#: Policies that are allowed to act on camera evidence at all.
SEES_CAMERAS = frozenset({"B1", "B2", "B3", "FULL"})
#: Policies that solve the joint MILP rather than picking the nearest unit per agency.
USES_JOINT_MILP = frozenset({"B2", "B3", "FULL"})
#: Policies that apply confidence tiers.
USES_TIERS = frozenset({"B3", "FULL"})
#: Policies that get the green corridor and pre-positioning.
USES_ANTICIPATION = frozenset({"FULL"})


@dataclass
class DispatchPlan:
    """What a policy decided, plus the per-agency release delays the engine must honour."""

    decisions: list[DispatchDecision]
    explanations: list[Explanation]
    #: (incident_id, unit_id) -> seconds to wait before this unit starts moving. Only B0 uses it.
    delays: dict[tuple[str, str], float]
    status: str = ""
    solve_s: float = 0.0


class Policy(Protocol):
    name: str

    def decide(
        self,
        incidents: Sequence[Incident],
        units: Sequence[Unit],
        etas: EtaTable,
        now: datetime,
    ) -> DispatchPlan: ...


def passes_gate(incident: Incident, cfg: SimConfig) -> bool:
    """Whether an UNTIERED policy (B0, B1, B2) will act on this candidate at all.

    Not a confidence tier: one fixed threshold, because a policy with no credibility gate would
    send a unit to each of the ~195,000 camera alarms a simulated week produces at the measured
    false-alarm rate (D-029). See ``configs/sim.yaml`` -> ``policies.min_dispatch_posterior``.
    B3 and FULL ignore this and use C2's tiers instead.
    """
    return incident.posterior >= cfg.policies.min_dispatch_posterior.value


def has_call_evidence(incident: Incident) -> bool:
    """Whether a human ever reported this incident.

    B0 is the call-based status quo, so it must not react to an incident whose only evidence is a
    camera alarm. Operator reviews count as human evidence: a review only happens because a
    control room looked, which is a call-era capability too.
    """
    return any(s.kind in (SourceKind.CALL, SourceKind.OPERATOR_REVIEW) for s in incident.sources)


def _to_dispatch_incident(incident: Incident, severity_class: str) -> DispatchIncident:
    return DispatchIncident(
        id=incident.id,
        node=int(incident.node),
        requirements=incident.requirements,
        severity_class=severity_class,
        ward_id=incident.ward_id,
        posterior=incident.posterior,
    )


def severity_class_of(incident: Incident, cfg: OptimizationConfig) -> str:
    """Map the fused severity score onto the dispatch model's weight classes.

    The thresholds are the optimisation config's own severity weights, so the policy cannot
    invent a class the objective does not price.
    """
    classes = sorted(cfg.dispatch.severity_weights.value)
    if not classes:  # pragma: no cover - config guarantees at least one
        return "serious"
    # severity is in [0, 1]; split it evenly across the declared classes, lowest weight first
    ordered = sorted(classes, key=lambda c: cfg.dispatch.severity_weights.value[c])
    index = min(int(incident.severity * len(ordered)), len(ordered) - 1)
    return ordered[index]


class NearestPerAgency:
    """B0 and B1: each agency independently sends its nearest reachable unit.

    This is the status quo rule, and it is the baseline the whole study is measured against, so
    it is deliberately simple: no look-ahead, no coordination, no sharing of information between
    agencies. The only difference between B0 and B1 is the evidence they accept and whether the
    agencies are asked one after another.
    """

    def __init__(self, name: str, cfg: SimConfig, opt: OptimizationConfig) -> None:
        if name not in {"B0", "B1"}:
            raise ValueError(f"NearestPerAgency implements B0 and B1, not {name}")
        self.name, self.cfg, self.opt = name, cfg, opt

    @property
    def handoff_s(self) -> float:
        return self.cfg.policies.b0.sequential_handoff_s.value if self.name == "B0" else 0.0

    def decide(
        self,
        incidents: Sequence[Incident],
        units: Sequence[Unit],
        etas: EtaTable,
        now: datetime,
    ) -> DispatchPlan:
        agencies = [u.agency for u in units]
        taken: set[int] = set()
        decisions, explanations, delays = [], [], {}
        for index, incident in enumerate(incidents):
            if self.name == "B0" and not has_call_evidence(incident):
                continue  # the call-based status quo never sees a camera-only candidate
            if not passes_gate(incident, self.cfg):
                continue
            free = EtaTable(etas.unit_nodes, etas.incident_nodes, etas.seconds.copy())
            for u in taken:
                free.seconds[u, :] = np.inf
            assignments: list[Assignment] = []
            # ask the agencies in a fixed order, so the hand-off penalty is reproducible
            step = 0
            for agency in Agency:
                need = int(getattr(incident.requirements, agency.value))
                for _ in range(need):
                    chosen = nearest_unit(free, agencies, agency, index)
                    if chosen is None:
                        break
                    free.seconds[chosen, :] = np.inf
                    taken.add(chosen)
                    delay = self.handoff_s * step
                    assignments.append(
                        Assignment(
                            unit_id=units[chosen].id,
                            eta_s=float(etas.seconds[chosen, index]) + delay,
                            role=AssignmentRole.RESPOND,
                        )
                    )
                    delays[(incident.id, units[chosen].id)] = delay
                if need > 0:
                    step += 1
            if not assignments:
                continue
            decision = DispatchDecision(
                id=f"DEC-{incident.id}",
                incident_id=incident.id,
                created_ts=now,
                policy=self.name,
                assignments=assignments,
                tier=DispatchTier.FULL,
                expected_last_arrival_s=max(a.eta_s for a in assignments),
                objective_breakdown={
                    "posterior": incident.posterior,
                    "sequential_handoff_s": self.handoff_s,
                },
            )
            decisions.append(decision)
            explanations.append(
                Explanation(
                    decision_id=decision.id,
                    text=(
                        f"{incident.id}: {self.name} sends each agency's nearest unit"
                        + (
                            f", with a {self.handoff_s:.0f} s hand-off before each further agency"
                            if self.handoff_s
                            else ", all at once"
                        )
                    ),
                    factors=[Factor(name="posterior", value=incident.posterior, weight=1.0)],
                )
            )
        return DispatchPlan(decisions, explanations, delays, status="NEAREST")


class JointMilp:
    """B2, B3 and FULL: the C1 solver, with confidence tiers and the corridor layered on.

    Each extra mechanism is switched on by membership of the sets at the top of this module, so
    the ladder is visible in one place rather than spread across subclasses.
    """

    def __init__(self, name: str, cfg: SimConfig, opt: OptimizationConfig) -> None:
        if name not in USES_JOINT_MILP:
            raise ValueError(f"JointMilp implements {sorted(USES_JOINT_MILP)}, not {name}")
        self.name, self.cfg, self.opt = name, cfg, opt

    @property
    def uses_tiers(self) -> bool:
        return self.name in USES_TIERS

    @property
    def uses_corridor(self) -> bool:
        return self.name in USES_ANTICIPATION

    def decide(
        self,
        incidents: Sequence[Incident],
        units: Sequence[Unit],
        etas: EtaTable,
        now: datetime,
    ) -> DispatchPlan:
        if self.uses_tiers:
            batch_incidents = list(incidents)  # C2's tiers decide; no fixed gate
        else:
            keep = [i for i, inc in enumerate(incidents) if passes_gate(inc, self.cfg)]
            batch_incidents = [incidents[i] for i in keep]
            etas = EtaTable(
                etas.unit_nodes,
                [etas.incident_nodes[i] for i in keep],
                etas.seconds[:, keep],
            )
        incidents = batch_incidents
        if not incidents:
            return DispatchPlan([], [], {}, status="EMPTY")
        agencies = [u.agency for u in units]
        batch = [_to_dispatch_incident(i, severity_class_of(i, self.opt)) for i in incidents]
        assisted = green_corridor_etas(self.opt, etas, agencies) if self.uses_corridor else None
        if self.uses_tiers:
            result = c2.decide(
                batch, list(units), etas, self.opt, now, self.name, assisted_etas=assisted
            )
        else:
            result = c1_solve(
                batch, list(units), etas, self.opt, now, self.name, assisted_etas=assisted
            )
        return DispatchPlan(
            decisions=list(result.decision),
            explanations=list(result.explanation),
            delays={},  # joint dispatch is the point: nobody waits for a hand-off
            status=result.status,
            solve_s=result.solve_s,
        )


def build(name: str, cfg: SimConfig, opt: OptimizationConfig) -> Policy:
    """The one place a policy name becomes an object."""
    if name in {"B0", "B1"}:
        return NearestPerAgency(name, cfg, opt)
    if name in USES_JOINT_MILP:
        return JointMilp(name, cfg, opt)
    raise ValueError(f"unknown policy {name!r}")
