"""C2 — confidence-aware dispatch policy (phase 6.3).

The fused posterior from phase 4 decides how much of the city's capacity an incident is allowed
to consume before anyone has confirmed it exists:

| tier | condition | action |
|---|---|---|
| ``FULL`` | `p >= tau_high` | dispatch every required unit (the C1 solver decides which) |
| ``VERIFY`` | `tau_low <= p < tau_high` | nearest police unit looks; the rest are pre-alerted |
| ``REVIEW`` | `p < tau_low` | nobody moves; an operator reviews, with a timeout |

**What the measurements say this is worth.** A camera alarm on its own multiplies the odds by
1.01-1.04 (D-029), so a camera-only candidate sits at a posterior of order 1e-5, far below any
sensible `tau_low`, and lands in REVIEW. Confidence tiering therefore does **not** turn CCTV
into a dispatch trigger. Its value, if it has one, is that a camera alarm already sitting in
REVIEW is upgraded the moment a call or a human review corroborates it, which is earlier than the
call alone would have arrived. Phase 10 measures whether that is worth anything.

On timeout an unreviewed candidate escalates to VERIFY, not to a full dispatch: something nobody
has looked at should cost one police unit, not the whole requirement.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DispatchDecision,
    DispatchTier,
    Explanation,
    Factor,
)
from resqcity.optimization.config import OptimizationConfig
from resqcity.optimization.dispatch import DispatchIncident, DispatchResult, solve
from resqcity.optimization.routing import EtaTable, nearest_unit


@dataclass(frozen=True)
class TierDecision:
    incident_id: str
    tier: DispatchTier
    posterior: float
    review_deadline: datetime | None = None


def tier_for(posterior: float, cfg: OptimizationConfig) -> DispatchTier:
    """Which tier a posterior falls in. Monotone by construction: a higher posterior never
    yields a weaker response."""
    p = cfg.policy
    if posterior >= p.tau_high.value:
        return DispatchTier.FULL
    if posterior >= p.tau_low.value:
        return DispatchTier.VERIFY
    return DispatchTier.REVIEW


def classify(
    incidents: list[DispatchIncident], cfg: OptimizationConfig, now: datetime
) -> list[TierDecision]:
    out = []
    for inc in incidents:
        tier = tier_for(inc.posterior, cfg)
        deadline = (
            now + timedelta(seconds=cfg.policy.t_review_s.value)
            if tier == DispatchTier.REVIEW
            else None
        )
        out.append(TierDecision(inc.id, tier, inc.posterior, deadline))
    return out


def escalate(tier: DispatchTier, cfg: OptimizationConfig) -> DispatchTier:
    """What a REVIEW candidate becomes when its timeout expires with no verdict."""
    if tier != DispatchTier.REVIEW or not cfg.policy.escalate_on_timeout:
        return tier
    return DispatchTier.VERIFY


def verify_decision(
    inc: DispatchIncident,
    units: list,
    etas: EtaTable,
    index: int,
    cfg: OptimizationConfig,
    now: datetime,
    policy: str,
    taken: set[int] | None = None,
) -> tuple[DispatchDecision, Explanation]:
    """Send the nearest police unit to look; everything else the incident would need is
    pre-alerted so it can start moving the moment the look confirms it."""
    agencies = [u.agency for u in units]
    # units already committed by the C1 solve are unavailable: one unit, one incident (C1)
    free = EtaTable(etas.unit_nodes, etas.incident_nodes, etas.seconds.copy())
    for u in taken or set():
        free.seconds[u, :] = np.inf
    etas = free
    chosen = nearest_unit(etas, agencies, Agency.POLICE, index)
    assignments = []
    if chosen is not None:
        assignments.append(
            Assignment(
                unit_id=units[chosen].id,
                eta_s=float(etas.seconds[chosen, index]),
                role=AssignmentRole.VERIFY,
            )
        )
    for agency in Agency:
        if agency == Agency.POLICE or getattr(inc.requirements, agency.value) <= 0:
            continue
        nearest = nearest_unit(etas, agencies, agency, index)
        if nearest is not None:
            assignments.append(
                Assignment(
                    unit_id=units[nearest].id,
                    eta_s=float(etas.seconds[nearest, index]),
                    role=AssignmentRole.PRE_ALERT,
                )
            )
    verified = [a for a in assignments if a.role == AssignmentRole.VERIFY]
    decision = DispatchDecision(
        id=f"DEC-{inc.id}",
        incident_id=inc.id,
        created_ts=now,
        policy=policy,
        assignments=assignments,
        tier=DispatchTier.VERIFY,
        expected_last_arrival_s=float(verified[0].eta_s) if verified else None,
        objective_breakdown={
            "posterior": inc.posterior,
            "pre_alerted": float(len(assignments) - len(verified)),
        },
    )
    text = (
        f"{inc.id}: posterior {inc.posterior:.2f} is below the full-dispatch threshold, so the "
        f"nearest police unit verifies"
        + (f" in {verified[0].eta_s / 60:.1f} min" if verified else " (none reachable)")
        + (
            f"; {len(assignments) - len(verified)} unit(s) pre-alerted"
            if len(assignments) > len(verified)
            else ""
        )
    )
    factors = [
        Factor(name="posterior", value=inc.posterior, weight=1.0),
        Factor(name="tau_high", value=cfg.policy.tau_high.value, weight=-1.0),
    ]
    return decision, Explanation(decision_id=decision.id, text=text, factors=factors)


def review_decision(
    inc: DispatchIncident, cfg: OptimizationConfig, now: datetime, policy: str
) -> tuple[DispatchDecision, Explanation]:
    """Nobody moves. The decision records the deadline so the simulator can escalate on time."""
    decision = DispatchDecision(
        id=f"DEC-{inc.id}",
        incident_id=inc.id,
        created_ts=now,
        policy=policy,
        assignments=[],
        tier=DispatchTier.REVIEW,
        expected_last_arrival_s=None,
        objective_breakdown={
            "posterior": inc.posterior,
            "review_timeout_s": cfg.policy.t_review_s.value,
        },
    )
    text = (
        f"{inc.id}: posterior {inc.posterior:.4f} is below tau_low "
        f"({cfg.policy.tau_low.value}); held for operator review, escalating after "
        f"{cfg.policy.t_review_s.value / 60:.0f} min"
    )
    return decision, Explanation(
        decision_id=decision.id,
        text=text,
        factors=[Factor(name="posterior", value=inc.posterior, weight=1.0)],
    )


def decide(
    incidents: list[DispatchIncident],
    units: list,
    etas: EtaTable,
    cfg: OptimizationConfig,
    now: datetime,
    policy: str = "C2",
    assisted_etas: np.ndarray | None = None,
    ward_units: dict[str, list[int]] | None = None,
    ward_value: dict[str, float] | None = None,
) -> DispatchResult:
    """Tier every incident, then solve C1 for the FULL ones only.

    VERIFY and REVIEW incidents are deliberately kept out of the MILP: they must not compete for
    units with incidents that are known to be real, and holding them back is the whole point of
    confidence tiering.
    """
    tiers = {d.incident_id: d.tier for d in classify(incidents, cfg, now)}
    full = [i for i in incidents if tiers[i.id] == DispatchTier.FULL]
    decisions, explanations = [], []
    taken: set[int] = set()
    result = DispatchResult([], [], "NO_FULL_TIER", 0.0, 0.0)
    if full:
        keep = [k for k, inc in enumerate(incidents) if tiers[inc.id] == DispatchTier.FULL]
        sub = EtaTable(
            etas.unit_nodes, [etas.incident_nodes[k] for k in keep], etas.seconds[:, keep]
        )
        sub_assisted = None if assisted_etas is None else assisted_etas[:, keep]
        result = solve(full, units, sub, cfg, now, policy, sub_assisted, ward_units, ward_value)
        decisions, explanations = list(result.decision), list(result.explanation)
        by_unit = {u.id: k for k, u in enumerate(units)}
        taken = {
            by_unit[a.unit_id]
            for d in result.decision
            for a in d.assignments
            if a.unit_id in by_unit
        }
    for index, inc in enumerate(incidents):
        if tiers[inc.id] == DispatchTier.FULL:
            continue
        if tiers[inc.id] == DispatchTier.VERIFY:
            decision, why = verify_decision(inc, units, etas, index, cfg, now, policy, taken)
            taken |= {
                k for k, u in enumerate(units) if u.id in {a.unit_id for a in decision.assignments}
            }
        else:
            decision, why = review_decision(inc, cfg, now, policy)
        decisions.append(decision)
        explanations.append(why)
    order = {inc.id: k for k, inc in enumerate(incidents)}
    decisions.sort(key=lambda d: order[d.incident_id])
    explanations.sort(key=lambda e: order[e.decision_id.removeprefix("DEC-")])
    return DispatchResult(
        decision=decisions,
        explanation=explanations,
        status=result.status,
        objective_s=result.objective_s,
        solve_s=result.solve_s,
        last_arrival_s=result.last_arrival_s,
        shortfall=result.shortfall,
        uncovered_wards=result.uncovered_wards,
    )
