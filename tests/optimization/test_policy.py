"""C2 confidence-aware policy (phase 6.3)."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from resqcity.common.schemas import (
    Agency,
    AssignmentRole,
    DispatchTier,
    ServiceCounts,
    Unit,
    UnitState,
)
from resqcity.optimization.config import OptimizationConfig, load_optimization_config
from resqcity.optimization.dispatch import DispatchIncident
from resqcity.optimization.policy import classify, decide, escalate, tier_for
from resqcity.optimization.routing import EtaTable

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 3, 3, 9, 0, tzinfo=IST)
CFG = load_optimization_config()


def _cfg(**policy: object) -> OptimizationConfig:
    data = CFG.model_dump(mode="json")
    data["routing"]["green_corridor"]["rho"]["value"] = 1.0
    for key, value in policy.items():
        if key in {"escalate_on_timeout"}:
            data["policy"][key] = value
        else:
            data["policy"][key]["value"] = value
    return OptimizationConfig.model_validate(data)


def _unit(uid: str, agency: Agency) -> Unit:
    return Unit(id=uid, agency=agency, home_station=f"S-{uid}", node=1,
                state=UnitState.IDLE, eta_fn_ref="default")  # fmt: skip


def _incident(iid: str, posterior: float, amb: int = 1, pol: int = 1) -> DispatchIncident:
    return DispatchIncident(
        id=iid, node=10, requirements=ServiceCounts(ambulance=amb, police=pol, fire=0),
        severity_class="serious", ward_id="W01", posterior=posterior,
    )  # fmt: skip


def _table(matrix: np.ndarray) -> EtaTable:
    return EtaTable(list(range(matrix.shape[0])), list(range(matrix.shape[1])), matrix)


def test_tiers_follow_the_thresholds():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    assert tier_for(0.95, cfg) is DispatchTier.FULL
    assert tier_for(0.7, cfg) is DispatchTier.FULL  # boundary is inclusive
    assert tier_for(0.5, cfg) is DispatchTier.VERIFY
    assert tier_for(0.2, cfg) is DispatchTier.VERIFY
    assert tier_for(0.19, cfg) is DispatchTier.REVIEW
    assert tier_for(0.0, cfg) is DispatchTier.REVIEW


def test_a_higher_posterior_never_yields_a_weaker_response():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    strength = {DispatchTier.REVIEW: 0, DispatchTier.VERIFY: 1, DispatchTier.FULL: 2}
    values = np.linspace(0, 1, 101)
    levels = [strength[tier_for(float(p), cfg)] for p in values]
    assert levels == sorted(levels)


def test_a_camera_only_posterior_lands_in_review():
    """D-029: one camera alarm multiplies the odds by ~1.02, so its posterior is ~1e-5."""
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    assert tier_for(1e-5, cfg) is DispatchTier.REVIEW


def test_review_records_a_deadline_and_escalates_to_verify():
    cfg = _cfg(tau_low=0.2, tau_high=0.7, t_review_s=120.0)
    decisions = classify([_incident("I1", 0.05)], cfg, NOW)
    assert decisions[0].tier is DispatchTier.REVIEW
    assert decisions[0].review_deadline == NOW + timedelta(seconds=120)
    assert escalate(DispatchTier.REVIEW, cfg) is DispatchTier.VERIFY
    assert escalate(DispatchTier.FULL, cfg) is DispatchTier.FULL
    off = _cfg(escalate_on_timeout=False)
    assert escalate(DispatchTier.REVIEW, off) is DispatchTier.REVIEW


def test_verify_sends_one_police_unit_and_pre_alerts_the_rest():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    etas = _table(np.array([[600.0], [300.0], [400.0]]))
    result = decide([_incident("I1", 0.5)], units, etas, cfg, NOW)
    decision = result.decision[0]
    assert decision.tier is DispatchTier.VERIFY
    roles = {a.unit_id: a.role for a in decision.assignments}
    assert roles == {"P2": AssignmentRole.VERIFY, "A1": AssignmentRole.PRE_ALERT}  # nearest police
    assert decision.expected_last_arrival_s == pytest.approx(300.0)


def test_review_moves_nobody():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    result = decide([_incident("I1", 0.01)], units, _table(np.array([[300.0], [400.0]])), cfg, NOW)
    assert result.decision[0].tier is DispatchTier.REVIEW
    assert result.decision[0].assignments == []
    assert "operator review" in result.explanation[0].text


def test_unconfirmed_incidents_do_not_compete_for_units():
    """The point of tiering: a REVIEW candidate must not take the unit a real incident needs."""
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    etas = _table(np.array([[100.0, 100.0], [100.0, 100.0]]))
    real, noise = _incident("REAL", 0.99), _incident("NOISE", 0.01)
    result = decide([real, noise], units, etas, cfg, NOW)
    by_id = {d.incident_id: d for d in result.decision}
    assert by_id["REAL"].tier is DispatchTier.FULL
    assert {a.unit_id for a in by_id["REAL"].assignments} == {"P1", "A1"}
    assert by_id["NOISE"].assignments == []


def test_full_tier_uses_the_joint_solver_and_keeps_input_order():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    etas = _table(np.array([[200.0, 500.0], [300.0, 600.0]]))
    incidents = [_incident("I1", 0.05), _incident("I2", 0.9)]
    result = decide(incidents, units, etas, cfg, NOW)
    assert [d.incident_id for d in result.decision] == ["I1", "I2"]
    assert [e.decision_id for e in result.explanation] == ["DEC-I1", "DEC-I2"]
    assert result.decision[1].tier is DispatchTier.FULL
    assert result.last_arrival_s["I2"] == pytest.approx(600.0)


def test_verify_handles_having_no_police_at_all():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("A1", Agency.AMBULANCE)]
    result = decide([_incident("I1", 0.5)], units, _table(np.array([[400.0]])), cfg, NOW)
    decision = result.decision[0]
    assert decision.tier is DispatchTier.VERIFY
    assert decision.expected_last_arrival_s is None
    assert all(a.role == AssignmentRole.PRE_ALERT for a in decision.assignments)


def test_thresholds_are_flagged_unverified():
    from resqcity.common.config import find_unverified

    flagged = set(find_unverified(CFG))
    assert {"policy.tau_low", "policy.tau_high", "policy.t_review_s"} <= flagged


def test_a_unit_committed_to_a_full_incident_cannot_also_verify_another():
    """(C1) across tiers: the OR review found the same unit dispatched twice, because the
    VERIFY path picked from the whole fleet including units the C1 solve had just taken."""
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    etas = _table(np.array([[100.0, 100.0], [100.0, 100.0]]))
    result = decide([_incident("I_FULL", 0.9), _incident("I_VER", 0.5)], units, etas, cfg, NOW)
    by_id = {d.incident_id: d for d in result.decision}
    assert {a.unit_id for a in by_id["I_FULL"].assignments} == {"P1", "A1"}
    assert by_id["I_VER"].assignments == []  # nothing left to send
    every = [a.unit_id for d in result.decision for a in d.assignments]
    assert len(every) == len(set(every))


def test_a_spare_unit_still_verifies_when_one_is_free():
    cfg = _cfg(tau_low=0.2, tau_high=0.7)
    units = [_unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE), _unit("A1", Agency.AMBULANCE)]
    etas = _table(np.array([[100.0, 500.0], [400.0, 200.0], [100.0, 100.0]]))
    result = decide([_incident("I_FULL", 0.9), _incident("I_VER", 0.5)], units, etas, cfg, NOW)
    by_id = {d.incident_id: d for d in result.decision}
    verify = {a.unit_id for a in by_id["I_VER"].assignments if a.role == AssignmentRole.VERIFY}
    assert verify and not verify & {a.unit_id for a in by_id["I_FULL"].assignments}
