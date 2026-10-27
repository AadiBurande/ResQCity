"""The five dispatch policies (phase 7 slice 3).

The ladder only means something if each rung isolates one mechanism, so most of these tests are
differential: B0 vs B1 for evidence, B1 vs B2 for joint assignment, B2 vs B3 for confidence,
B3 vs FULL for anticipation. A policy that beat another for a reason outside its own rung would
make the phase-10 attribution wrong, which is what these pin down.
"""

from datetime import datetime

import numpy as np
import pytest

from resqcity.common.schemas import (
    Agency,
    AssignmentRole,
    DispatchTier,
    GeoPoint,
    Incident,
    ServiceCounts,
    SourceKind,
    SourceRef,
    Unit,
    UnitState,
)
from resqcity.optimization.config import OptimizationConfig, load_optimization_config
from resqcity.optimization.dispatch import DispatchIncident
from resqcity.optimization.dispatch import solve as c1_solve
from resqcity.optimization.routing import EtaTable
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.policies import (
    SEES_CAMERAS,
    USES_ANTICIPATION,
    USES_JOINT_MILP,
    USES_TIERS,
    build,
    has_call_evidence,
    severity_class_of,
)
from tests.conftest import IST

NOW = datetime(2026, 3, 3, 9, 0, tzinfo=IST)
CFG = load_sim_config()
OPT = load_optimization_config()


def _incident(
    iid: str,
    *,
    amb: int = 0,
    pol: int = 0,
    fire: int = 0,
    posterior: float = 0.95,
    severity: float = 0.5,
    kinds: tuple[SourceKind, ...] = (SourceKind.CALL,),
    node: int = 10,
) -> Incident:
    return Incident(
        id=iid,
        type="accident",
        location=GeoPoint(lat=18.52, lon=73.85),
        node=node,
        ward_id="W01",
        created_ts=NOW,
        updated_ts=NOW,
        sources=[SourceRef(kind=k, ref_id=f"{iid}-{i}") for i, k in enumerate(kinds)],
        posterior=posterior,
        severity=severity,
        requirements=ServiceCounts(ambulance=amb, police=pol, fire=fire),
    )


def _unit(uid: str, agency: Agency, node: int = 1) -> Unit:
    return Unit(
        id=uid, agency=agency, home_station=f"S-{uid}", node=node,
        state=UnitState.IDLE, eta_fn_ref="default",
    )  # fmt: skip


def _table(matrix: np.ndarray) -> EtaTable:
    return EtaTable(list(range(matrix.shape[0])), list(range(matrix.shape[1])), matrix)


def _cfg_handoff(seconds: float) -> SimConfig:
    raw = CFG.model_dump(mode="json")
    raw["policies"]["b0"]["sequential_handoff_s"]["value"] = seconds
    return SimConfig.model_validate(raw)


def _opt_rho(value: float) -> OptimizationConfig:
    raw = OPT.model_dump(mode="json")
    raw["routing"]["green_corridor"]["rho"]["value"] = value
    return OptimizationConfig.model_validate(raw)


# -- the ladder is wired the way the docs say -------------------------------------------------


def test_the_ladder_adds_one_mechanism_at_a_time() -> None:
    assert "B0" not in SEES_CAMERAS and "B1" in SEES_CAMERAS
    assert "B1" not in USES_JOINT_MILP and {"B2", "B3", "FULL"} == USES_JOINT_MILP
    assert {"B3", "FULL"} == USES_TIERS
    assert {"FULL"} == USES_ANTICIPATION


def test_an_unknown_policy_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        build("B9", CFG, OPT)


# -- B0 vs B1: evidence -----------------------------------------------------------------------


def test_b0_ignores_a_camera_only_incident_and_b1_acts_on_it() -> None:
    """The status quo is call-based, so a camera-only candidate must not move a unit under B0.

    This is the one difference between B0 and B1, so if it did not hold the evidence rung of the
    ladder would be measuring nothing.
    """
    camera_only = _incident("C1", pol=1, kinds=(SourceKind.CAMERA_EVENT,))
    units = [_unit("P1", Agency.POLICE)]
    etas = _table(np.array([[300.0]]))
    assert not has_call_evidence(camera_only)
    assert build("B0", CFG, OPT).decide([camera_only], units, etas, NOW).decisions == []
    b1 = build("B1", CFG, OPT).decide([camera_only], units, etas, NOW)
    assert [a.unit_id for d in b1.decisions for a in d.assignments] == ["P1"]


def test_an_operator_review_counts_as_human_evidence_for_b0() -> None:
    """A review only happens because a control room looked, which the call era could also do."""
    reviewed = _incident("R1", pol=1, kinds=(SourceKind.CAMERA_EVENT, SourceKind.OPERATOR_REVIEW))
    assert has_call_evidence(reviewed)
    plan = build("B0", CFG, OPT).decide(
        [reviewed], [_unit("P1", Agency.POLICE)], _table(np.array([[300.0]])), NOW
    )
    assert len(plan.decisions) == 1


# -- B0's hand-off, the parameter that drives RQ2 ---------------------------------------------


def test_b0_delays_each_additional_agency_and_b1_does_not() -> None:
    incident = _incident("I1", amb=1, pol=1, fire=1)
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE), _unit("F1", Agency.FIRE)]
    etas = _table(np.array([[300.0], [300.0], [300.0]]))
    b0 = build("B0", _cfg_handoff(120.0), OPT).decide([incident], units, etas, NOW)
    # agencies are asked in a fixed order, so the delays are 0, 120, 240
    assert sorted(b0.delays.values()) == [0.0, 120.0, 240.0]
    assert b0.decisions[0].expected_last_arrival_s == pytest.approx(300.0 + 240.0)
    b1 = build("B1", CFG, OPT).decide([incident], units, etas, NOW)
    assert set(b1.delays.values()) == {0.0}
    assert b1.decisions[0].expected_last_arrival_s == pytest.approx(300.0)


def test_a_zero_handoff_makes_b0_and_b1_agree_on_a_called_in_incident() -> None:
    """The phase-10 sweep must include 0, where the only remaining B0/B1 difference is evidence.
    If these two disagreed here, the hand-off sweep would not isolate what it claims to."""
    incident = _incident("I1", amb=1, pol=1)
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = _table(np.array([[240.0], [420.0]]))
    b0 = build("B0", _cfg_handoff(0.0), OPT).decide([incident], units, etas, NOW)
    b1 = build("B1", CFG, OPT).decide([incident], units, etas, NOW)
    assert {(a.unit_id, a.eta_s) for d in b0.decisions for a in d.assignments} == {
        (a.unit_id, a.eta_s) for d in b1.decisions for a in d.assignments
    }


def test_a_negative_handoff_is_refused_by_the_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="negative time"):
        _cfg_handoff(-1.0)


# -- nobody double-books a unit ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3", "FULL"])
def test_no_policy_sends_one_unit_to_two_incidents(name: str) -> None:
    incidents = [_incident("I1", pol=1), _incident("I2", pol=1)]
    units = [_unit("P1", Agency.POLICE)]
    etas = _table(np.array([[120.0, 180.0]]))
    plan = build(name, CFG, OPT).decide(incidents, units, etas, NOW)
    sent = [a.unit_id for d in plan.decisions for a in d.assignments
            if a.role in (AssignmentRole.RESPOND, AssignmentRole.VERIFY)]  # fmt: skip
    assert len(sent) == len(set(sent)), f"{name} double-booked {sent}"


def test_nearest_per_agency_prefers_the_closer_unit() -> None:
    incident = _incident("I1", pol=1)
    units = [_unit("Pfar", Agency.POLICE), _unit("Pnear", Agency.POLICE)]
    etas = _table(np.array([[900.0], [120.0]]))
    plan = build("B1", CFG, OPT).decide([incident], units, etas, NOW)
    assert [a.unit_id for d in plan.decisions for a in d.assignments] == ["Pnear"]


def test_an_unreachable_agency_is_simply_not_sent() -> None:
    incident = _incident("I1", amb=1, pol=1)
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = _table(np.array([[300.0], [np.inf]]))
    plan = build("B1", CFG, OPT).decide([incident], units, etas, NOW)
    assert [a.unit_id for d in plan.decisions for a in d.assignments] == ["A1"]


# -- B1 vs B2: the joint assignment -----------------------------------------------------------


def test_b2_reproduces_the_c1_solver_exactly() -> None:
    """B2 must be the MILP and nothing else, or the joint-assignment rung is contaminated."""
    incidents = [_incident("I1", amb=1, pol=1), _incident("I2", amb=1, pol=1, severity=0.95)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE),
             _unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]  # fmt: skip
    etas = _table(np.array([[300.0, 600.0], [900.0, 240.0], [180.0, 480.0], [720.0, 360.0]]))
    plan = build("B2", CFG, OPT).decide(incidents, units, etas, NOW)
    batch = [
        DispatchIncident(
            id=i.id,
            node=int(i.node),
            requirements=i.requirements,
            severity_class=severity_class_of(i, OPT),
            ward_id=i.ward_id,
            posterior=i.posterior,
        )
        for i in incidents
    ]
    direct = c1_solve(batch, units, etas, OPT, NOW, "B2")
    assert plan.status == direct.status
    mine = {(d.incident_id, a.unit_id) for d in plan.decisions for a in d.assignments}
    theirs = {(d.incident_id, a.unit_id) for d in direct.decision for a in d.assignments}
    assert mine == theirs


def test_the_joint_model_can_beat_nearest_per_agency() -> None:
    """Not a guarantee for every instance, but the mechanism must be able to show up at all:
    here the greedy rule takes the unit the other incident needs more."""
    incidents = [_incident("I1", pol=1), _incident("I2", pol=1, severity=0.95)]
    units = [_unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]
    # P1 is nearest to I1 by a hair but is the only unit that can reach I2 quickly
    etas = _table(np.array([[100.0, 110.0], [120.0, 900.0]]))
    greedy = build("B1", CFG, OPT).decide(incidents, units, etas, NOW)
    joint = build("B2", CFG, OPT).decide(incidents, units, etas, NOW)
    greedy_last = max(a.eta_s for d in greedy.decisions for a in d.assignments)
    joint_last = max(a.eta_s for d in joint.decisions for a in d.assignments)
    assert joint_last <= greedy_last


# -- B2 vs B3: confidence ----------------------------------------------------------------------


def test_b3_holds_back_a_middling_incident_and_b2_does_not() -> None:
    """0.6 is above B2's fixed gate (0.5) but below tau_high (0.7), which is exactly the band
    where tiering changes the answer."""
    low = _incident("L1", amb=1, pol=1, posterior=0.6, kinds=(SourceKind.CAMERA_EVENT,))
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = _table(np.array([[300.0], [240.0]]))
    b2 = build("B2", CFG, OPT).decide([low], units, etas, NOW)
    b3 = build("B3", CFG, OPT).decide([low], units, etas, NOW)
    assert b2.decisions[0].tier is DispatchTier.FULL
    assert {a.role for a in b2.decisions[0].assignments} == {AssignmentRole.RESPOND}
    # 0.2 <= 0.6 < 0.7, so B3 verifies with one police unit and pre-alerts the ambulance
    assert b3.decisions[0].tier is DispatchTier.VERIFY
    roles = {a.role for a in b3.decisions[0].assignments}
    assert AssignmentRole.VERIFY in roles and AssignmentRole.PRE_ALERT in roles


def test_a_camera_only_posterior_reaches_nobody_under_b3() -> None:
    """D-029: a camera alarm alone puts the posterior near 1e-5, far below tau_low, so it lands
    in REVIEW. Confidence tiering does not turn CCTV into a dispatch trigger."""
    faint = _incident("F1", pol=1, posterior=1e-5, kinds=(SourceKind.CAMERA_EVENT,))
    plan = build("B3", CFG, OPT).decide(
        [faint], [_unit("P1", Agency.POLICE)], _table(np.array([[120.0]])), NOW
    )
    assert plan.decisions[0].tier is DispatchTier.REVIEW
    assert plan.decisions[0].assignments == []


def test_a_confident_incident_is_dispatched_in_full_under_b3() -> None:
    sure = _incident("S1", amb=1, pol=1, posterior=0.99)
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    plan = build("B3", CFG, OPT).decide([sure], units, _table(np.array([[300.0], [240.0]])), NOW)
    assert plan.decisions[0].tier is DispatchTier.FULL
    assert {a.unit_id for a in plan.decisions[0].assignments} == {"A1", "P1"}


# -- B3 vs FULL: anticipation -------------------------------------------------------------------


def test_full_with_the_corridor_disabled_reproduces_b3() -> None:
    """rho = 1 is the documented ablation, so FULL must then be B3 exactly. If it were not, any
    measured FULL-over-B3 gain could not be attributed to the corridor."""
    incidents = [_incident("I1", amb=1, pol=1, posterior=0.99)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = _table(np.array([[600.0], [400.0]]))
    opt = _opt_rho(1.0)
    b3 = build("B3", CFG, opt).decide(incidents, units, etas, NOW)
    full = build("FULL", CFG, opt).decide(incidents, units, etas, NOW)
    assert {(d.incident_id, a.unit_id, a.eta_s) for d in b3.decisions for a in d.assignments} == {
        (d.incident_id, a.unit_id, a.eta_s) for d in full.decisions for a in d.assignments
    }


def test_full_applies_the_corridor_when_police_can_open_it() -> None:
    incidents = [_incident("I1", amb=1, pol=1, posterior=0.99)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = _table(np.array([[600.0], [400.0]]))  # 0.85 * 600 = 510 >= 400
    full = build("FULL", CFG, OPT).decide(incidents, units, etas, NOW)
    amb = next(a for d in full.decisions for a in d.assignments if a.unit_id == "A1")
    assert amb.eta_s == pytest.approx(510.0)
    b3 = build("B3", CFG, OPT).decide(incidents, units, etas, NOW)
    amb_b3 = next(a for d in b3.decisions for a in d.assignments if a.unit_id == "A1")
    assert amb_b3.eta_s == pytest.approx(600.0)


# -- severity mapping -------------------------------------------------------------------------


def test_severity_maps_onto_the_objectives_own_classes() -> None:
    """A policy must not invent a class the dispatch objective does not price."""
    classes = set(OPT.dispatch.severity_weights.value)
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert severity_class_of(_incident("X", severity=score), OPT) in classes
    # and it is monotone: a higher score never maps to a lighter class
    weights = OPT.dispatch.severity_weights.value
    seen = [
        weights[severity_class_of(_incident("X", severity=s), OPT)] for s in np.linspace(0, 1, 11)
    ]
    assert seen == sorted(seen)


# -- what a policy is allowed to see -----------------------------------------------------------


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3", "FULL"])
def test_a_policy_runs_on_fused_incidents_alone(name: str) -> None:
    """The signature takes Incident, never GroundTruthIncident: a policy cannot reach the truth
    even by accident (tests/test_architecture.py enforces the import side of this)."""
    plan = build(name, CFG, OPT).decide(
        [_incident("I1", pol=1)], [_unit("P1", Agency.POLICE)], _table(np.array([[120.0]])), NOW
    )
    assert isinstance(plan.decisions, list)


# -- the credibility gate on the untiered policies ---------------------------------------------


def _cfg_gate(value: float) -> SimConfig:
    raw = CFG.model_dump(mode="json")
    raw["policies"]["min_dispatch_posterior"]["value"] = value
    return SimConfig.model_validate(raw)


@pytest.mark.parametrize("name", ["B0", "B1", "B2"])
def test_an_untiered_policy_ignores_a_candidate_below_the_gate(name: str) -> None:
    """Without a gate, each of these would send a unit to every one of the ~195,000 camera
    alarms a simulated week produces at the measured false-alarm rate (D-029)."""
    faint = _incident("F1", pol=1, posterior=0.01, kinds=(SourceKind.CALL,))
    plan = build(name, CFG, OPT).decide(
        [faint], [_unit("P1", Agency.POLICE)], _table(np.array([[120.0]])), NOW
    )
    assert plan.decisions == []


@pytest.mark.parametrize("name", ["B0", "B1", "B2"])
def test_lowering_the_gate_lets_the_same_candidate_through(name: str) -> None:
    faint = _incident("F1", pol=1, posterior=0.01, kinds=(SourceKind.CALL,))
    plan = build(name, _cfg_gate(0.001), OPT).decide(
        [faint], [_unit("P1", Agency.POLICE)], _table(np.array([[120.0]])), NOW
    )
    assert [a.unit_id for d in plan.decisions for a in d.assignments] == ["P1"]


def test_the_tiered_policies_do_not_use_the_fixed_gate() -> None:
    """B3 must reach its own REVIEW tier rather than being filtered out beforehand, or the
    confidence rung would be measuring the gate instead of the tiers."""
    faint = _incident("F1", pol=1, posterior=0.01, kinds=(SourceKind.CAMERA_EVENT,))
    plan = build("B3", CFG, OPT).decide(
        [faint], [_unit("P1", Agency.POLICE)], _table(np.array([[120.0]])), NOW
    )
    assert len(plan.decisions) == 1
    assert plan.decisions[0].tier is DispatchTier.REVIEW


def test_a_zero_gate_is_refused_by_the_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not a baseline anyone would run"):
        _cfg_gate(0.0)


def test_the_gate_keeps_the_eta_columns_aligned_with_the_incidents() -> None:
    """B2 drops filtered incidents from the ETA table too; a mismatch would silently assign
    units using another incident's travel times."""
    incidents = [
        _incident("Low", pol=1, posterior=0.01),
        _incident("High", pol=1, posterior=0.99),
    ]
    units = [_unit("P1", Agency.POLICE)]
    etas = _table(np.array([[900.0, 120.0]]))  # Low is far, High is near
    plan = build("B2", CFG, OPT).decide(incidents, units, etas, NOW)
    assert [d.incident_id for d in plan.decisions] == ["High"]
    assert plan.decisions[0].assignments[0].eta_s == pytest.approx(120.0)
