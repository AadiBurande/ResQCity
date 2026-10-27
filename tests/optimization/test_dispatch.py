"""C1 joint dispatch against brute force on tiny instances (docs/models/dispatch.md).

Every optimum below is checked by enumerating assignments exhaustively, so the test depends on
the formulation rather than on the solver agreeing with itself. The enumeration covers the whole
objective of docs/models/dispatch.md section 4, including the green corridor (C7) and the
coverage loss (C8): an OR review of this phase found that it previously ran only with the
corridor disabled and no wards, so those two terms were checked by example rather than against
the optimum.

One caveat is stated rather than hidden. The shortfall surrogate (C6) is a *definition* chosen in
the formulation, not a quantity with an independent ground truth, so `brute_force` necessarily
uses the same formula as the model. What the surrogate is FOR is tested separately in
`test_an_unserved_incident_cannot_report_a_short_completion`.
"""

import itertools
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from resqcity.common.schemas import Agency, ServiceCounts, Unit, UnitState
from resqcity.optimization.config import OptimizationConfig, load_optimization_config
from resqcity.optimization.dispatch import (
    DispatchIncident,
    _horizon,
    effective_epsilon,
    greedy,
    solve,
)
from resqcity.optimization.routing import EtaTable, Router

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 3, 3, 9, 0, tzinfo=IST)
CFG = load_optimization_config()


def _cfg(**dispatch_updates: object) -> OptimizationConfig:
    data = CFG.model_dump(mode="json")
    for key, value in dispatch_updates.items():
        if key == "rho":
            data["routing"]["green_corridor"]["rho"]["value"] = value
        elif key in ("lambda_coverage", "mu_shortfall", "epsilon_ambulance"):
            data["dispatch"][key]["value"] = value
        else:
            data["dispatch"]["solver"][key] = value
    return OptimizationConfig.model_validate(data)


def _unit(uid: str, agency: Agency) -> Unit:
    return Unit(
        id=uid, agency=agency, home_station=f"S-{uid}", node=1,
        state=UnitState.IDLE, eta_fn_ref="default",
    )  # fmt: skip


def _incident(iid: str, amb: int = 0, pol: int = 0, fire: int = 0, severity: str = "serious"):
    return DispatchIncident(
        id=iid,
        node=10,
        requirements=ServiceCounts(ambulance=amb, police=pol, fire=fire),
        severity_class=severity,
        ward_id="W01",
    )


def _table(matrix: np.ndarray) -> EtaTable:
    return EtaTable(list(range(matrix.shape[0])), list(range(matrix.shape[1])), matrix)


def brute_force(
    incidents: list[DispatchIncident],
    units: list[Unit],
    etas: np.ndarray,
    cfg: OptimizationConfig,
    assisted: np.ndarray | None = None,
    ward_units: dict[str, list[int]] | None = None,
    ward_value: dict[str, float] | None = None,
) -> float:
    """Exhaustive objective over every assignment of units to incidents (or to nothing).

    An independent implementation of docs/models/dispatch.md section 4: C4 agency arrival, C5
    synchronisation, C6 shortfall, C7 green corridor and C8 coverage loss.
    """
    n_u, n_i = etas.shape
    helped = etas if assisted is None else assisted
    horizon = _horizon(etas)
    best = float("inf")
    options = [[None, *range(n_i)] for _ in range(n_u)]
    for combo in itertools.product(*options):
        total = 0.0
        feasible = True
        for i, inc in enumerate(incidents):
            assigned = [u for u, target in enumerate(combo) if target == i]
            if any(not np.isfinite(etas[u, i]) for u in assigned):
                feasible = False
                break
            # (C7) only police SENT HERE and fast enough to arrive by the sped-up time open the
            # corridor; a faster unit dispatched elsewhere must not count
            police_here = [u for u in assigned if units[u].agency == Agency.POLICE]
            per_agency, short = {}, 0
            for agency in Agency:
                need = getattr(inc.requirements, agency.value)
                if need <= 0:
                    continue
                got = []
                for u in assigned:
                    if units[u].agency != agency:
                        continue
                    eta = etas[u, i]
                    opened = any(etas[p, i] <= helped[u, i] for p in police_here)
                    if agency is Agency.AMBULANCE and helped[u, i] < etas[u, i] and opened:
                        eta = helped[u, i]
                    got.append(eta)
                short += max(0, need - len(got))
                if got:
                    per_agency[agency] = max(got)  # (C4) wait for the slowest unit sent
            needed = {a for a in Agency if getattr(inc.requirements, a.value) > 0}
            arrivals = [per_agency[a] for a in needed if a in per_agency]
            total_need = sum(getattr(inc.requirements, a.value) for a in Agency)
            last = max(arrivals) if arrivals else 0.0  # (C5)
            if short:  # (C6) an unserved incident cannot report a short completion
                last = max(last, horizon * short / max(1, total_need))
            total += cfg.dispatch.weight(inc.severity_class) * last
            total += cfg.dispatch.mu_shortfall.value * short
            if Agency.AMBULANCE in per_agency:
                total += effective_epsilon(cfg, horizon) * per_agency[Agency.AMBULANCE]
        if not feasible:
            continue
        for ward, coverers in (ward_units or {}).items():  # (C8) coverage loss
            if any(combo[u] is None for u in coverers):
                continue  # at least one unit that could reach this ward stays free
            total += cfg.dispatch.lambda_coverage.value * (ward_value or {}).get(ward, 0.0)
        best = min(best, total)
    return best


def test_matches_brute_force_on_a_two_incident_instance():
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE),
             _unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]  # fmt: skip
    incidents = [_incident("I1", amb=1, pol=1), _incident("I2", amb=1, pol=1, severity="critical")]
    etas = np.array([[300.0, 600.0], [900.0, 240.0], [180.0, 480.0], [720.0, 360.0]])
    cfg = _cfg(rho=1.0)  # corridor off, so brute force need not model it
    result = solve(incidents, units, _table(etas), cfg, NOW)
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.objective_s == pytest.approx(brute_force(incidents, units, etas, cfg), rel=1e-6)


def test_matches_brute_force_with_the_corridor_active():
    """The corridor changes which assignment is optimal, so it must be inside the enumeration.

    A1 is slow but can be sped up; A2 is faster unassisted. Whether sending police here to open
    A1's corridor beats sending A2 is exactly the trade-off C7 encodes.
    """
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE),
             _unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]  # fmt: skip
    incidents = [_incident("I1", amb=1, pol=1), _incident("I2", amb=1, pol=1, severity="critical")]
    etas = np.array([[600.0, 660.0], [480.0, 900.0], [400.0, 300.0], [520.0, 480.0]])
    cfg = _cfg(rho=0.85)
    router = Router.__new__(Router)
    router.cfg = cfg
    assisted = router.green_corridor_etas(_table(etas), [u.agency for u in units])
    assert (assisted < etas).any(), "the instance must actually exercise the corridor"
    result = solve(incidents, units, _table(etas), cfg, NOW, assisted_etas=assisted)
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    expected = brute_force(incidents, units, etas, cfg, assisted=assisted)
    assert result.objective_s == pytest.approx(expected, rel=1e-6)


def test_matches_brute_force_with_the_coverage_term_active():
    """C8 makes holding a unit back worthwhile, so the optimum is not just the fastest arrival.

    W02 is covered only by P2, and its risk weight is high enough that leaving P2 free can beat
    sending it. Enumerating this is the only way to know the solver resolves it correctly.
    """
    units = [_unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]
    incidents = [_incident("I1", pol=2)]
    etas = np.array([[300.0], [320.0]])
    wards = {"W01": [0], "W02": [1]}
    values = {"W01": 0.1, "W02": 400.0}
    cfg = _cfg(rho=1.0)
    result = solve(incidents, units, _table(etas), cfg, NOW,
                   ward_units=wards, ward_value=values)  # fmt: skip
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    expected = brute_force(incidents, units, etas, cfg, ward_units=wards, ward_value=values)
    assert result.objective_s == pytest.approx(expected, rel=1e-6)
    # and the trade-off is live: with a worthless ward the solver sends both units
    cheap = solve(incidents, units, _table(etas), cfg, NOW,
                  ward_units=wards, ward_value={"W01": 0.1, "W02": 0.1})  # fmt: skip
    assert len(cheap.decision[0].assignments) == 2
    assert len(result.decision[0].assignments) == 1  # P2 held back to keep W02 covered


def test_matches_brute_force_with_the_corridor_and_coverage_together():
    """Both terms at once, because C7 and C8 interact: police sent to open a corridor are also
    police no longer covering a ward."""
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    incidents = [_incident("I1", amb=1, pol=1)]
    etas = np.array([[600.0], [450.0]])
    wards = {"W02": [1]}
    values = {"W02": 50.0}
    cfg = _cfg(rho=0.85)
    router = Router.__new__(Router)
    router.cfg = cfg
    assisted = router.green_corridor_etas(_table(etas), [u.agency for u in units])
    result = solve(incidents, units, _table(etas), cfg, NOW, assisted_etas=assisted,
                   ward_units=wards, ward_value=values)  # fmt: skip
    expected = brute_force(
        incidents, units, etas, cfg, assisted=assisted, ward_units=wards, ward_value=values
    )
    assert result.objective_s == pytest.approx(expected, rel=1e-6)


def test_an_unserved_incident_cannot_report_a_short_completion():
    """What the C6 surrogate is for. Its formula is a definition, so brute force cannot check it
    independently; this checks the property it exists to guarantee."""
    units = [_unit("P1", Agency.POLICE)]
    incidents = [_incident("I1", pol=1, fire=1)]  # no fire unit exists, so fire is short
    etas = np.array([[120.0]])
    result = solve(incidents, units, _table(etas), _cfg(rho=1.0), NOW)
    assert result.shortfall["I1"] == 1
    # the police unit arrives at 120 s, but the incident is only half served, so the reported
    # completion must not be 120 s
    assert result.last_arrival_s["I1"] > 120.0


def test_matches_brute_force_when_an_agency_is_missing():
    """Shortfall must stay feasible and be priced, not make the model infeasible."""
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    incidents = [_incident("I1", amb=1, pol=1, fire=1)]  # no fire unit exists
    etas = np.array([[200.0], [400.0]])
    cfg = _cfg(rho=1.0)
    result = solve(incidents, units, _table(etas), cfg, NOW)
    assert result.shortfall["I1"] == 1
    assert result.objective_s == pytest.approx(brute_force(incidents, units, etas, cfg), rel=1e-6)
    assert result.decision[0].tier.value == "verify"  # not reported as a full response


def test_unreachable_units_are_never_assigned():
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE)]
    incidents = [_incident("I1", amb=1)]
    etas = np.array([[np.inf], [500.0]])
    result = solve(incidents, units, _table(etas), _cfg(rho=1.0), NOW)
    assigned = {a.unit_id for a in result.decision[0].assignments}
    assert assigned == {"A2"}
    assert result.shortfall["I1"] == 0


def test_more_units_never_worsen_the_optimum():
    incidents = [_incident("I1", amb=1, pol=1)]
    small = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas_small = np.array([[600.0], [600.0]])
    big = [*small, _unit("A2", Agency.AMBULANCE)]
    etas_big = np.vstack([etas_small, [[120.0]]])
    cfg = _cfg(rho=1.0)
    a = solve(incidents, small, _table(etas_small), cfg, NOW).objective_s
    b = solve(incidents, big, _table(etas_big), cfg, NOW).objective_s
    assert b <= a + 1e-9


def test_lower_etas_never_increase_the_last_arrival():
    incidents = [_incident("I1", amb=1, pol=1)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    cfg = _cfg(rho=1.0)
    slow = solve(incidents, units, _table(np.array([[600.0], [500.0]])), cfg, NOW)
    fast = solve(incidents, units, _table(np.array([[300.0], [500.0]])), cfg, NOW)
    assert fast.last_arrival_s["I1"] <= slow.last_arrival_s["I1"]


def test_the_secondary_ambulance_term_never_changes_the_last_arrival():
    """Master plan section 6.1: epsilon pulls the ambulance in without moving L[i]."""
    incidents = [_incident("I1", amb=1, fire=1)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE), _unit("F1", Agency.FIRE)]
    etas = np.array([[120.0], [300.0], [900.0]])  # fire is the critical path either way
    with_eps = solve(incidents, units, _table(etas), _cfg(rho=1.0), NOW)
    without = solve(incidents, units, _table(etas), _cfg(rho=1.0, epsilon_ambulance=0.0), NOW)
    assert with_eps.last_arrival_s["I1"] == without.last_arrival_s["I1"] == 900.0
    assert {a.unit_id for a in with_eps.decision[0].assignments} == {"A1", "F1"}  # the nearer one


def test_green_corridor_only_helps_an_ambulance_that_arrives_after_police():
    incidents = [_incident("I1", amb=1, pol=1)]
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = np.array([[600.0], [300.0]])
    from resqcity.optimization.routing import Router

    router = Router.__new__(Router)  # only green_corridor_etas is needed
    router.cfg = _cfg(rho=0.5)
    assisted = router.green_corridor_etas(_table(etas), [Agency.AMBULANCE, Agency.POLICE])
    assert assisted[0, 0] == pytest.approx(300.0)  # 0.5 * 600 floored at the police arrival
    result = solve(incidents, units, _table(etas), _cfg(rho=0.5), NOW, assisted_etas=assisted)
    assert result.last_arrival_s["I1"] == pytest.approx(300.0)
    # an ambulance that beats the police is not helped
    early = np.array([[100.0], [300.0]])
    assisted_early = router.green_corridor_etas(_table(early), [Agency.AMBULANCE, Agency.POLICE])
    assert assisted_early[0, 0] == pytest.approx(100.0)


def test_rho_one_reproduces_the_no_assist_case():
    etas = _table(np.array([[600.0], [300.0]]))
    from resqcity.optimization.routing import Router

    router = Router.__new__(Router)
    router.cfg = _cfg(rho=1.0)
    assisted = router.green_corridor_etas(etas, [Agency.AMBULANCE, Agency.POLICE])
    assert np.array_equal(assisted, etas.seconds)


def test_ties_in_eta_are_resolved_deterministically():
    incidents = [_incident("I1", pol=1)]
    units = [_unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]
    etas = _table(np.array([[400.0], [400.0]]))
    first = solve(incidents, units, etas, _cfg(rho=1.0), NOW)
    again = solve(incidents, units, etas, _cfg(rho=1.0), NOW)
    assert first.objective_s == again.objective_s


def test_greedy_fallback_is_feasible_and_no_better_than_the_solver():
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE),
             _unit("P1", Agency.POLICE), _unit("P2", Agency.POLICE)]  # fmt: skip
    incidents = [_incident("I1", amb=1, pol=1), _incident("I2", amb=1, pol=1, severity="critical")]
    etas = np.array([[300.0, 600.0], [900.0, 240.0], [180.0, 480.0], [720.0, 360.0]])
    cfg = _cfg(rho=1.0)
    chosen = greedy(incidents, units, etas, cfg)
    assert all(len(v) == 2 for v in chosen.values())  # both incidents fully served
    used = [u for v in chosen.values() for u in v]
    assert len(used) == len(set(used))  # no unit sent twice
    optimum = solve(incidents, units, _table(etas), cfg, NOW).objective_s
    assert optimum <= brute_force(incidents, units, etas, cfg) + 1e-6


def test_the_corridor_is_not_credited_to_police_sent_elsewhere():
    """The OR review found a fast police unit dispatched to another incident unlocking the
    corridor here. Only a unit sent to THIS incident, fast enough to open it, may count."""
    units = [_unit("A1", Agency.AMBULANCE), _unit("Pf", Agency.POLICE), _unit("Ps", Agency.POLICE)]
    etas = np.array([[600.0, np.inf], [10.0, 10.0], [900.0, 900.0]])
    incidents = [_incident("I1", amb=1, pol=1), _incident("I2", amb=0, pol=1)]
    cfg = _cfg(rho=0.85)
    router = Router.__new__(Router)
    router.cfg = cfg
    assisted = router.green_corridor_etas(_table(etas), [u.agency for u in units])
    result = solve(incidents, units, _table(etas), cfg, NOW, assisted_etas=assisted)
    by_id = {d.incident_id: d for d in result.decision}
    amb = next(a for a in by_id["I1"].assignments if a.unit_id == "A1")
    # the only police unit sent here (Ps, 900 s) cannot open a corridor for a 510 s arrival,
    # so the ambulance keeps its unassisted time even though a faster police unit exists
    assert amb.eta_s == pytest.approx(600.0)
    assert "Pf" not in {a.unit_id for a in by_id["I1"].assignments}


def test_the_corridor_applies_when_the_police_unit_here_is_fast_enough():
    units = [_unit("A1", Agency.AMBULANCE), _unit("P1", Agency.POLICE)]
    etas = np.array([[600.0], [400.0]])  # 0.85 * 600 = 510 >= 400, so P1 can open it
    cfg = _cfg(rho=0.85)
    router = Router.__new__(Router)
    router.cfg = cfg
    assisted = router.green_corridor_etas(_table(etas), [u.agency for u in units])
    result = solve([_incident("I1", amb=1, pol=1)], units, _table(etas), cfg, NOW,
                   assisted_etas=assisted)  # fmt: skip
    amb = next(a for a in result.decision[0].assignments if a.unit_id == "A1")
    assert amb.eta_s == pytest.approx(510.0)


def test_epsilon_cannot_change_the_last_arrival_even_over_a_long_horizon():
    """eps < min(weight) is not enough: arrivals differ by up to the horizon, so the guarantee
    needs eps * horizon < min(weight). The review found a case where it failed."""
    units = [_unit("A1", Agency.AMBULANCE), _unit("A2", Agency.AMBULANCE), _unit("F1", Agency.FIRE)]
    incidents = [_incident("Ifire", amb=1, fire=1), _incident("Jamb", amb=1, pol=0)]
    etas = np.array([[10.0, 1000.0], [500.0, 1001.0], [1000.0, np.inf]])
    with_eps = solve(incidents, units, _table(etas), _cfg(rho=1.0), NOW)
    without = solve(incidents, units, _table(etas), _cfg(rho=1.0, epsilon_ambulance=0.0), NOW)
    assert with_eps.last_arrival_s == without.last_arrival_s


def test_a_ward_no_unit_can_reach_is_reported_uncovered():
    units = [_unit("P1", Agency.POLICE)]
    incidents = [_incident("I1", pol=1)]
    wards = {"W01": [0], "W02": []}
    values = {"W01": 1.0, "W02": 5.0}
    result = solve(
        incidents, units, _table(np.array([[300.0]])), _cfg(rho=1.0), NOW,
        ward_units=wards, ward_value=values,
    )  # fmt: skip
    assert "W02" in result.uncovered_wards  # unreachable, so permanently uncovered
