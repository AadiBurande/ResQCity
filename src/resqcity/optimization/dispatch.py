"""C1 — joint multi-agency dispatch. Formulation: ``docs/models/dispatch.md``.

Constraint comments below carry the equation numbers from that document; if you change one,
change both. Everything is solved in **integer seconds** (CP-SAT), which is far finer than the
resolution of anything being modelled.
"""

import math
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DispatchDecision,
    DispatchTier,
    Explanation,
    Factor,
    ServiceCounts,
    Unit,
)
from resqcity.optimization.config import OptimizationConfig
from resqcity.optimization.routing import EtaTable

SCALE = 1  # objective is in seconds; weights are scaled to integers by WEIGHT_SCALE
WEIGHT_SCALE = 1000  # severity weights and epsilon are floats; CP-SAT needs integer coefficients


@dataclass(frozen=True)
class DispatchIncident:
    """What the dispatcher knows about one open incident."""

    id: str
    node: int
    requirements: ServiceCounts
    severity_class: str
    ward_id: str = ""
    posterior: float = 1.0


@dataclass
class DispatchResult:
    decision: list[DispatchDecision]
    explanation: list[Explanation]
    status: str
    objective_s: float
    solve_s: float
    last_arrival_s: dict[str, float] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)
    uncovered_wards: list[str] = field(default_factory=list)


def _required(incident: DispatchIncident, agency: Agency) -> int:
    return int(getattr(incident.requirements, agency.value))


def _horizon(matrix: np.ndarray) -> int:
    """T: the slowest finite ETA in the instance -- a valid big-M (docs §2)."""
    finite = matrix[np.isfinite(matrix)]
    return math.ceil(finite.max()) if finite.size else 1


def greedy(
    incidents: list[DispatchIncident],
    units: list[Unit],
    etas: np.ndarray,
    cfg: OptimizationConfig,
) -> dict[int, list[int]]:
    """Fallback (docs §7): incidents by decreasing severity weight, nearest free units first."""
    order = sorted(
        range(len(incidents)),
        key=lambda i: (-cfg.dispatch.weight(incidents[i].severity_class), incidents[i].id),
    )
    taken: set[int] = set()
    chosen: dict[int, list[int]] = {i: [] for i in range(len(incidents))}
    for i in order:
        for agency in Agency:
            need = _required(incidents[i], agency)
            if need <= 0:
                continue
            free = [
                u
                for u, unit in enumerate(units)
                if unit.agency == agency and u not in taken and np.isfinite(etas[u, i])
            ]
            for u in sorted(free, key=lambda u: (etas[u, i], units[u].id))[:need]:
                taken.add(u)
                chosen[i].append(u)
    return chosen


def _coverage_sets(
    units: list[Unit], ward_units: dict[str, list[int]], wards: list[str]
) -> dict[str, list[int]]:
    return {w: ward_units.get(w, []) for w in wards}


def solve(
    incidents: list[DispatchIncident],
    units: list[Unit],
    etas: EtaTable,
    cfg: OptimizationConfig,
    now: datetime,
    policy: str = "C1",
    assisted_etas: np.ndarray | None = None,
    ward_units: dict[str, list[int]] | None = None,
    ward_value: dict[str, float] | None = None,
) -> DispatchResult:
    """Solve C1 for one batch. ``assisted_etas`` are the green-corridor ETAs from
    :meth:`routing.Router.green_corridor_etas`; pass ``None`` to disable the corridor."""
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    base = etas.seconds
    n_i, n_u = len(incidents), len(units)
    if n_i == 0 or n_u == 0:
        return DispatchResult([], [], "EMPTY", 0.0, time.perf_counter() - started)
    assisted = base if assisted_etas is None else assisted_etas
    horizon = _horizon(base)
    d = cfg.dispatch
    model = cp_model.CpModel()

    x = {
        (u, i): model.new_bool_var(f"x_{u}_{i}")
        for i in range(n_i)
        for u in range(n_u)
        if np.isfinite(base[u, i])  # (C3) unreachable units are not assignable
    }
    arrival = {
        (i, a): model.new_int_var(0, horizon, f"A_{i}_{a.value}")
        for i in range(n_i)
        for a in Agency
        if _required(incidents[i], a) > 0
    }
    last = [model.new_int_var(0, horizon, f"L_{i}") for i in range(n_i)]
    short = {
        (i, a): model.new_int_var(0, _required(incidents[i], a), f"s_{i}_{a.value}")
        for i in range(n_i)
        for a in Agency
        if _required(incidents[i], a) > 0
    }

    for u in range(n_u):  # (C1) one incident per unit
        here = [x[(u, i)] for i in range(n_i) if (u, i) in x]
        if here:
            model.add_at_most_one(here)

    for i in range(n_i):
        for a in Agency:
            need = _required(incidents[i], a)
            if need <= 0:
                continue
            sent = [x[(u, i)] for u in range(n_u) if units[u].agency == a and (u, i) in x]
            model.add(sum(sent) + short[(i, a)] >= need)  # (C2) cover or pay

    # (C7) green corridor: g = x AND (some police sent here)
    police_here: dict[int, list] = {
        i: [x[(p, i)] for p in range(n_u) if units[p].agency == Agency.POLICE and (p, i) in x]
        for i in range(n_i)
    }
    g: dict[tuple[int, int], object] = {}
    for i in range(n_i):
        if not police_here[i] or (i, Agency.AMBULANCE) not in arrival:
            continue
        any_police = model.new_bool_var(f"pol_{i}")
        model.add_max_equality(any_police, police_here[i])
        for u in range(n_u):
            if units[u].agency != Agency.AMBULANCE or (u, i) not in x:
                continue
            if assisted[u, i] >= base[u, i]:
                continue  # not eligible: the corridor cannot help this ambulance
            var = model.new_bool_var(f"g_{u}_{i}")
            model.add_bool_and([x[(u, i)], any_police]).only_enforce_if(var)
            model.add_bool_or([x[(u, i)].negated(), any_police.negated()]).only_enforce_if(
                var.negated()
            )
            g[(u, i)] = var

    for i in range(n_i):
        for a in Agency:
            if (i, a) not in arrival:
                continue
            for u in range(n_u):
                if units[u].agency != a or (u, i) not in x:
                    continue
                # (C4) agency arrival is at least the slowest assigned unit
                plain, helped = round(base[u, i]), round(assisted[u, i])
                if (u, i) in g:
                    model.add(arrival[(i, a)] >= helped).only_enforce_if(g[(u, i)])
                    model.add(arrival[(i, a)] >= plain).only_enforce_if(
                        [x[(u, i)], g[(u, i)].negated()]
                    )
                else:
                    model.add(arrival[(i, a)] >= plain).only_enforce_if(x[(u, i)])
            model.add(last[i] >= arrival[(i, a)])  # (C5) wait for the last required agency

    for i in range(n_i):  # (C6) an unserved incident cannot report a short completion
        total_need = sum(_required(incidents[i], a) for a in Agency)
        missing = [short[(i, a)] for a in Agency if (i, a) in short]
        if missing and total_need > 0:
            model.add(last[i] * max(1, total_need) >= horizon * sum(missing))

    # (C8) coverage loss: a ward is uncovered when this batch takes every unit that could reach it
    wards = sorted(ward_value or {})
    cover_sets = _coverage_sets(units, ward_units or {}, wards)
    z = {w: model.new_bool_var(f"z_{w}") for w in wards if cover_sets[w]}
    for w, var in z.items():
        kept = []
        for u in cover_sets[w]:
            busy = [x[(u, i)] for i in range(n_i) if (u, i) in x]
            if not busy:
                kept.append(1)
                continue
            free = model.new_bool_var(f"free_{w}_{u}")
            model.add(sum(busy) == 0).only_enforce_if(free)
            model.add(sum(busy) >= 1).only_enforce_if(free.negated())
            kept.append(free)
        model.add(sum(kept) >= 1).only_enforce_if(var.negated())
        model.add(sum(kept) == 0).only_enforce_if(var)

    weights = {i: round(d.weight(incidents[i].severity_class) * WEIGHT_SCALE) for i in range(n_i)}
    eps = round(d.epsilon_ambulance.value * WEIGHT_SCALE)
    terms = [weights[i] * last[i] for i in range(n_i)]
    terms += [
        eps * arrival[(i, Agency.AMBULANCE)] for i in range(n_i) if (i, Agency.AMBULANCE) in arrival
    ]
    terms += [
        round(d.lambda_coverage.value * WEIGHT_SCALE * (ward_value or {}).get(w, 0.0)) * var
        for w, var in z.items()
    ]
    terms += [round(d.mu_shortfall.value * WEIGHT_SCALE) * var for var in short.values()]
    model.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = d.solver.time_limit_s
    solver.parameters.num_workers = d.solver.workers
    status = solver.solve(model)
    names = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}
    if status not in names:
        if not d.solver.greedy_fallback:
            raise RuntimeError(f"dispatch solver returned {solver.status_name(status)}")
        chosen = greedy(incidents, units, base, cfg)
        return _build(
            incidents, units, etas, assisted, chosen, cfg, now, policy, "GREEDY_FALLBACK",
            float("nan"), time.perf_counter() - started,
        )  # fmt: skip
    chosen = {
        i: [u for u in range(n_u) if (u, i) in x and solver.value(x[(u, i)])] for i in range(n_i)
    }
    return _build(
        incidents, units, etas, assisted, chosen, cfg, now, policy, names[status],
        solver.objective_value / WEIGHT_SCALE, time.perf_counter() - started,
        uncovered=[w for w, var in z.items() if solver.value(var)],
    )  # fmt: skip


def _build(
    incidents: list[DispatchIncident],
    units: list[Unit],
    etas: EtaTable,
    assisted: np.ndarray,
    chosen: dict[int, list[int]],
    cfg: OptimizationConfig,
    now: datetime,
    policy: str,
    status: str,
    objective: float,
    solve_s: float,
    uncovered: list[str] | None = None,
) -> DispatchResult:
    """Turn the chosen assignments into schema objects, recomputing arrival times from the data."""
    decisions, explanations = [], []
    last_arrival, shortfall = {}, {}
    for i, inc in enumerate(incidents):
        sent = chosen.get(i, [])
        police_sent = any(units[u].agency == Agency.POLICE for u in sent)
        assignments, per_agency = [], {}
        for u in sent:
            agency = units[u].agency
            eta = etas.seconds[u, i]
            if agency == Agency.AMBULANCE and police_sent:
                eta = assisted[u, i]
            per_agency.setdefault(agency, []).append(eta)
            assignments.append(
                Assignment(unit_id=units[u].id, eta_s=float(eta), role=AssignmentRole.RESPOND)
            )
        needed = {a for a in Agency if _required(inc, a) > 0}
        got = {a: len(v) for a, v in per_agency.items()}
        missing = {a: _required(inc, a) - got.get(a, 0) for a in needed}
        short_total = sum(max(0, m) for m in missing.values())
        arrivals = [max(per_agency[a]) for a in needed if a in per_agency]
        complete = short_total == 0 and arrivals
        last = float(max(arrivals)) if arrivals else float("nan")
        last_arrival[inc.id] = last if complete else float("inf")
        shortfall[inc.id] = short_total
        tier = DispatchTier.FULL if complete else DispatchTier.VERIFY
        decisions.append(
            DispatchDecision(
                id=f"DEC-{inc.id}",
                incident_id=inc.id,
                created_ts=now,
                policy=policy,
                assignments=assignments,
                tier=tier,
                expected_last_arrival_s=last if complete else None,
                objective_breakdown={
                    "severity_weight": cfg.dispatch.weight(inc.severity_class),
                    "last_arrival_s": last if complete else float("inf"),
                    "shortfall_units": float(short_total),
                },
            )
        )
        explanations.append(_explain(inc, units, per_agency, missing, police_sent, cfg))
    return DispatchResult(
        decision=decisions,
        explanation=explanations,
        status=status,
        objective_s=objective,
        solve_s=solve_s,
        last_arrival_s=last_arrival,
        shortfall=shortfall,
        uncovered_wards=sorted(uncovered or []),
    )


def _explain(
    inc: DispatchIncident,
    units: list[Unit],
    per_agency: dict[Agency, list[float]],
    missing: dict[Agency, int],
    police_sent: bool,
    cfg: OptimizationConfig,
) -> Explanation:
    factors = [
        Factor(name="severity", value=cfg.dispatch.weight(inc.severity_class), weight=1.0),
        Factor(name="posterior", value=inc.posterior, weight=1.0),
    ]
    parts = []
    for agency in sorted(per_agency, key=lambda a: a.value):
        slowest = max(per_agency[agency])
        factors.append(Factor(name=f"{agency.value}_arrival_s", value=float(slowest), weight=1.0))
        parts.append(f"{len(per_agency[agency])} {agency.value} in {slowest / 60:.1f} min")
    short = {a.value: m for a, m in missing.items() if m > 0}
    if short:
        parts.append("short of " + ", ".join(f"{n} {a}" for a, n in sorted(short.items())))
        factors.append(
            Factor(name="shortfall_units", value=float(sum(short.values())), weight=-1.0)
        )
    if police_sent and cfg.routing.green_corridor.rho.value < 1.0:
        parts.append("green corridor applied to the ambulance")
    text = (
        f"{inc.severity_class} {inc.id}: " + "; ".join(parts)
        if parts
        else f"{inc.id}: nothing sent"
    )
    return Explanation(decision_id=f"DEC-{inc.id}", text=text, factors=factors)
