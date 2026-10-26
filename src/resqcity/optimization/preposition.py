"""C3 — pre-positioning with fairness (MEXCLP). Formulation: ``docs/models/preposition.md``.

Constraint comments carry the equation numbers from that document. The model is solved twice,
with and without the fairness constraint, so the **price of fairness** (RQ4) is measured rather
than asserted.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from resqcity.optimization.config import OptimizationConfig

WEIGHT_SCALE = 1000  # demand weights are floats; CP-SAT needs integer coefficients
SHORTFALL_PENALTY = 10_000  # per ward left below the fairness requirement (soft constraint)


@dataclass
class PrepositionResult:
    """Where each unit should end up, and what it bought."""

    placement: list[int]  # base index per unit
    expected_covered_demand: float
    moves: int
    move_seconds: float
    uncovered_wards: list[str] = field(default_factory=list)
    status: str = ""
    solve_s: float = 0.0

    def moved_units(self, current: list[int]) -> list[int]:
        return [u for u, k in enumerate(self.placement) if k != current[u]]


def required_units(alpha: float, q: float) -> int:
    """(P5) units within reach so that at least one is free with probability >= alpha.

    P(at least one of n free) = 1 - q^n >= alpha  =>  n >= log(1-alpha)/log(q).
    """
    if alpha <= 0:
        return 0
    if q <= 0:
        return 1
    if alpha >= 1:
        raise ValueError("alpha = 1 would need infinitely many units")
    # ceil() on a floating ratio over-provisions at exact boundaries (alpha 0.9, q 0.1 gives
    # 1.0000000000000002 -> 2, although one unit already satisfies it), so verify and step back.
    n = max(1, math.ceil(math.log1p(-alpha) / math.log(q)))
    while n > 1 and 1.0 - q ** (n - 1) >= alpha - 1e-12:
        n -= 1
    return n


def expected_covered(
    placement: list[int],
    demand: np.ndarray,
    covers: np.ndarray,
    q: float,
) -> float:
    """Objective value of a placement: sum_j d_j (1 - q^{n_j}). Used by the tests' brute force."""
    counts = np.zeros(covers.shape[0])
    for base in placement:
        counts += covers[:, base]
    return float(np.sum(demand * (1.0 - q**counts)))


def solve(
    unit_bases: list[int],
    move_seconds: np.ndarray,
    demand: np.ndarray,
    covers: np.ndarray,
    reach: np.ndarray,
    cfg: OptimizationConfig,
    capacity: list[int] | None = None,
    ward_ids: list[str] | None = None,
    fairness: bool | None = None,
) -> PrepositionResult:
    """Place idle units of one agency.

    ``covers[j, k]``  base k covers ward j within the coverage radius R.
    ``reach[j, k]``   base k reaches ward j within the fairness time T_ward.
    ``move_seconds[u, k]`` travel time for unit u to reach base k.
    """
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    n_units, n_bases = move_seconds.shape
    n_wards = demand.shape[0]
    p = cfg.preposition
    q = p.busy_fraction.value
    use_fairness = p.fairness.enabled if fairness is None else fairness
    caps = capacity or [n_units] * n_bases
    # L must be at least max_j n_j, where n_j sums the capacity of every base covering ward j.
    # Capping at the largest SINGLE base capacity truncates the objective and makes the solver
    # move units away from a ward served by several bases.
    per_ward = [
        sum(caps[k] for k in range(n_bases) if covers[j, k]) for j in range(demand.shape[0])
    ]
    max_cover = max(1, min(n_units, max(per_ward, default=1)))
    model = cp_model.CpModel()

    y = {(u, k): model.new_bool_var(f"y_{u}_{k}") for u in range(n_units) for k in range(n_bases)}
    for u in range(n_units):  # (P1) every unit ends somewhere
        model.add_exactly_one([y[(u, k)] for k in range(n_bases)])
    counts = []
    for k in range(n_bases):  # (P2) capacity
        n_k = model.new_int_var(0, caps[k], f"n_{k}")
        model.add(n_k == sum(y[(u, k)] for u in range(n_units)))
        counts.append(n_k)

    z = {
        (j, level): model.new_bool_var(f"z_{j}_{level}")
        for j in range(n_wards)
        for level in range(1, max_cover + 1)
    }
    for j in range(n_wards):  # (P3) a ward cannot claim more cover than it has
        near = [counts[k] for k in range(n_bases) if covers[j, k]]
        model.add(sum(z[(j, level)] for level in range(1, max_cover + 1)) <= sum(near))

    # (P4) relocation budget
    moves = [y[(u, k)] for u in range(n_units) for k in range(n_bases) if k != unit_bases[u]]
    if moves:
        model.add(sum(moves) <= p.relocation.max_moves.value)
        model.add(
            sum(
                round(move_seconds[u, k]) * y[(u, k)]
                for u in range(n_units)
                for k in range(n_bases)
                if k != unit_bases[u]
            )
            <= round(p.relocation.max_total_s.value)
        )

    # (P5) fairness, soft so an impossible requirement is reported rather than raised
    shortfall = {}
    if use_fairness:
        need = required_units(p.fairness.alpha.value, q)
        for j in range(n_wards):
            within = [counts[k] for k in range(n_bases) if reach[j, k]]
            gap = model.new_int_var(0, need, f"gap_{j}")
            model.add(sum(within) + gap >= need)
            shortfall[j] = gap

    objective = [
        round(WEIGHT_SCALE * demand[j] * (1 - q) * q ** (level - 1)) * z[(j, level)]
        for j in range(n_wards)
        for level in range(1, max_cover + 1)
    ]
    objective += [-SHORTFALL_PENALTY * WEIGHT_SCALE * gap for gap in shortfall.values()]
    model.maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.dispatch.solver.time_limit_s
    solver.parameters.num_workers = cfg.dispatch.solver.workers
    status = solver.solve(model)
    names = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}
    if status not in names:
        return PrepositionResult(
            placement=list(unit_bases),
            expected_covered_demand=expected_covered(list(unit_bases), demand, covers, q),
            moves=0,
            move_seconds=0.0,
            status="NO_SOLUTION_KEPT_CURRENT",
            solve_s=time.perf_counter() - started,
        )
    placement = [next(k for k in range(n_bases) if solver.value(y[(u, k)])) for u in range(n_units)]
    moved = [u for u in range(n_units) if placement[u] != unit_bases[u]]
    uncovered = [
        (ward_ids[j] if ward_ids else str(j))
        for j, gap in shortfall.items()
        if solver.value(gap) > 0
    ]
    return PrepositionResult(
        placement=placement,
        expected_covered_demand=expected_covered(placement, demand, covers, q),
        moves=len(moved),
        move_seconds=float(sum(move_seconds[u, placement[u]] for u in moved)),
        uncovered_wards=sorted(uncovered),
        status=names[status],
        solve_s=time.perf_counter() - started,
    )


def price_of_fairness(
    unit_bases: list[int],
    move_seconds: np.ndarray,
    demand: np.ndarray,
    covers: np.ndarray,
    reach: np.ndarray,
    cfg: OptimizationConfig,
    capacity: list[int] | None = None,
    ward_ids: list[str] | None = None,
) -> dict[str, object]:
    """RQ4: solve with and without (P5) and report the relative loss in expected coverage."""
    free = solve(unit_bases, move_seconds, demand, covers, reach, cfg, capacity, ward_ids, False)
    fair = solve(unit_bases, move_seconds, demand, covers, reach, cfg, capacity, ward_ids, True)
    loss = (
        (free.expected_covered_demand - fair.expected_covered_demand) / free.expected_covered_demand
        if free.expected_covered_demand > 0
        else float("nan")
    )
    return {
        "unconstrained_expected_covered": free.expected_covered_demand,
        "fair_expected_covered": fair.expected_covered_demand,
        "price_of_fairness": loss,
        "wards_left_short": fair.uncovered_wards,
        "unconstrained_moves": free.moves,
        "fair_moves": fair.moves,
        "status": {"unconstrained": free.status, "fair": fair.status},
    }
