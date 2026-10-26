"""Camera placement (MCLP) and Erlang-C fleet sizing (phase 6.5, 6.6).

**MCLP.** Choose K junctions to maximise risk-weighted coverage of incident locations. Coverage
is solved exactly by CP-SAT and the marginal-gain curve for K = 0..k_max comes from the greedy
algorithm, which for a maximum-coverage objective is submodular and therefore within
(1 - 1/e) ~ 63 % of the optimum; the test checks greedy against exhaustive search on small
instances and records where they differ.

**Honest framing (D-029).** Coverage is an UPPER BOUND on what a camera is worth here. A covered
incident is still detected only with the measured effective probability of 0.31, because the
detector's own false alarms silence it for 48 % of the time; and every camera added also adds
false alarms at ~29 per camera-hour. The marginal-gain curve therefore answers "where would
cameras see the most incidents", not "how many crashes would be detected".
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from resqcity.optimization.config import OptimizationConfig


@dataclass
class PlacementResult:
    chosen: list[int]
    covered_weight: float
    status: str = ""
    solve_s: float = 0.0
    marginal_gain: list[float] = field(default_factory=list)


def covered_weight(chosen: list[int], weights: np.ndarray, covers: np.ndarray) -> float:
    """Total weight of demand points covered by at least one chosen site."""
    if not chosen:
        return 0.0
    return float(weights[covers[:, list(chosen)].any(axis=1)].sum())


def greedy_placement(
    weights: np.ndarray, covers: np.ndarray, k_max: int
) -> tuple[list[int], list[float]]:
    """Add the site covering the most uncovered weight, k_max times.

    Maximum coverage is submodular, so this is within (1 - 1/e) of the optimum; the curve it
    traces is what the marginal-gain plot needs.
    """
    n_sites = covers.shape[1]
    uncovered = np.ones(len(weights), dtype=bool)
    chosen, gains = [], []
    for _ in range(min(k_max, n_sites)):
        best_site, best_gain = -1, 0.0
        for site in range(n_sites):
            if site in chosen:
                continue
            gain = float(weights[uncovered & covers[:, site]].sum())
            if gain > best_gain or (best_site < 0 and gain >= best_gain):
                best_site, best_gain = site, gain
        if best_site < 0:
            break
        chosen.append(best_site)
        gains.append(best_gain)
        uncovered &= ~covers[:, best_site]
    return chosen, gains


def solve_mclp(
    weights: np.ndarray, covers: np.ndarray, k: int, cfg: OptimizationConfig
) -> PlacementResult:
    """Exact MCLP: max sum_i w_i y_i  s.t.  y_i <= sum_{s covering i} x_s,  sum_s x_s <= k."""
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    n_points, n_sites = covers.shape
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{s}") for s in range(n_sites)]
    y = [model.new_bool_var(f"y_{i}") for i in range(n_points)]
    for i in range(n_points):
        options = [x[s] for s in range(n_sites) if covers[i, s]]
        if options:
            model.add(y[i] <= sum(options))
        else:
            model.add(y[i] == 0)
    model.add(sum(x) <= k)
    scale = 1000
    model.maximize(sum(round(float(weights[i]) * scale) * y[i] for i in range(n_points)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.dispatch.solver.time_limit_s
    solver.parameters.num_workers = cfg.dispatch.solver.workers
    status = solver.solve(model)
    names = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}
    if status not in names:
        chosen, gains = greedy_placement(weights, covers, k)
        return PlacementResult(chosen, covered_weight(chosen, weights, covers),
                               "GREEDY_FALLBACK", time.perf_counter() - started, gains)  # fmt: skip
    chosen = [s for s in range(n_sites) if solver.value(x[s])]
    return PlacementResult(
        chosen=chosen,
        covered_weight=covered_weight(chosen, weights, covers),
        status=names[status],
        solve_s=time.perf_counter() - started,
    )


# -- Erlang-C (phase 6.6) ------------------------------------------------------------------------


def erlang_c(servers: int, offered_load: float) -> float:
    """P(wait) in M/M/c with ``servers`` servers and offered load a = lambda/mu (erlangs).

    Computed with the numerically stable recurrence for the Erlang-B blocking probability,
    B(0) = 1 and B(n) = a B(n-1) / (n + a B(n-1)), then C = B / (1 - rho(1 - B)).
    """
    if servers <= 0:
        raise ValueError("need at least one server")
    if offered_load <= 0:
        return 0.0
    rho = offered_load / servers
    if rho >= 1:
        return 1.0  # unstable: the queue grows without bound
    blocking = 1.0
    for n in range(1, servers + 1):
        blocking = offered_load * blocking / (n + offered_load * blocking)
    return blocking / (1.0 - rho * (1.0 - blocking))


def queue_metrics(arrival_rate_per_s: float, service_time_s: float, servers: int) -> dict:
    """Utilisation, P(wait) and expected wait for one agency in one zone."""
    offered = arrival_rate_per_s * service_time_s
    rho = offered / servers if servers else float("inf")
    p_wait = erlang_c(servers, offered) if servers else 1.0
    wait = (
        p_wait * service_time_s / (servers * (1.0 - rho)) if servers and rho < 1 else float("inf")
    )
    return {
        "servers": servers,
        "offered_load_erlangs": offered,
        "utilisation": rho,
        "p_wait": p_wait,
        "expected_wait_s": wait,
        "stable": bool(servers and rho < 1),
    }


def servers_for_target(
    arrival_rate_per_s: float, service_time_s: float, target_wait_s: float
) -> int:
    """Smallest server count whose expected wait is at or below the target."""
    offered = arrival_rate_per_s * service_time_s
    n = max(1, math.ceil(offered))
    while n < 10_000:
        if queue_metrics(arrival_rate_per_s, service_time_s, n)["expected_wait_s"] <= target_wait_s:
            return n
        n += 1
    raise RuntimeError("no stable server count reaches the target wait")
