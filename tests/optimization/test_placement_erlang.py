"""MCLP camera placement and Erlang-C fleet sizing against brute force (phase 6.5, 6.6)."""

import itertools

import numpy as np
import pytest

from resqcity.optimization.config import load_optimization_config
from resqcity.optimization.placement import (
    covered_weight,
    erlang_c,
    greedy_placement,
    queue_metrics,
    servers_for_target,
    solve_mclp,
)

CFG = load_optimization_config()


def brute_force_mclp(weights: np.ndarray, covers: np.ndarray, k: int) -> float:
    sites = range(covers.shape[1])
    return max(
        (
            covered_weight(list(combo), weights, covers)
            for combo in itertools.combinations(sites, k)
        ),
        default=0.0,
    )


# -- MCLP ---------------------------------------------------------------------------------------


def test_mclp_matches_brute_force():
    rng = np.random.default_rng(0)
    weights = rng.random(12) * 10
    covers = rng.random((12, 6)) < 0.35
    for k in (1, 2, 3):
        result = solve_mclp(weights, covers, k, CFG)
        assert result.status in {"OPTIMAL", "FEASIBLE"}
        assert len(result.chosen) <= k
        assert result.covered_weight == pytest.approx(
            brute_force_mclp(weights, covers, k), rel=1e-6
        )


def test_mclp_prefers_the_site_covering_the_most_weight():
    weights = np.array([10.0, 1.0, 1.0])
    covers = np.array([[1, 0], [0, 1], [0, 1]], dtype=bool)
    assert solve_mclp(weights, covers, 1, CFG).chosen == [0]  # 10 beats 2


def test_mclp_ignores_points_no_site_can_cover():
    weights = np.array([5.0, 100.0])
    covers = np.array([[1], [0]], dtype=bool)  # the heavy point is unreachable
    result = solve_mclp(weights, covers, 1, CFG)
    assert result.covered_weight == pytest.approx(5.0)


def test_greedy_is_within_the_submodular_bound_of_the_optimum():
    """Maximum coverage is submodular, so greedy is guaranteed >= (1 - 1/e) of the optimum."""
    rng = np.random.default_rng(3)
    bound = 1 - 1 / np.e
    for _ in range(8):
        weights = rng.random(14) * 5
        covers = rng.random((14, 7)) < 0.3
        k = 3
        chosen, _ = greedy_placement(weights, covers, k)
        got = covered_weight(chosen, weights, covers)
        best = brute_force_mclp(weights, covers, k)
        assert got >= bound * best - 1e-9
        assert got <= best + 1e-9


def test_marginal_gains_are_non_increasing():
    """The diminishing-returns curve the phase reports for K = 0..k_max."""
    rng = np.random.default_rng(1)
    weights = rng.random(30) * 4
    covers = rng.random((30, 10)) < 0.25
    _, gains = greedy_placement(weights, covers, 10)
    assert gains == sorted(gains, reverse=True)
    assert all(g >= 0 for g in gains)


def test_greedy_stops_when_nothing_is_left_to_cover():
    weights = np.array([1.0, 1.0])
    covers = np.array([[1, 0], [1, 0]], dtype=bool)  # site 1 covers nothing
    chosen, gains = greedy_placement(weights, covers, 5)
    assert covered_weight(chosen, weights, covers) == pytest.approx(2.0)
    assert sum(gains) == pytest.approx(2.0)


# -- Erlang-C -----------------------------------------------------------------------------------


def test_erlang_c_matches_textbook_values():
    """Standard worked examples: a = 1 erlang with c = 2, and the classic a = 8, c = 10."""
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3, rel=1e-9)
    assert erlang_c(10, 8.0) == pytest.approx(0.4092, abs=5e-4)
    assert erlang_c(1, 0.5) == pytest.approx(0.5, rel=1e-9)  # M/M/1: P(wait) = rho


def test_erlang_c_matches_the_direct_formula():
    """Cross-check the stable recurrence against the textbook ratio of factorial terms."""
    import math

    for servers, load in ((3, 2.0), (5, 3.5), (8, 6.0)):
        top = load**servers / math.factorial(servers) * servers / (servers - load)
        bottom = sum(load**n / math.factorial(n) for n in range(servers)) + top
        assert erlang_c(servers, load) == pytest.approx(top / bottom, rel=1e-9)


def test_an_overloaded_system_always_waits():
    assert erlang_c(2, 5.0) == 1.0
    metrics = queue_metrics(arrival_rate_per_s=1 / 100, service_time_s=1000.0, servers=2)
    assert not metrics["stable"]
    assert metrics["expected_wait_s"] == float("inf")


def test_more_servers_shorten_the_wait():
    waits = [queue_metrics(1 / 600, 2700.0, c)["expected_wait_s"] for c in (6, 8, 12, 20)]
    assert all(np.isfinite(waits))
    assert waits == sorted(waits, reverse=True)


def test_servers_for_target_is_the_smallest_that_works():
    rate, service, target = 1 / 600, 2700.0, 60.0
    n = servers_for_target(rate, service, target)
    assert queue_metrics(rate, service, n)["expected_wait_s"] <= target
    assert queue_metrics(rate, service, n - 1)["expected_wait_s"] > target


def test_erlang_rejects_a_fleet_of_zero():
    with pytest.raises(ValueError, match="at least one server"):
        erlang_c(0, 1.0)
