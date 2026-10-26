"""C3 pre-positioning (MEXCLP) against exhaustive search (docs/models/preposition.md)."""

import itertools

import numpy as np
import pytest

from resqcity.optimization.config import OptimizationConfig, load_optimization_config
from resqcity.optimization.preposition import (
    expected_covered,
    price_of_fairness,
    required_units,
    solve,
)

CFG = load_optimization_config()


def _cfg(**updates: object) -> OptimizationConfig:
    data = CFG.model_dump(mode="json")
    pre = data["preposition"]
    for key, value in updates.items():
        if key in ("alpha", "t_ward_s"):
            pre["fairness"][key]["value"] = value
        elif key == "enabled":
            pre["fairness"][key] = value
        elif key in ("max_moves", "max_total_s"):
            pre["relocation"][key]["value"] = value
        else:
            pre[key]["value"] = value
    return OptimizationConfig.model_validate(data)


def brute_force(
    unit_bases: list[int],
    move_seconds: np.ndarray,
    demand: np.ndarray,
    covers: np.ndarray,
    q: float,
    max_moves: int,
    max_total_s: float,
    caps: list[int],
) -> float:
    """Best expected covered demand over every feasible placement."""
    n_units, n_bases = move_seconds.shape
    best = -1.0
    for placement in itertools.product(range(n_bases), repeat=n_units):
        moved = [u for u in range(n_units) if placement[u] != unit_bases[u]]
        if len(moved) > max_moves:
            continue
        if sum(move_seconds[u, placement[u]] for u in moved) > max_total_s:
            continue
        counts = [placement.count(k) for k in range(n_bases)]
        if any(c > caps[k] for k, c in enumerate(counts)):
            continue
        best = max(best, expected_covered(list(placement), demand, covers, q))
    return best


def test_required_units_matches_the_probability_it_encodes():
    for alpha, q in ((0.8, 0.3), (0.9, 0.5), (0.5, 0.2)):
        n = required_units(alpha, q)
        assert 1 - q**n >= alpha - 1e-12
        assert 1 - q ** (n - 1) < alpha or n == 1  # and it is the smallest such n
    assert required_units(0.0, 0.3) == 0
    with pytest.raises(ValueError, match="infinitely"):
        required_units(1.0, 0.3)


def test_matches_brute_force_on_a_tiny_instance():
    unit_bases = [0, 0, 1]
    move_seconds = np.array([[0.0, 120.0, 300.0], [0.0, 150.0, 280.0], [200.0, 0.0, 160.0]])
    demand = np.array([10.0, 4.0, 1.0])
    covers = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
    cfg = _cfg(enabled=False, max_moves=3, max_total_s=1000.0, busy_fraction=0.3)
    result = solve(unit_bases, move_seconds, demand, covers, covers, cfg, capacity=[3, 3, 3])
    want = brute_force(unit_bases, move_seconds, demand, covers, 0.3, 3, 1000.0, [3, 3, 3])
    assert result.expected_covered_demand == pytest.approx(want, rel=1e-6)


def test_the_relocation_budget_binds():
    unit_bases = [0, 0]
    move_seconds = np.array([[0.0, 100.0], [0.0, 100.0]])
    demand = np.array([1.0, 50.0])
    covers = np.array([[1, 0], [0, 1]], dtype=bool)  # all the demand is at base 1
    args = (unit_bases, move_seconds, demand, covers, covers)
    loose = solve(*args, _cfg(enabled=False, max_moves=2, max_total_s=1000.0), capacity=[2, 2])
    tight = solve(*args, _cfg(enabled=False, max_moves=0, max_total_s=1000.0), capacity=[2, 2])
    assert loose.moves == 2 and tight.moves == 0
    assert tight.expected_covered_demand < loose.expected_covered_demand
    budget = solve(*args, _cfg(enabled=False, max_moves=2, max_total_s=100.0), capacity=[2, 2])
    assert budget.move_seconds <= 100.0


def test_second_cover_is_worth_less_than_first_cover():
    """The MEXCLP property: with q busy, a second unit still helps, but less."""
    demand, q = np.array([1.0]), 0.3
    covers = np.ones((1, 1), dtype=bool)
    one = expected_covered([0], demand, covers, q)
    two = expected_covered([0, 0], demand, covers, q)
    three = expected_covered([0, 0, 0], demand, covers, q)
    assert one == pytest.approx(0.7)
    assert two - one == pytest.approx(q * (1 - q))
    assert 0 < three - two < two - one


def test_fairness_forces_cover_into_a_low_demand_ward():
    unit_bases = [0, 0]
    move_seconds = np.array([[0.0, 60.0], [0.0, 60.0]])
    demand = np.array([100.0, 1.0])  # ward 1 barely matters to the objective
    covers = np.array([[1, 0], [0, 1]], dtype=bool)
    free = solve(unit_bases, move_seconds, demand, covers, covers,
                 _cfg(enabled=False, max_moves=2, max_total_s=500.0, busy_fraction=0.5),
                 capacity=[2, 2])  # fmt: skip
    fair = solve(unit_bases, move_seconds, demand, covers, covers,
                 _cfg(enabled=True, alpha=0.5, max_moves=2, max_total_s=500.0, busy_fraction=0.5),
                 capacity=[2, 2])  # fmt: skip
    assert free.placement == [0, 0]  # everything stays where the demand is
    assert 1 in fair.placement  # fairness drags a unit to the quiet ward
    assert fair.expected_covered_demand < free.expected_covered_demand


def test_price_of_fairness_is_reported_and_non_negative():
    unit_bases = [0, 0]
    move_seconds = np.array([[0.0, 60.0], [0.0, 60.0]])
    demand = np.array([100.0, 1.0])
    covers = np.array([[1, 0], [0, 1]], dtype=bool)
    cfg = _cfg(alpha=0.5, max_moves=2, max_total_s=500.0, busy_fraction=0.5)
    report = price_of_fairness(unit_bases, move_seconds, demand, covers, covers, cfg,
                               capacity=[2, 2], ward_ids=["W01", "W02"])  # fmt: skip
    assert report["price_of_fairness"] > 0
    assert report["fair_expected_covered"] <= report["unconstrained_expected_covered"]
    assert report["wards_left_short"] == []


def test_an_impossible_fairness_requirement_is_reported_not_raised():
    """A ward no base can reach in time must surface as a finding, not an exception."""
    unit_bases = [0]
    move_seconds = np.array([[0.0, 60.0]])
    demand = np.array([5.0, 5.0])
    covers = np.array([[1, 0], [0, 1]], dtype=bool)
    reach = np.array([[1, 0], [0, 0]], dtype=bool)  # ward 1 is out of reach of every base
    # alpha 0.5 with q 0.3 needs one unit, which W01 can have and W02 cannot
    cfg = _cfg(enabled=True, alpha=0.5, busy_fraction=0.3, max_moves=1, max_total_s=500.0)
    result = solve(unit_bases, move_seconds, demand, covers, reach, cfg,
                   capacity=[1, 1], ward_ids=["W01", "W02"])  # fmt: skip
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.uncovered_wards == ["W02"]


def test_more_units_never_reduce_expected_coverage():
    demand = np.array([5.0, 5.0])
    covers = np.array([[1, 0], [0, 1]], dtype=bool)
    cfg = _cfg(enabled=False, max_moves=4, max_total_s=1000.0)
    small = solve([0], np.array([[0.0, 60.0]]), demand, covers, covers, cfg, capacity=[2, 2])
    big = solve([0, 1], np.array([[0.0, 60.0], [60.0, 0.0]]), demand, covers, covers, cfg,
                capacity=[2, 2])  # fmt: skip
    assert big.expected_covered_demand >= small.expected_covered_demand - 1e-9


def test_placement_respects_capacity():
    unit_bases = [0, 0, 0]
    move_seconds = np.zeros((3, 2))
    demand = np.array([10.0, 0.0])
    covers = np.array([[1, 0], [0, 1]], dtype=bool)
    cfg = _cfg(enabled=False, max_moves=3, max_total_s=1000.0)
    result = solve(unit_bases, move_seconds, demand, covers, covers, cfg, capacity=[1, 2])
    assert result.placement.count(0) <= 1
    assert result.placement.count(1) <= 2


def test_cover_levels_reach_what_several_bases_can_stack():
    """The OR review found the objective truncated at the largest SINGLE base capacity, so a
    ward covered by two bases was under-valued and the solver moved units away from it."""
    demand = np.array([10.0, 1.0])
    covers = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)  # bases 0 and 1 both cover ward 0
    move_seconds = np.zeros((4, 3))
    cfg = _cfg(enabled=False, busy_fraction=0.5, max_moves=4, max_total_s=1e6)
    result = solve([2, 2, 0, 0], move_seconds, demand, covers, covers, cfg, capacity=[2, 2, 2])
    want = brute_force([2, 2, 0, 0], move_seconds, demand, covers, 0.5, 4, 1e6, [2, 2, 2])
    assert result.expected_covered_demand == pytest.approx(want, rel=1e-6)
    assert sorted(result.placement) == [0, 0, 1, 1]  # both bases serving the heavy ward


def test_required_units_at_exact_probability_boundaries():
    """alpha 0.9 with q 0.1 is satisfied by one unit exactly; floating point used to ask for two."""
    assert required_units(0.9, 0.1) == 1
    assert required_units(0.75, 0.5) == 2
    assert required_units(0.875, 0.5) == 3
    for alpha, q in ((0.9, 0.1), (0.75, 0.5), (0.875, 0.5), (0.99, 0.1)):
        n = required_units(alpha, q)
        assert 1 - q**n >= alpha - 1e-12
        assert n == 1 or 1 - q ** (n - 1) < alpha - 1e-12
