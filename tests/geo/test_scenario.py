"""Scenario effects: rain, flooding (monotone in rain) and festival closures."""

from datetime import UTC, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from resqcity.geo.scenario import ScenarioConfig, ScenarioState, available_scenarios, load_scenario
from tests.conftest import LAT0, LON0, STEP, at, node_id


def _scenario(**overrides) -> ScenarioConfig:
    data = {
        "name": "toy",
        "description": "toy scenario",
        "synthetic": True,
        "start": at(0, 0),
        "days": 2,
        "incident_rate_multiplier": 1.0,
        "rain": None,
        "flooding": None,
        "festival": None,
    }
    data.update(overrides)
    return ScenarioConfig.model_validate(data)


def _rain(peak: float) -> dict:
    hourly = [0.0] * 24
    hourly[12] = hourly[13] = hourly[14] = peak
    return {"hourly_mm_h": hourly, "day_factors": [1.0]}


FLOOD = {
    "buffer_m": 150,
    "window_h": 3,
    "vulnerability_seed": 1,
    "stages": {
        "source": "UNVERIFIED",
        "value": [
            {
                "min_accum_mm": 10,
                "impassable_fraction": 0.0,
                "slowed_fraction": 0.5,
                "slow_multiplier": 0.5,
            },
            {
                "min_accum_mm": 20,
                "impassable_fraction": 1.0,
                "slowed_fraction": 0.0,
                "slow_multiplier": 0.5,
            },
        ],
    },
}

FESTIVAL = {
    "name": "Toy festival",
    "routes": [{"name": "core", "street_names": ["laxmi road"], "buffer_m": 25}],
    "windows": [
        {"days": [0], "start_hour": 18, "end_hour": 22, "effect": "slowed"},
        {"days": [1], "start_hour": 9, "end_hour": 12, "effect": "closed"},
    ],
    "slowed_multiplier": {"value": 0.3, "source": "UNVERIFIED"},
    "influence_radius_m": 250,
    "influence_speed_multiplier": {"value": 0.6, "source": "UNVERIFIED"},
    "incident_multiplier_near": {"value": 2.0, "source": "UNVERIFIED"},
}


@pytest.mark.parametrize("name", ["normal", "monsoon", "ganeshotsav"])
def test_project_scenarios_validate(name):
    cfg = load_scenario(name)
    assert cfg.synthetic is True
    assert cfg.name == name
    assert cfg.start.tzinfo is not None


def test_available_scenarios():
    assert {"normal", "monsoon", "ganeshotsav"} <= set(available_scenarios())


def test_monsoon_rain_profile(toy_network):
    cfg = load_scenario("monsoon")
    state = ScenarioState(cfg, toy_network)
    t = cfg.start + timedelta(days=3, hours=16)  # cloudburst day, 16:00
    assert state.rain_mm_h(t) == pytest.approx(1.5 * 8.0)


def test_no_rain_means_no_effect(toy_network):
    state = ScenarioState(_scenario(), toy_network)
    assert state.rain_mm_h(at(0, 13)) == 0.0
    assert np.all(state.edge_factors(at(0, 13)) == 1.0)
    np.testing.assert_allclose(
        toy_network.edge_times(at(0, 13), state), toy_network.edge_times(at(0, 13))
    )


def test_flood_stages_follow_accumulated_rain(toy_network):
    state = ScenarioState(_scenario(rain=_rain(5.0), flooding=FLOOD), toy_network)
    assert state.flood_stage(at(0, 11)) == -1
    assert state.flood_stage(at(0, 13)) == 0  # 10 mm over the last 3 h
    assert state.rain_accum_mm(at(0, 14), 3) == pytest.approx(15.0)
    assert state.flood_stage(at(0, 14)) == 0  # stage 1 needs 20 mm
    assert state.flood_stage(at(0, 18)) == -1  # rain stopped; window has passed


def test_only_flood_prone_edges_are_affected(toy_network):
    state = ScenarioState(_scenario(rain=_rain(10.0), flooding=FLOOD), toy_network)
    t = at(0, 14)
    assert state.flood_stage(t) == 1
    factors = state.edge_factors(t)
    assert np.all(factors[~state.flood_prone] == 1.0)
    assert np.all(factors[state.flood_prone] == 0.0)
    assert state.flood_prone.any() and not state.flood_prone.all()


def test_flooded_nodes_become_unreachable(toy_network):
    state = ScenarioState(_scenario(rain=_rain(10.0), flooding=FLOOD), toy_network)
    t = at(0, 14)
    before = toy_network.shortest_times([node_id(3, 2)], at(0, 11), state, targets=[node_id(0, 2)])
    after = toy_network.shortest_times([node_id(3, 2)], t, state, targets=[node_id(0, 2)])
    assert np.isfinite(before[0, 0])
    assert np.isinf(after[0, 0])


@settings(max_examples=50, deadline=None)
@given(low=st.floats(0, 30), extra=st.floats(0, 30))
def test_flooding_is_monotone_in_rain(toy_network, low, extra):
    """More rain never reopens a road or speeds one up."""
    a = ScenarioState(_scenario(rain=_rain(low / 3), flooding=FLOOD), toy_network)
    b = ScenarioState(_scenario(rain=_rain((low + extra) / 3), flooding=FLOOD), toy_network)
    t = at(0, 14)
    assert np.all(b.edge_factors(t) <= a.edge_factors(t))


def test_non_monotone_stages_rejected():
    bad = {**FLOOD, "stages": {"source": "x", "value": list(reversed(FLOOD["stages"]["value"]))}}
    with pytest.raises(ValidationError, match="monotone"):
        _scenario(rain=_rain(1.0), flooding=bad)


def test_flooding_is_symmetric_per_road(toy_network):
    """Both directions of a two-way road always share flood state (vulnerability per road)."""
    state = ScenarioState(_scenario(rain=_rain(10.0), flooding=FLOOD), toy_network)
    pair_of_edge, _ = toy_network.undirected_pairs()
    for t in (at(0, 12), at(0, 13), at(0, 14)):
        factors = state.edge_factors(t)
        for pair in np.unique(pair_of_edge):
            assert len(set(factors[pair_of_edge == pair])) == 1


def test_route_match_counts_recorded(toy_network):
    festival = {
        **FESTIVAL,
        "routes": [
            {"name": "core", "street_names": ["laxmi road", "no such road"], "buffer_m": 25}
        ],
    }
    state = ScenarioState(_scenario(festival=festival), toy_network)
    assert state.route_match_counts == {"core: laxmi road": 8, "core: no such road": 0}


def test_flooding_requires_rain():
    with pytest.raises(ValidationError, match="rain"):
        _scenario(flooding=FLOOD)


def test_festival_closure_only_in_window(toy_network):
    state = ScenarioState(_scenario(festival=FESTIVAL), toy_network)
    route = state.route_edges
    assert route.sum() == 8  # 4 segments of Laxmi Road x 2 directions
    assert state.festival_effect(at(0, 12)) is None
    assert np.all(state.edge_factors(at(0, 12)) == 1.0)

    slowed = state.edge_factors(at(0, 19))
    assert np.all(slowed[route] == pytest.approx(0.3))
    assert np.all(slowed[state.influence_edges] == pytest.approx(0.6))

    closed = state.edge_factors(at(1, 10))
    assert np.all(closed[route] == 0.0)
    assert state.festival_effect(at(1, 12)) is None


def test_route_names_can_be_restricted_to_an_area(toy_network):
    route = {"name": "core", "street_names": ["laxmi road"], "buffer_m": 25}
    near = {**route, "center": [LAT0 + 2 * STEP, LON0], "max_distance_m": 300}  # west end only
    far = {**route, "center": [LAT0 + 1.0, LON0], "max_distance_m": 300}  # ~110 km away
    n_near = ScenarioState(_scenario(festival={**FESTIVAL, "routes": [near]}), toy_network)
    n_far = ScenarioState(_scenario(festival={**FESTIVAL, "routes": [far]}), toy_network)
    assert 0 < n_near.route_edges.sum() < 8
    assert n_far.route_edges.sum() == 0


def test_route_center_needs_radius():
    bad = {
        **FESTIVAL,
        "routes": [{"name": "x", "street_names": ["a"], "buffer_m": 25, "center": [18.5, 73.8]}],
    }
    with pytest.raises(ValidationError, match="center"):
        _scenario(festival=bad)


def test_festival_incident_multiplier(toy_network):
    state = ScenarioState(_scenario(festival=FESTIVAL), toy_network)
    near = state.near_route_nodes
    assert near.any() and not near.all()
    assert np.all(state.incident_node_multipliers(at(0, 12)) == 1.0)
    during = state.incident_node_multipliers(at(0, 19))
    assert np.all(during[near] == 2.0) and np.all(during[~near] == 1.0)


def test_state_key_changes_with_conditions(toy_network):
    state = ScenarioState(
        _scenario(rain=_rain(10.0), flooding=FLOOD, festival=FESTIVAL), toy_network
    )
    keys = {state.state_key(at(0, h)) for h in (8, 12, 14, 19)}  # dry, stage 0, stage 1, festival
    assert len(keys) == 4


def test_day_offset_uses_local_midnight(toy_network):
    state = ScenarioState(_scenario(), toy_network)
    assert state.day_offset(at(0, 23, 59)) == 0
    assert state.day_offset(at(1, 0, 0)) == 1
    assert state.day_offset(at(1, 0, 30).astimezone(UTC)) == 1  # 19:00 UTC the day before


def test_window_must_end_after_start():
    bad = {
        **FESTIVAL,
        "windows": [{"days": [0], "start_hour": 22, "end_hour": 2, "effect": "closed"}],
    }
    with pytest.raises(ValidationError):
        _scenario(festival=bad)
