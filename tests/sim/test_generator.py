"""Synthetic incident generator on the toy network."""

import numpy as np
import pytest

from resqcity.common.requirements import load_requirements_config
from resqcity.geo.scenario import ScenarioConfig, ScenarioState, load_scenario
from resqcity.sim.generator import (
    IncidentGenerator,
    IncidentsConfig,
    hotspot_weights,
    load_incidents_config,
    to_frame,
)
from tests.conftest import node_id

INCIDENTS = load_incidents_config()
REQUIREMENTS = load_requirements_config()


def _generator(network, scenario: ScenarioConfig) -> IncidentGenerator:
    return IncidentGenerator(network, ScenarioState(scenario, network), INCIDENTS, REQUIREMENTS)


@pytest.fixture(scope="module")
def normal_gen(toy_network):
    return _generator(toy_network, load_scenario("normal"))


def test_config_is_synthetic_and_flagged():
    assert INCIDENTS.synthetic
    assert INCIDENTS.base_rate_per_day.source == "UNVERIFIED"
    assert INCIDENTS.hour_factors.mean() == pytest.approx(1.0)


def test_hotspot_weights(toy_network):
    w = hotspot_weights(toy_network, INCIDENTS.hotspot)
    assert w.sum() == pytest.approx(1.0)
    outside = ~toy_network.nodes["in_city"].to_numpy(bool)
    assert np.all(w[outside] == 0)
    centre = toy_network.index_of(node_id(2, 2))
    corner = toy_network.index_of(node_id(0, 0))
    assert w[centre] > w[corner]


def test_reproducible_for_a_seed(normal_gen):
    a = normal_gen.generate(seed_base=42, replication=0)
    b = normal_gen.generate(seed_base=42, replication=0)
    assert a == b
    c = normal_gen.generate(seed_base=42, replication=1)
    assert [i.onset_ts for i in a] != [i.onset_ts for i in c]


def test_incident_fields(normal_gen):
    incidents = normal_gen.generate(seed_base=1, replication=0)
    assert len(incidents) > 0
    assert len({i.id for i in incidents}) == len(incidents)
    cfg = normal_gen.scenario.cfg
    in_city = set(normal_gen.network.nodes.loc[normal_gen.network.nodes["in_city"], "node_id"])
    for inc in incidents:
        assert inc.synthetic
        assert cfg.start <= inc.onset_ts < cfg.end
        assert inc.node in in_city
        assert inc.requirements.total >= 1
    onsets = [i.onset_ts for i in incidents]
    assert onsets == sorted(onsets)
    frame = to_frame(incidents)
    assert frame["synthetic"].all() and len(frame) == len(incidents)


def test_counts_match_intensity(normal_gen):
    """Mean total count over replications matches the integrated NHPP intensity."""
    expected = normal_gen.ward_hour_rates()["expected"].sum()
    assert expected == pytest.approx(INCIDENTS.base_rate_per_day.value * 7, rel=0.02)
    reps = 40
    counts = [len(normal_gen.generate(seed_base=7, replication=r)) for r in range(reps)]
    se = np.sqrt(expected / reps)
    assert abs(np.mean(counts) - expected) < 4 * se


def test_ward_shares_match_intensity(normal_gen):
    rates = normal_gen.ward_hour_rates().groupby("ward_id")["expected"].sum()
    share = rates / rates.sum()
    incidents = [i for r in range(30) for i in normal_gen.generate(seed_base=3, replication=r)]
    observed = to_frame(incidents)["ward_id"].value_counts(normalize=True)
    for ward, p in share.items():
        assert observed.get(ward, 0.0) == pytest.approx(p, abs=0.05)


def test_rain_increases_rate(toy_network):
    gen = _generator(toy_network, load_scenario("monsoon"))
    rates = gen.ward_hour_rates().groupby("hour_start")["expected"].sum()
    start = gen.scenario.cfg.start
    dry_ish = rates[start + np.timedelta64(3, "D") + np.timedelta64(3, "h")]  # 03:00, light rain
    wet = rates[start + np.timedelta64(3, "D") + np.timedelta64(16, "h")]  # 16:00, cloudburst
    hour = INCIDENTS.hour_factors
    assert wet / hour[16] > dry_ish / hour[3]


def test_festival_raises_rate_near_route_only_in_window(toy_network):
    cfg = load_scenario("ganeshotsav")
    gen = _generator(toy_network, cfg)
    near = gen.scenario.near_route_nodes
    assert near.any()
    quiet = cfg.start.replace(hour=12)
    busy = cfg.start.replace(hour=19)
    with np.errstate(invalid="ignore"):  # 0/0 on out-of-city nodes (excluded below)
        ratio = gen.node_rates(busy) / gen.node_rates(quiet)
    base_ratio = gen.global_multiplier(busy) / gen.global_multiplier(quiet)
    share = gen.node_share > 0
    np.testing.assert_allclose(ratio[near & share], base_ratio * 1.8)
    np.testing.assert_allclose(ratio[~near & share], base_ratio)


def test_counts_are_poisson_dispersed(normal_gen):
    """Variance of the weekly count across replications is close to its mean (Poisson)."""
    counts = np.array([len(normal_gen.generate(seed_base=11, replication=r)) for r in range(60)])
    dispersion = counts.var(ddof=1) / counts.mean()
    assert 0.6 < dispersion < 1.5


def test_crn_feature_change_keeps_arrivals(toy_network):
    """Common random numbers: a feature parameter changes marks only, never onsets or places."""
    data = INCIDENTS.model_dump(mode="json")
    data["features"]["accident"]["value"]["p_person_down"] = 0.9
    changed = IncidentsConfig.model_validate(data)
    scenario = ScenarioState(load_scenario("normal"), toy_network)
    a = IncidentGenerator(toy_network, scenario, INCIDENTS, REQUIREMENTS).generate(5, 0)
    b = IncidentGenerator(toy_network, scenario, changed, REQUIREMENTS).generate(5, 0)
    assert [(i.onset_ts, i.node, i.type) for i in a] == [(i.onset_ts, i.node, i.type) for i in b]
    assert sum(i.severity_features.person_down for i in b) > sum(
        i.severity_features.person_down for i in a
    )
