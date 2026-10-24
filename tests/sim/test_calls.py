"""Synthetic 112 call process on the toy network (phase 4)."""

import math
from datetime import timedelta

import numpy as np
import pytest

from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import EventType, GeoPoint, Language, SeverityFeatures
from resqcity.fusion.config import CallsConfig, load_calls_config, load_templates
from resqcity.sim.calls import CallSimulator
from resqcity.sim.groundtruth import GroundTruthIncident
from tests.conftest import at, node_id

CALLS = load_calls_config()
TEMPLATES = load_templates()
REQUIREMENTS = load_requirements_config()
MIX = {EventType.ACCIDENT: 0.5, EventType.FIRE_SMOKE: 0.5}


def _with(cfg: CallsConfig, path: str, value: object) -> CallsConfig:
    data = cfg.model_dump(mode="json")
    node = data
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    node[last] = value
    return CallsConfig.model_validate(data)


def _incidents(n: int, severity_class: str = "serious") -> list[GroundTruthIncident]:
    features = SeverityFeatures(
        vehicle_classes=["car", "motorcycle", "truck"], num_vehicles=3, person_down=True
    )
    return [
        GroundTruthIncident(
            id=f"GT-T-{k:04d}",
            onset_ts=at(0, 10) + timedelta(minutes=k),
            type=EventType.ACCIDENT,
            node=node_id(2, k % 5),
            location=GeoPoint(lat=18.504, lon=73.85),
            ward_id="W01",
            severity_features=features,
            severity_score=0.6,
            severity_class=severity_class,
            requirements={"ambulance": 1, "police": 1, "fire": 0},
            scenario="test",
            replication=0,
        )
        for k in range(n)
    ]


def _sim(network, cfg: CallsConfig = CALLS) -> CallSimulator:
    share = np.ones(network.n_nodes)
    return CallSimulator(network, cfg, TEMPLATES, REQUIREMENTS, share, MIX)


def test_config_is_synthetic_and_flagged():
    from resqcity.common.config import find_unverified

    assert CALLS.synthetic and TEMPLATES.synthetic
    flagged = set(find_unverified(CALLS))
    assert {"reporting.p_unreported", "reporting.first_delay_median_s"} <= flagged
    assert CALLS.location.modes["els"].median_error_m.verified  # cited press source
    assert CALLS.location.mode == "cell"  # ELS is a scenario, not the default (D-028)


def test_reproducible_and_synthetic(toy_network):
    incs = _incidents(20)
    a = _sim(toy_network).generate(incs, at(0, 10), 2, 1, 0)
    b = _sim(toy_network).generate(incs, at(0, 10), 2, 1, 0)
    assert [c.call.model_dump() for c in a] == [c.call.model_dump() for c in b]
    assert all(c.call.synthetic for c in a)
    assert len({c.call.id for c in a}) == len(a)
    assert [c.call.ts for c in a] == sorted(c.call.ts for c in a)


def test_common_random_numbers(toy_network):
    """Changing the false-call rate or the reporting probability leaves the other draws alone."""
    incs = _incidents(40)
    base = {c.call.id: c for c in _sim(toy_network).generate(incs, at(0, 10), 24, 1, 0)}
    more_false = _sim(toy_network, _with(CALLS, "false_calls.rate_per_day.value", 50.0))
    other = {c.call.id: c for c in more_false.generate(incs, at(0, 10), 24, 1, 0)}
    genuine = [k for k, c in base.items() if c.incident_id is not None]
    assert genuine and all(other[k].call == base[k].call for k in genuine)

    always = {"minor": 0.0, "serious": 0.0, "critical": 0.0}
    all_reported = _sim(toy_network, _with(CALLS, "reporting.p_unreported.value", always))
    full = {c.call.id: c for c in all_reported.generate(incs, at(0, 10), 0, 1, 0)}
    assert all(full[k].call == base[k].call for k in genuine)  # reported ones are unchanged
    assert len(full) >= len(genuine)


def test_unreported_and_extra_calls(toy_network):
    incs = _incidents(50)
    never = {"minor": 1.0, "serious": 1.0, "critical": 1.0}
    silent = _sim(toy_network, _with(CALLS, "reporting.p_unreported.value", never))
    assert silent.generate(incs, at(0, 10), 0, 1, 0) == []
    many = _with(
        CALLS, "reporting.extra_calls_mean.value", {"minor": 3, "serious": 3, "critical": 3}
    )
    calls = _sim(toy_network, many).generate(incs, at(0, 10), 0, 1, 0)
    assert len(calls) > 2 * len(incs)


def test_first_call_delay_and_location_error(toy_network):
    incs = _incidents(400)
    sim = _sim(toy_network)
    first, errors = [], []
    for k, inc in enumerate(incs):
        calls = sim.for_incident(inc, k, 7, 0)
        if calls:
            first.append((calls[0].call.ts - inc.onset_ts).total_seconds())
            errors.extend(c.true_error_m for c in calls)
    median = CALLS.reporting.first_delay_median_s.value
    assert np.median(first) == pytest.approx(median, rel=0.15)
    assert np.median(errors) == pytest.approx(CALLS.location.active.median_error_m.value, rel=0.15)
    assert all(c.call.location_error_m == 300.0 for c in sim.for_incident(incs[0], 0, 7, 0))


def test_reported_location_is_displaced_by_the_true_error(toy_network):
    sim = _sim(toy_network)
    inc = _incidents(1)[0]
    for c in sim.for_incident(inc, 0, 3, 0):
        x0, y0 = toy_network.xy[toy_network.index_of(inc.node)]
        loc = c.call.reported_location
        x, y = toy_network.project(loc.lat, loc.lon)[0]
        assert math.hypot(x - x0, y - y0) == pytest.approx(c.true_error_m, abs=0.5)


def test_all_languages_and_held_out_phrasings_appear(toy_network):
    calls = _sim(toy_network).generate(_incidents(200), at(0, 10), 0, 1, 0)
    assert {c.call.language for c in calls} == set(Language)
    texts = " ".join(c.call.caller_text for c in calls)
    assert "Laxmi Road" in texts  # road names come from the network
    assert "{" not in texts  # every placeholder filled


def test_false_call_rate(toy_network):
    sim = _sim(toy_network, _with(CALLS, "false_calls.rate_per_day.value", 24.0))
    calls = sim.false_calls(at(0, 0), 24 * 50, 1, 0)
    assert len(calls) == pytest.approx(24 * 50, rel=0.1)
    assert all(c.incident_id is None for c in calls)
