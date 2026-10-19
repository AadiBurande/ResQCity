"""Severity score and requirement rules (configs/requirements.yaml)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from resqcity.common.config import find_unverified
from resqcity.common.requirements import (
    RequirementsConfig,
    RuleCondition,
    load_requirements_config,
    matching_rules,
    required_services,
    severity_class,
    severity_score,
)
from resqcity.common.schemas import EventType, ServiceCounts, SeverityFeatures

CFG = load_requirements_config()


def f(**kw) -> SeverityFeatures:
    return SeverityFeatures(**kw)


def test_config_is_flagged_unverified():
    flagged = find_unverified(CFG)
    assert {"provenance", "severity.base", "severity.weights"} <= set(flagged)


@pytest.mark.parametrize(
    ("event_type", "features", "expected"),
    [
        (EventType.ACCIDENT, f(vehicle_classes=["car"], num_vehicles=1), ServiceCounts(police=1)),
        (
            EventType.ACCIDENT,
            f(vehicle_classes=["car"], num_vehicles=1, person_down=True),
            ServiceCounts(ambulance=1, police=1),
        ),
        (
            EventType.ACCIDENT,
            f(vehicle_classes=["truck", "car"], num_vehicles=2, overturned=True),
            ServiceCounts(ambulance=2, police=1, fire=1),
        ),
        (EventType.FIRE_SMOKE, f(fire_visible=True), ServiceCounts(police=1, fire=1)),
        (EventType.CROWD, f(), ServiceCounts(police=2)),
        (EventType.OTHER, f(), ServiceCounts(police=1)),
    ],
)
def test_rule_examples(event_type, features, expected):
    assert required_services(event_type, features, CFG) == expected


def test_heavy_overturned_is_explained_by_named_rule():
    names = {
        r.name
        for r in matching_rules(
            EventType.ACCIDENT, f(vehicle_classes=["bus"], num_vehicles=1, overturned=True), CFG
        )
    }
    assert "heavy_vehicle_overturned" in names


features_strategy = st.builds(
    SeverityFeatures,
    vehicle_classes=st.lists(st.sampled_from(["car", "motorcycle", "truck", "bus"]), max_size=4),
    overturned=st.booleans(),
    person_down=st.booleans(),
    fire_visible=st.booleans(),
    num_vehicles=st.integers(0, 8),
)


@given(st.sampled_from(EventType), features_strategy)
def test_severity_in_unit_interval_and_every_incident_needs_someone(event_type, features):
    score = severity_score(event_type, features, CFG.severity)
    assert 0.0 <= score <= 1.0
    assert severity_class(score, CFG.severity) in {"minor", "serious", "critical"}
    assert required_services(event_type, features, CFG).total >= 1


@given(
    st.sampled_from(EventType),
    features_strategy,
    st.sampled_from(["overturned", "person_down", "fire_visible", "heavy", "vehicles"]),
)
def test_adding_a_cue_never_reduces_requirements(event_type, features, cue):
    """Monotonicity: more evidence of severity never sends fewer units of any agency."""
    if cue == "heavy":
        more = features.model_copy(update={"vehicle_classes": [*features.vehicle_classes, "truck"]})
    elif cue == "vehicles":
        more = features.model_copy(update={"num_vehicles": features.num_vehicles + 1})
    else:
        more = features.model_copy(update={cue: True})
    before = required_services(event_type, features, CFG)
    after = required_services(event_type, more, CFG)
    assert after.ambulance >= before.ambulance
    assert after.police >= before.police
    assert after.fire >= before.fire
    assert severity_score(event_type, more, CFG.severity) >= severity_score(
        event_type, features, CFG.severity
    )


def test_negative_condition_rejected():
    with pytest.raises(ValidationError, match="monotone"):
        RuleCondition(types=[EventType.ACCIDENT], person_down=False)


def test_missing_base_rule_rejected():
    data = CFG.model_dump(mode="json")
    data["rules"] = [r for r in data["rules"] if r["name"] != "crowd_base"]
    with pytest.raises(ValidationError, match="crowd"):
        RequirementsConfig.model_validate(data)
