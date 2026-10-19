"""Round-trip and validation tests for the shared schemas."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from resqcity.common.schemas import (
    SCHEMA_MODELS,
    Agency,
    Assignment,
    AssignmentRole,
    Call,
    Camera,
    DispatchDecision,
    DispatchTier,
    Event,
    EventType,
    Explanation,
    FacilitySource,
    Factor,
    GeoPoint,
    Hospital,
    Incident,
    IncidentStatus,
    Language,
    ServiceCounts,
    SeverityFeatures,
    SourceKind,
    SourceRef,
    Station,
    Unit,
    UnitState,
)

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 9, 1, 8, 30, tzinfo=IST)
SHIVAJINAGAR = GeoPoint(lat=18.5308, lon=73.8475)


def _examples() -> dict[type[BaseModel], BaseModel]:
    """One realistic, fully populated instance of every exported model."""
    counts = ServiceCounts(ambulance=2, police=1, fire=1)
    features = SeverityFeatures(
        vehicle_classes=["truck", "car"],
        overturned=True,
        person_down=True,
        fire_visible=False,
        num_vehicles=2,
    )
    source = SourceRef(kind=SourceKind.CAMERA_EVENT, ref_id="E-0001")
    factor = Factor(name="ambulance_eta_s", value=412.0, weight=0.6)
    assignment = Assignment(
        unit_id="AMB-07", eta_s=412.0, role=AssignmentRole.RESPOND, hospital_id="H-03"
    )
    return {
        GeoPoint: SHIVAJINAGAR,
        ServiceCounts: counts,
        SeverityFeatures: features,
        SourceRef: source,
        Factor: factor,
        Camera: Camera(
            id="C01",
            lat=18.5308,
            lon=73.8475,
            junction_node=245698123,
            ward_id="W-12",
            fov_note="faces north along JM Road",
            stream_url="rtsp://localhost:8554/C01",
        ),
        Event: Event(
            id="E-0001",
            camera_id="C01",
            ts=T0,
            type=EventType.ACCIDENT,
            confidence=0.83,
            severity_features=features,
            severity_score=0.9,
            required_services=counts,
            clip_ref="clips/C01/2026-09-01T08-30-00.mp4",
            confidence_components={"heuristics": 0.7, "classifier": 0.88},
        ),
        Call: Call(
            id="CALL-001",
            ts=T0 + timedelta(minutes=3),
            reported_location=GeoPoint(lat=18.531, lon=73.848),
            location_error_m=120.0,
            caller_text="Truck ulta ho gaya hai, log ghayal hain",
            language=Language.HINDI,
            triaged_type=EventType.ACCIDENT,
            triaged_severity=0.8,
            triage_confidence=0.7,
        ),
        Incident: Incident(
            id="INC-001",
            type=EventType.ACCIDENT,
            location=SHIVAJINAGAR,
            node=245698123,
            ward_id="W-12",
            created_ts=T0,
            updated_ts=T0 + timedelta(minutes=3),
            sources=[source, SourceRef(kind=SourceKind.CALL, ref_id="CALL-001")],
            posterior=0.97,
            severity=0.9,
            status=IncidentStatus.CONFIRMED,
            requirements=counts,
        ),
        Unit: Unit(
            id="AMB-07",
            agency=Agency.AMBULANCE,
            home_station="ST-AMB-02",
            node=11223344,
            state=UnitState.EN_ROUTE,
            eta_fn_ref="ambulance",
            position=GeoPoint(lat=18.52, lon=73.85),
        ),
        Station: Station(
            id="ST-FIRE-01",
            name="Central Fire Station",
            agency=Agency.FIRE,
            location=GeoPoint(lat=18.512, lon=73.857),
            node=99887766,
            ward_id="W-03",
            source=FacilitySource.OSM,
        ),
        Hospital: Hospital(
            id="H-03",
            name="Example General Hospital",
            location=GeoPoint(lat=18.51, lon=73.83),
            node=55443322,
            has_trauma=True,
            has_burns=False,
            icu_beds=12,
            beds_available=40,
        ),
        Assignment: assignment,
        DispatchDecision: DispatchDecision(
            id="DEC-001",
            incident_id="INC-001",
            created_ts=T0 + timedelta(seconds=5),
            policy="FULL",
            assignments=[
                assignment,
                Assignment(unit_id="POL-02", eta_s=300.0, role=AssignmentRole.GREEN_CORRIDOR),
            ],
            tier=DispatchTier.FULL,
            expected_last_arrival_s=612.0,
            objective_breakdown={"weighted_last_arrival": 550.8, "coverage_loss": 12.0},
        ),
        Explanation: Explanation(
            decision_id="DEC-001",
            text="Sent AMB-07 (nearest trauma-capable route) and POL-02 to clear traffic.",
            factors=[factor],
        ),
    }


def test_every_exported_model_has_an_example():
    assert set(_examples()) == set(SCHEMA_MODELS)


@pytest.mark.parametrize("model", SCHEMA_MODELS, ids=lambda m: m.__name__)
def test_json_round_trip(model: type[BaseModel]):
    original = _examples()[model]
    restored = model.model_validate_json(original.model_dump_json())
    assert restored == original


@pytest.mark.parametrize("model", SCHEMA_MODELS, ids=lambda m: m.__name__)
def test_python_dict_round_trip(model: type[BaseModel]):
    original = _examples()[model]
    assert model.model_validate(original.model_dump(mode="json")) == original


def test_timezone_preserved_through_json():
    event = _examples()[Event]
    restored = Event.model_validate_json(event.model_dump_json())
    assert restored.ts.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_confidence_must_be_probability(bad: float):
    data = _examples()[Event].model_dump()
    data["confidence"] = bad
    with pytest.raises(ValidationError):
        Event.model_validate(data)


def test_naive_datetime_rejected():
    data = _examples()[Event].model_dump()
    data["ts"] = datetime(2026, 9, 1, 8, 30)
    with pytest.raises(ValidationError):
        Event.model_validate(data)


def test_unknown_fields_rejected():
    data = _examples()[Camera].model_dump()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        Camera.model_validate(data)


def test_unknown_enum_value_rejected():
    data = _examples()[Event].model_dump()
    data["type"] = "earthquake"
    with pytest.raises(ValidationError):
        Event.model_validate(data)


def test_negative_service_counts_rejected():
    with pytest.raises(ValidationError):
        ServiceCounts(ambulance=-1)


def test_incident_updated_before_created_rejected():
    data = _examples()[Incident].model_dump()
    data["updated_ts"] = T0 - timedelta(seconds=1)
    with pytest.raises(ValidationError):
        Incident.model_validate(data)


def test_incident_needs_at_least_one_source():
    data = _examples()[Incident].model_dump()
    data["sources"] = []
    with pytest.raises(ValidationError):
        Incident.model_validate(data)


def test_decision_rejects_duplicate_units():
    data = _examples()[DispatchDecision].model_dump()
    data["assignments"] = [data["assignments"][0], data["assignments"][0]]
    with pytest.raises(ValidationError):
        DispatchDecision.model_validate(data)


def test_assignment_validated_on_mutation():
    assignment = _examples()[Assignment]
    with pytest.raises(ValidationError):
        assignment.eta_s = -5.0


def test_service_counts_helpers():
    counts = ServiceCounts(ambulance=2, police=1, fire=0)
    assert counts.get(Agency.AMBULANCE) == 2
    assert counts.get(Agency.FIRE) == 0
    assert counts.total == 3


def test_call_is_synthetic_by_default():
    call = Call(
        id="CALL-9",
        ts=T0,
        reported_location=SHIVAJINAGAR,
        location_error_m=0.0,
        caller_text="accident",
    )
    assert call.synthetic is True


# --- property-based round trips ------------------------------------------------------------

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
ids = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", min_size=1, max_size=20)
aware_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
service_counts = st.builds(
    ServiceCounts,
    ambulance=st.integers(0, 5),
    police=st.integers(0, 5),
    fire=st.integers(0, 5),
)
features = st.builds(
    SeverityFeatures,
    vehicle_classes=st.lists(st.sampled_from(["car", "bus", "truck", "motorcycle"]), max_size=4),
    overturned=st.booleans(),
    person_down=st.booleans(),
    fire_visible=st.booleans(),
    num_vehicles=st.integers(0, 10),
)
events = st.builds(
    Event,
    id=ids,
    camera_id=ids,
    ts=aware_datetimes,
    type=st.sampled_from(EventType),
    confidence=probabilities,
    severity_features=features,
    severity_score=probabilities,
    required_services=service_counts,
    clip_ref=st.none() | st.text(max_size=30),
    confidence_components=st.dictionaries(
        st.text(min_size=1, max_size=10), st.floats(allow_nan=False, allow_infinity=False)
    ),
)
decisions = st.builds(
    DispatchDecision,
    id=ids,
    incident_id=ids,
    created_ts=aware_datetimes,
    policy=st.sampled_from(["B0", "B1", "B2", "B3", "FULL"]),
    assignments=st.lists(
        st.builds(
            Assignment,
            unit_id=ids,
            eta_s=st.floats(min_value=0, max_value=1e5, allow_nan=False),
            role=st.sampled_from(AssignmentRole),
            hospital_id=st.none() | ids,
        ),
        max_size=6,
        unique_by=lambda a: a.unit_id,
    ),
    tier=st.sampled_from(DispatchTier),
    expected_last_arrival_s=st.none() | st.floats(min_value=0, max_value=1e5, allow_nan=False),
)


@settings(max_examples=200)
@given(events)
def test_event_property_round_trip(event: Event):
    assert Event.model_validate_json(event.model_dump_json()) == event


@settings(max_examples=200)
@given(decisions)
def test_decision_property_round_trip(decision: DispatchDecision):
    assert DispatchDecision.model_validate_json(decision.model_dump_json()) == decision
