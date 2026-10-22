"""Shared data contracts for every ResQCity module.

These pydantic models are the single source of truth for data exchanged between vision, fusion,
optimization, the simulator, the backend and the dashboard (TypeScript types are generated from
them by ``tools/gen_ts_types.py``).

Units: durations in seconds, distances in metres, probabilities in [0, 1], timestamps are
timezone-aware datetimes. Graph nodes are road-graph node ids (OSM ids after simplification).
"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
Seconds = Annotated[float, Field(ge=0.0, description="Duration in seconds.")]
Metres = Annotated[float, Field(ge=0.0, description="Distance in metres.")]
NodeId = Annotated[int, Field(ge=0, description="Road-graph node id.")]
Count = Annotated[int, Field(ge=0)]
Id = Annotated[str, Field(min_length=1, max_length=64)]


class Schema(BaseModel):
    """Base for all shared models: strict fields, validated on assignment.

    ``json_schema_serialization_defaults_required`` makes fields with defaults *required* in the
    serialization JSON schema, because serialized payloads always contain them. That keeps the
    generated TypeScript types free of spurious optional fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


# ---------------------------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------------------------


class Agency(StrEnum):
    AMBULANCE = "ambulance"
    POLICE = "police"
    FIRE = "fire"


class EventType(StrEnum):
    ACCIDENT = "accident"
    FIRE_SMOKE = "fire_smoke"
    CROWD = "crowd"
    OTHER = "other"


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"


class SourceKind(StrEnum):
    CAMERA_EVENT = "camera_event"
    CALL = "call"


class IncidentStatus(StrEnum):
    CANDIDATE = "candidate"
    VERIFYING = "verifying"
    CONFIRMED = "confirmed"
    DISPATCHED = "dispatched"
    ON_SCENE = "on_scene"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class UnitState(StrEnum):
    IDLE = "idle"
    READY = "ready"
    """Pre-alerted (crew in vehicle, engine on) but not yet moving."""
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    AT_HOSPITAL = "at_hospital"
    RETURNING = "returning"
    RELOCATING = "relocating"
    OUT_OF_SERVICE = "out_of_service"


class FacilitySource(StrEnum):
    OSM = "osm"
    CONFIG = "config"
    SYNTHETIC = "synthetic"


class DispatchTier(StrEnum):
    FULL = "full"
    """Posterior >= tau_high: dispatch every required unit."""
    VERIFY = "verify"
    """tau_low <= posterior < tau_high: nearest police verifies, other agencies pre-alerted."""
    REVIEW = "review"
    """Posterior < tau_low: operator review with timeout, auto-escalate on timeout."""


class AssignmentRole(StrEnum):
    RESPOND = "respond"
    VERIFY = "verify"
    PRE_ALERT = "pre_alert"
    GREEN_CORRIDOR = "green_corridor"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    OVERRIDDEN = "overridden"
    AUTO_EXECUTED = "auto_executed"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------------------------


class GeoPoint(Schema):
    lat: Latitude
    lon: Longitude


class ServiceCounts(Schema):
    """Number of units needed (or sent) per agency."""

    ambulance: Count = 0
    police: Count = 0
    fire: Count = 0

    def get(self, agency: Agency) -> int:
        return int(getattr(self, agency.value))

    @property
    def total(self) -> int:
        return self.ambulance + self.police + self.fire


class SeverityFeatures(Schema):
    """Observable cues from the video that drive the severity score and unit requirements."""

    vehicle_classes: list[str] = Field(default_factory=list)
    overturned: bool = False
    person_down: bool = False
    fire_visible: bool = False
    num_vehicles: Count = 0


class SourceRef(Schema):
    """Pointer from an incident to one piece of evidence (a camera event or a call)."""

    kind: SourceKind
    ref_id: Id


class Factor(Schema):
    """One contributing factor in an explanation (e.g. ETA of the ambulance, posterior)."""

    name: str = Field(min_length=1)
    value: float
    weight: float


# ---------------------------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------------------------


class Camera(Schema):
    id: Id
    lat: Latitude
    lon: Longitude
    junction_node: NodeId
    ward_id: Id
    fov_note: str = ""
    stream_url: str | None = None


class Event(Schema):
    """A detection emitted by the vision pipeline. Only metadata, never frames."""

    id: Id
    camera_id: Id
    ts: AwareDatetime
    type: EventType
    confidence: Probability = Field(description="Calibrated probability that the event is real.")
    severity_features: SeverityFeatures
    severity_score: Probability
    required_services: ServiceCounts
    clip_ref: str | None = None
    confidence_components: dict[str, float] = Field(
        default_factory=dict,
        description="Explainable parts of the confidence score (heuristics, classifier, ...).",
    )
    description: str | None = Field(
        default=None,
        description="One-sentence scene description written by the offline verifier model "
        "(machine-generated; may be wrong).",
    )


class Call(Schema):
    """An emergency (112) call. Calls are synthetic in this project unless stated otherwise."""

    id: Id
    ts: AwareDatetime
    reported_location: GeoPoint
    location_error_m: Metres
    caller_text: str
    language: Language = Language.ENGLISH
    triaged_type: EventType | None = None
    triaged_severity: Probability | None = None
    triage_confidence: Probability | None = None
    synthetic: bool = True


class Incident(Schema):
    """A fused incident hypothesis built from camera events and calls."""

    id: Id
    type: EventType
    location: GeoPoint
    node: NodeId
    ward_id: Id
    created_ts: AwareDatetime
    updated_ts: AwareDatetime
    sources: list[SourceRef] = Field(min_length=1)
    posterior: Probability = Field(description="P(incident is real | evidence).")
    severity: Probability
    status: IncidentStatus = IncidentStatus.CANDIDATE
    requirements: ServiceCounts

    @model_validator(mode="after")
    def _updated_after_created(self) -> Self:
        if self.updated_ts < self.created_ts:
            raise ValueError("updated_ts must not be earlier than created_ts")
        return self


class Unit(Schema):
    id: Id
    agency: Agency
    home_station: Id
    node: NodeId
    state: UnitState = UnitState.IDLE
    eta_fn_ref: str = Field(
        min_length=1, description="Name of the travel-time profile used for this unit's ETAs."
    )
    position: GeoPoint | None = None


class Station(Schema):
    id: Id
    name: str
    agency: Agency
    location: GeoPoint
    node: NodeId
    ward_id: Id
    source: FacilitySource


class Hospital(Schema):
    id: Id
    name: str
    location: GeoPoint
    node: NodeId
    has_trauma: bool
    has_burns: bool
    icu_beds: Count
    beds_available: Count
    capabilities_synthetic: bool = Field(
        default=True, description="True when capabilities/beds are synthetic, not real data."
    )


class Assignment(Schema):
    unit_id: Id
    eta_s: Seconds
    role: AssignmentRole = AssignmentRole.RESPOND
    hospital_id: Id | None = None


class DispatchDecision(Schema):
    id: Id
    incident_id: Id
    created_ts: AwareDatetime
    policy: str = Field(min_length=1, description="Policy that produced it, e.g. B0 or FULL.")
    assignments: list[Assignment] = Field(default_factory=list)
    tier: DispatchTier
    expected_last_arrival_s: Seconds | None = Field(
        default=None, description="Time until the last required unit arrives; null if none sent."
    )
    objective_breakdown: dict[str, float] = Field(default_factory=dict)
    status: DecisionStatus = DecisionStatus.PROPOSED

    @model_validator(mode="after")
    def _units_unique(self) -> Self:
        unit_ids = [a.unit_id for a in self.assignments]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("a unit may appear at most once in a decision")
        return self


class Explanation(Schema):
    decision_id: Id
    text: str = Field(min_length=1)
    factors: list[Factor] = Field(default_factory=list)


SCHEMA_MODELS: tuple[type[Schema], ...] = (
    GeoPoint,
    ServiceCounts,
    SeverityFeatures,
    SourceRef,
    Factor,
    Camera,
    Event,
    Call,
    Incident,
    Unit,
    Station,
    Hospital,
    Assignment,
    DispatchDecision,
    Explanation,
)
"""Every model exported to the dashboard (TypeScript) and covered by round-trip tests."""
