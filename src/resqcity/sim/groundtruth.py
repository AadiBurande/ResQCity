"""Ground-truth incidents of the simulator. SYNTHETIC by construction.

Policies must never see these objects. They only see Events and Calls produced from them by the
detection and call processes (phases 4 and 7).
"""

from pydantic import AwareDatetime, Field

from resqcity.common.schemas import (
    EventType,
    GeoPoint,
    Id,
    NodeId,
    Probability,
    Schema,
    ServiceCounts,
    SeverityFeatures,
)


class GroundTruthIncident(Schema):
    id: Id
    onset_ts: AwareDatetime
    type: EventType
    node: NodeId
    location: GeoPoint
    ward_id: Id
    severity_features: SeverityFeatures
    severity_score: Probability
    severity_class: str
    requirements: ServiceCounts
    near_festival_route: bool = False
    scenario: str
    replication: int = Field(ge=0)
    synthetic: bool = True
