"""Observable severity cues from tracks, then severity score and required services via the shared
rule table (``common.requirements``, the same functions used for simulator ground truth)."""

from resqcity.common.requirements import RequirementsConfig, required_services, severity_score
from resqcity.common.schemas import EventType, ServiceCounts, SeverityFeatures
from resqcity.vision.heuristics import HeuristicResult
from resqcity.vision.types import Detections


def severity_features(
    det: Detections, heur: HeuristicResult, vehicle_classes: set[str], fire_visible: bool
) -> SeverityFeatures:
    involved = [t for t in det.tracks if t.track_id in heur.involved and t.cls in vehicle_classes]
    if not involved:  # alarm from the learned stage only: fall back to the vehicles in view
        involved = [t for t in det.tracks if t.cls in vehicle_classes]
    return SeverityFeatures(
        vehicle_classes=sorted(t.cls for t in involved),
        overturned=bool(heur.overturned_ids),
        person_down=bool(heur.person_down_ids),
        fire_visible=fire_visible,
        num_vehicles=len(involved),
    )


def assess(
    event_type: EventType, features: SeverityFeatures, rules: RequirementsConfig
) -> tuple[float, ServiceCounts]:
    return (
        severity_score(event_type, features, rules.severity),
        required_services(event_type, features, rules),
    )
