"""Statistical camera-event process on the toy network (phase 4)."""

import itertools
from datetime import timedelta

import pytest

from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Camera, EventType, GeoPoint, SeverityFeatures
from resqcity.fusion.config import DetectorModel, FusionConfig, load_fusion_config
from resqcity.sim.detection import DetectionProcess, coverage
from resqcity.sim.groundtruth import GroundTruthIncident
from tests.conftest import at, node_id

REQUIREMENTS = load_requirements_config()
FUSION = load_fusion_config()


def _detector(**kw: float) -> DetectorModel:
    base = {
        "p_detect": 0.6,
        "delay_s": 3.0,
        "false_rate_per_armed_h": 55.0,
        "cooldown_s": 60.0,
        "bin_edges": [0.3, 0.4, 0.5, 1.0],
        "real_shares": [0.2, 0.2, 0.6],
        "false_shares": [1 / 3, 1 / 3, 1 / 3],
        "real_confidences": [0.6, 0.7],
        "false_confidences": [0.35, 0.45],
    }
    return DetectorModel(**{**base, **kw})


def _cfg(**camera: float) -> FusionConfig:
    data = FUSION.model_dump(mode="json")
    for key, value in camera.items():
        data["camera"][key]["value"] = value
    return FusionConfig.model_validate(data)


def _camera(cid: str, r: int, c: int) -> Camera:
    return Camera(
        id=cid, lat=18.5, lon=73.85, junction_node=node_id(r, c), ward_id="W01", fov_note="toy"
    )


def _incident(k: int, node: int, minutes: float, etype: EventType = EventType.ACCIDENT):
    return GroundTruthIncident(
        id=f"GT-D-{k:05d}",
        onset_ts=at(0, 0) + timedelta(minutes=minutes),
        type=etype,
        node=node,
        location=GeoPoint(lat=18.5, lon=73.85),
        ward_id="W01",
        severity_features=SeverityFeatures(person_down=True, num_vehicles=2),
        severity_score=0.5,
        severity_class="serious",
        requirements={"ambulance": 1, "police": 1, "fire": 0},
        scenario="test",
        replication=0,
    )


def _process(network, cams, detector=None, cfg=None) -> DetectionProcess:
    return DetectionProcess(network, cams, detector or _detector(), cfg or FUSION, REQUIREMENTS)


def test_coverage_is_a_road_graph_ball(toy_network):
    cams = [_camera("C1", 2, 2)]
    near = coverage(toy_network, cams, 50.0)["C1"]
    assert [int(toy_network.node_ids[i]) for i in near] == [node_id(2, 2)]
    wide = coverage(toy_network, cams, 250.0)["C1"]
    assert len(wide) == 5  # the junction and its four neighbours (~211-222 m away)


def test_false_alarm_long_run_rate(toy_network):
    det = _detector()
    events = _process(toy_network, [_camera("C1", 2, 2)], det).generate([], at(0, 0), 500.0, 1, 0)
    assert all(e.incident_id is None for e in events)
    assert len(events) / 500.0 == pytest.approx(det.long_run_false_rate_per_h, rel=0.05)
    gaps = [(b.event.ts - a.event.ts).total_seconds() for a, b in itertools.pairwise(events)]
    assert min(gaps) >= det.cooldown_s
    assert {e.event.confidence for e in events} <= {0.35, 0.45}


def test_true_detections_and_coverage(toy_network):
    det = _detector(false_rate_per_armed_h=1e-6)  # no false alarms
    cams = [_camera("C1", 2, 2)]
    seen = [_incident(k, node_id(2, 2), 10 * k) for k in range(600)]
    unseen = [_incident(1000 + k, node_id(0, 0), 10 * k) for k in range(50)]
    fires = [
        _incident(2000 + k, node_id(2, 2), 10 * k + 5, EventType.FIRE_SMOKE) for k in range(50)
    ]
    events = _process(toy_network, cams, det).generate(seen + unseen + fires, at(0, 0), 200.0, 1, 0)
    hit = {e.incident_id for e in events}
    assert hit <= {i.id for i in seen}  # out of view and fires are never detected
    assert len(hit) / len(seen) == pytest.approx(det.p_detect, abs=0.06)
    e = events[0]
    inc = next(i for i in seen if i.id == e.incident_id)
    assert e.event.ts - inc.onset_ts == timedelta(seconds=det.delay_s)
    assert e.event.required_services.police >= 1


def test_cooldown_masks_a_crash(toy_network):
    det = _detector(false_rate_per_armed_h=1e-6, p_detect=1.0)
    cams = [_camera("C1", 2, 2)]
    a, b = _incident(0, node_id(2, 2), 10.0), _incident(1, node_id(2, 2), 10.5)  # 30 s apart
    events = _process(toy_network, cams, det).generate([a, b], at(0, 0), 1.0, 1, 0)
    assert [e.incident_id for e in events] == [a.id]


def test_reproducible(toy_network):
    cams = [_camera("C1", 2, 2), _camera("C2", 1, 1)]
    incs = [_incident(k, node_id(2, 2), 7 * k) for k in range(30)]
    a = _process(toy_network, cams).generate(incs, at(0, 0), 4.0, 3, 0)
    b = _process(toy_network, cams).generate(incs, at(0, 0), 4.0, 3, 0)
    assert [e.event for e in a] == [e.event for e in b]
    assert len({e.event.id for e in a}) == len(a)


def test_persistent_cameras_alarm_more(toy_network):
    cams = [_camera(f"C{k}", k % 5, k // 5) for k in range(20)]
    base = _process(toy_network, cams).generate([], at(0, 0), 20.0, 1, 0)
    cfg = _cfg(persistent_share=1.0, persistent_rate_multiplier=3.0)
    more = _process(toy_network, cams, cfg=cfg).generate([], at(0, 0), 20.0, 1, 0)
    assert len(more) > 1.3 * len(base)
