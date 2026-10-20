"""End-to-end pipeline with a scripted tracker (no YOLO / torch): crash -> Event on the bus."""

from datetime import UTC, datetime

import numpy as np

from resqcity.common.bus import InProcessBus, JsonlBus
from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Camera, Event, EventType
from resqcity.vision.pipeline import TOPIC, CameraPipeline
from resqcity.vision.privacy import Anonymizer
from tests.vision.conftest import ScriptedTracker, collision_script, frames, free_flow_script

CAMERA = Camera(id="C01", lat=18.53, lon=73.85, junction_node=1, ward_id="W01")


class NoFaces:
    def detect(self, image):
        return []


def make(vcfg, script, bus, previews=None):
    anon = Anonymizer(
        vcfg.privacy, vcfg.detector.vehicle_classes, vcfg.detector.person_class, faces=NoFaces()
    )
    return CameraPipeline(
        CAMERA,
        vcfg,
        ScriptedTracker(script),
        anon,
        load_requirements_config(),
        bus,
        start_time=datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
        on_preview=previews.append if previews is not None else None,
    )


def test_crash_emits_one_accident_event(vcfg):
    received: list[Event] = []
    bus = InProcessBus()
    bus.subscribe(TOPIC, lambda _t, msg: received.append(msg))
    previews: list[np.ndarray] = []
    pipe = make(vcfg, collision_script, bus, previews)
    for f in frames(12.0):
        pipe.process(f)
    assert len(received) == 1
    event = received[0]
    assert event.type == EventType.ACCIDENT and event.camera_id == "C01"
    assert event.confidence >= vcfg.alarm.threshold
    assert event.severity_features.num_vehicles == 2
    assert event.required_services.police >= 1
    assert "c_collision" in event.confidence_components
    assert Event.model_validate_json(event.model_dump_json()) == event
    assert len(previews) == len(frames(12.0))


def test_normal_traffic_emits_nothing(vcfg):
    bus = InProcessBus()
    got = []
    bus.subscribe(TOPIC, lambda _t, m: got.append(m))
    pipe = make(vcfg, free_flow_script, bus)
    for f in frames(12.0):
        pipe.process(f)
    assert got == []


def test_events_reach_jsonl_bus(vcfg, tmp_path):
    bus = JsonlBus(tmp_path / "events.jsonl")
    pipe = make(vcfg, collision_script, bus)
    for f in frames(12.0):
        pipe.process(f)
    records = list(bus.read(TOPIC))
    assert len(records) == 1 and records[0]["type"] == "Event"
    Event.model_validate(records[0]["payload"])
