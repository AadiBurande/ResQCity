"""Fusion engine on the toy network: association, posterior updates, reviews, expiry (phase 4)."""

from datetime import timedelta

import numpy as np
import pytest

from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import (
    Call,
    Camera,
    Event,
    EventType,
    GeoPoint,
    IncidentStatus,
    Language,
    Review,
    SeverityFeatures,
    SourceKind,
)
from resqcity.fusion.config import (
    DetectorModel,
    load_calls_config,
    load_fusion_config,
    load_triage_config,
)
from resqcity.fusion.engine import FusionEngine
from resqcity.fusion.model import Background
from resqcity.fusion.triage import RuleTriager
from resqcity.geo.cameras import camera_coverage
from tests.conftest import at, node_id

REQ = load_requirements_config()
FUSION = load_fusion_config()
TRIAGE = RuleTriager(load_triage_config(), REQ)
DET = DetectorModel(
    p_detect=0.6,
    delay_s=3.0,
    false_rate_per_armed_h=55.0,
    cooldown_s=60.0,
    bin_edges=[0.3, 0.4, 0.5, 1.0],
    real_shares=[0.15, 0.17, 0.68],
    false_shares=[1 / 3, 1 / 3, 1 / 3],
    real_confidences=[0.6],
    false_confidences=[0.35],
)
T0 = at(0, 10)


def _engine(network) -> FusionEngine:
    cams = [
        Camera(id=cid, lat=18.504, lon=73.854, junction_node=node_id(2, 2), ward_id="W01",
               fov_note="toy")  # two views of the same junction
        for cid in ("C1", "C2")
    ]  # fmt: skip
    bg = Background(
        node_share=np.full(network.n_nodes, 1.0 / network.n_nodes),
        incident_rate_per_day=5.6,
        type_share={EventType.ACCIDENT: 0.5, EventType.FIRE_SMOKE: 0.5},
        false_call_rate_per_day=2.0,
    )
    cover = camera_coverage(network, cams, FUSION.camera.coverage_radius_m.value)
    return FusionEngine(network, cams, DET, load_calls_config(), FUSION, REQ, bg, cover)


def _event(cid: str, seconds: float, confidence: float = 0.6, eid: str | None = None) -> Event:
    return Event(
        id=eid or f"EV-{cid}-{seconds:g}",
        camera_id=cid,
        ts=T0 + timedelta(seconds=seconds),
        type=EventType.ACCIDENT,
        confidence=confidence,
        severity_features=SeverityFeatures(person_down=True, num_vehicles=2),
        severity_score=0.5,
        required_services={"ambulance": 1, "police": 1, "fire": 0},
    )


def _call(network, r: int, c: int, seconds: float, text: str, err: float = 50.0, cid: str = ""):
    i = network.index_of(node_id(r, c))
    call = Call(
        id=cid or f"CALL-{r}{c}-{seconds:g}",
        ts=T0 + timedelta(seconds=seconds),
        reported_location=GeoPoint(
            lat=float(network.nodes["lat"][i]), lon=float(network.nodes["lon"][i])
        ),
        location_error_m=err,
        caller_text=text,
        language=Language.ENGLISH,
    )
    return call, TRIAGE.triage(call)


CRASH = "A car and a bike have collided near Laxmi Road. Someone is injured and lying on the road."


def test_two_cameras_and_one_call_make_one_incident(toy_network):
    eng = _engine(toy_network)
    eng.on_event(_event("C1", 0.0))
    before = eng.on_event(_event("C2", 1.0))[0]
    after = eng.on_call(*_call(toy_network, 2, 2, 150.0, CRASH))[0]
    assert len(eng.candidates) == 1
    assert {s.kind for s in after.sources} == {SourceKind.CAMERA_EVENT, SourceKind.CALL}
    assert len(after.sources) == 3
    assert after.node == node_id(2, 2)  # the camera junction is the most precise anchor
    assert after.posterior > before.posterior
    assert after.requirements.ambulance >= 1 and after.type == EventType.ACCIDENT
    assert set(after.evidence_log_odds) >= {"prior", "calls", "camera:C1", "camera:C2"}


def test_separate_crashes_stay_separate(toy_network):
    eng = _engine(toy_network)
    eng.on_call(*_call(toy_network, 0, 0, 0.0, CRASH))
    eng.on_call(*_call(toy_network, 4, 4, 30.0, CRASH))
    assert len(eng.candidates) == 2
    # a second caller about the first crash joins it
    eng.on_call(*_call(toy_network, 0, 1, 90.0, CRASH))
    assert sorted(len(c.calls) for c in eng.candidates.values()) == [1, 2]


def test_wide_location_error_merges_by_the_gate(toy_network):
    eng = _engine(toy_network)
    eng.on_call(*_call(toy_network, 0, 0, 0.0, CRASH, err=300.0))
    eng.on_call(*_call(toy_network, 1, 1, 60.0, CRASH, err=300.0))  # ~430 m away, gate 750 m
    assert len(eng.candidates) == 1


def test_reviews_move_the_posterior(toy_network):
    eng = _engine(toy_network)
    base = eng.on_event(_event("C1", 0.0, eid="EV-A"))[0].posterior
    rejected = eng.on_review(Review(id="R1", ts=T0 + timedelta(seconds=30), event_id="EV-A",
                                    confirmed=False))[0]  # fmt: skip
    assert rejected.posterior < base
    eng2 = _engine(toy_network)
    eng2.on_event(_event("C1", 0.0, eid="EV-B"))
    confirmed = eng2.on_review(Review(id="R2", ts=T0 + timedelta(seconds=30), event_id="EV-B",
                                      confirmed=True))[0]  # fmt: skip
    assert confirmed.posterior > base
    assert eng.on_review(Review(id="R3", ts=T0, event_id="nope", confirmed=True)) == []


def test_camera_alone_is_weak_and_decays_then_expires(toy_network):
    eng = _engine(toy_network)
    first = eng.on_event(_event("C1", 0.0))[0]
    assert first.posterior < 0.01  # a camera alarm alone is weak evidence (D-028)
    eng.tick(T0 + timedelta(seconds=600))
    cand = next(iter(eng.candidates.values()))
    assert cand.posterior < first.posterior  # no call has come
    out = eng.tick(T0 + timedelta(seconds=FUSION.update.expiry_s + 10))
    assert out and out[0].status == IncidentStatus.DISMISSED


def test_posterior_does_not_depend_on_arrival_order(toy_network):
    now = T0 + timedelta(seconds=400)
    a = _engine(toy_network)
    a.on_event(_event("C1", 0.0))
    a.on_call(*_call(toy_network, 2, 2, 150.0, CRASH))
    b = _engine(toy_network)
    b.on_call(*_call(toy_network, 2, 2, 150.0, CRASH))
    b.on_event(_event("C1", 0.0))
    ca, cb = next(iter(a.candidates.values())), next(iter(b.candidates.values()))
    pa, _ = a.model.posterior(a.evidence(ca, now))
    pb, _ = b.model.posterior(b.evidence(cb, now))
    assert pa == pytest.approx(pb)


def test_call_only_incident_type_and_severity(toy_network):
    eng = _engine(toy_network)
    inc = eng.on_call(*_call(toy_network, 1, 3, 0.0, "There is a fire on Tilak Road."))[0]
    assert inc.type == EventType.FIRE_SMOKE
    assert inc.requirements.fire >= 1
    assert 0.3 < inc.posterior < 0.99  # one call: likely but not certain
