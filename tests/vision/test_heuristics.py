"""Stage-1 heuristics on scripted tracks."""

import pytest

from resqcity.vision.heuristics import HeuristicScorer
from resqcity.vision.types import Box, Detections, TrackObs
from tests.vision.conftest import (
    CRASH_T,
    H,
    ScriptedTracker,
    car,
    collision_script,
    frames,
    free_flow_script,
    parked_script,
)


def run(vcfg, script, duration=6.0):
    scorer = HeuristicScorer(
        vcfg.heuristics, set(vcfg.detector.vehicle_classes), vcfg.detector.person_class
    )
    tracker = ScriptedTracker(script)
    return [scorer.update(tracker.update(f)) for f in frames(duration)]


def test_crash_triggers_collision_and_deceleration(vcfg):
    results = run(vcfg, collision_script)
    settled = [r for r in results if r.t >= CRASH_T + vcfg.heuristics.stop_after_overlap_s]
    assert settled, "scenario too short"
    best = max(settled, key=lambda r: r.score)
    assert best.components["collision"] == 1.0
    assert best.components["deceleration"] == 1.0
    assert best.involved == {1, 2}
    assert best.score >= 0.65


def test_collision_is_pending_before_outcome_known(vcfg):
    results = run(vcfg, collision_script, duration=CRASH_T + 0.6)
    assert max(r.components["collision"] for r in results) == 0.5


@pytest.mark.parametrize("script", [parked_script, free_flow_script])
def test_normal_traffic_scores_zero(vcfg, script):
    results = run(vcfg, script)
    assert max(r.score for r in results) == pytest.approx(0.0, abs=1e-9)


def test_orientation_flip_marks_overturn(vcfg):
    def script(t):
        if t < 2.0:
            return [car(7, 300, w=60, h=H)]
        return [car(7, 300, y=210, w=40, h=60)]  # box turned upright: w/h 1.5 -> 0.67

    results = run(vcfg, script, duration=4.0)
    assert any(7 in r.overturned_ids for r in results)
    assert max(r.components["orientation"] for r in results) == 1.0


def test_person_down(vcfg):
    def script(t):
        return [TrackObs(9, "person", 0.9, Box(400, 300, 470, 330))]  # wide, still

    results = run(vcfg, script, duration=4.0)
    assert results[-1].components["person_down"] == 1.0
    assert results[-1].person_down_ids == {9}
    assert results[0].components["person_down"] == 0.0  # not yet still long enough


def test_tracks_outside_window_are_forgotten(vcfg):
    scorer = HeuristicScorer(vcfg.heuristics, {"car"}, "person")
    scorer.update(Detections(0.0, [car(1, 0)]))
    scorer.update(Detections(100.0, [car(2, 0)]))
    assert set(scorer.tracks) == {2}
