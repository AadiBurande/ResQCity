"""Phase-3b pieces without models: verified alarms, feature tags, UCA labels, tracker merge."""

import numpy as np
import pandas as pd

from resqcity.vision.alarm import alarm_times, verified_alarm_times
from resqcity.vision.clip_classifier import feature_tag, frame_dim, head_input_dim
from resqcity.vision.config import load_vision_config
from resqcity.vision.detect_track import PERSON_ID_OFFSET, DualTracker
from resqcity.vision.train_clip import build_windows
from resqcity.vision.types import Box, Detections, FramePacket, TrackObs

CFG = load_vision_config()


def test_rejected_alarm_starts_no_cooldown():
    alarm = CFG.alarm.model_copy(update={"consecutive_windows": 1, "cooldown_s": 60.0})
    times, scores = [1.0, 2.0, 3.0], [0.9, 0.9, 0.9]
    assert alarm_times(times, scores, alarm, 0.5) == [1.0]  # cooldown blocks 2 and 3
    assert verified_alarm_times(times, scores, alarm, 0.5, lambda t: t >= 2.0) == [2.0]
    assert verified_alarm_times(times, scores, alarm, 0.5, lambda t: False) == []


def test_siglip_feature_tag_and_dims_follow_the_prompts():
    cc = CFG.clip_classifier.model_copy(update={"backbone": "siglip2_base"})
    assert frame_dim(cc) == 768 + len(cc.text_prompts)
    assert head_input_dim(cc) == 3 * frame_dim(cc)
    other = cc.model_copy(update={"text_prompts": [*cc.text_prompts, "a fire"]})
    assert feature_tag(cc).startswith("siglip2_base_") and feature_tag(cc) != feature_tag(other)
    mnet = CFG.clip_classifier.model_copy(update={"backbone": "mobilenet_v3_small"})
    assert feature_tag(mnet) == "mobilenet_v3_small" and head_input_dim(mnet) == 3 * 576


def test_uca_intervals_label_train_videos_only(tmp_path):
    cfg = CFG.model_copy(
        update={"clip_classifier": CFG.clip_classifier.model_copy(update={"window_s": 2.0})}
    )
    times = np.arange(0.0, 10.0, 0.5)
    folder = tmp_path / "features" / "mobilenet_v3_small"
    folder.mkdir(parents=True)
    for vid in ("ucf_a", "ucf_b"):
        np.savez_compressed(folder / f"{vid}.npz", times=times, feats=np.ones((len(times), 4)))
    ann = pd.DataFrame(
        {
            "video_id": ["ucf_a", "ucf_b"],
            "source": "ucf_crime",
            "split": ["train", "val"],
            "label": "accident",
            "has_onset": False,
            "accident_onset_s": np.nan,
            "accident_offset_s": np.nan,
        }
    )
    iv = pd.DataFrame(
        {"video_id": ["ucf_a", "ucf_b"], "train_onset_s": [4.0, 4.0], "train_offset_s": [6.0, 6.0]}
    )
    w, _ = build_windows(ann, cfg, tmp_path, iv)
    a, b = w[w["video_id"] == "ucf_a"], w[w["video_id"] == "ucf_b"]
    assert set(a["label"].dropna()) == {0.0, 1.0}
    assert b["label"].isna().all()  # validation keeps official labels only (none here)
    w_mil, _ = build_windows(ann, cfg, tmp_path, None)
    assert w_mil["label"].isna().all()


class _Fake:
    def __init__(self, tracks):
        self.tracks = tracks

    def update(self, frame):
        return Detections(t=frame.t, tracks=list(self.tracks))

    def reset(self):
        pass


def test_dual_tracker_merges_without_id_collisions():
    box = Box(0, 0, 10, 10)
    dual = DualTracker.__new__(DualTracker)
    dual.vehicles = _Fake([TrackObs(1, "auto_rickshaw", 0.9, box)])
    dual.persons = _Fake([TrackObs(1, "person", 0.8, box)])
    out = dual.update(FramePacket(t=0.0, image=np.zeros((4, 4, 3), np.uint8)))
    assert [(o.track_id, o.cls) for o in out.tracks] == [
        (1, "auto_rickshaw"),
        (1 + PERSON_ID_OFFSET, "person"),
    ]


def test_temporal_score_matches_the_accident_benchmark_definition():
    from resqcity.vision.eval.final_v5 import temporal_score

    truth = np.array([10.0, 20.0])
    assert temporal_score(truth, truth, 1.0) == 1.0
    off = temporal_score(np.array([11.0, 20.0]), truth, 1.0)
    assert off == (np.exp(-0.5) + 1.0) / 2


def test_predicted_impact_is_the_best_window_centre_or_midpoint():
    from resqcity.vision.eval.final_v5 import predicted_impact

    w = pd.DataFrame(
        {
            "video_id": ["a", "a", "a"],
            "start_s": [0.0, 1.0, 2.0],
            "end_s": [3.0, 4.0, 5.0],
            "p": [0.1, 0.9, 0.2],
        }
    )
    videos = pd.DataFrame({"video_id": ["a", "b"], "duration_s": [30.0, 8.0]})
    assert predicted_impact(w, videos, "p").tolist() == [2.5, 4.0]


def test_make_vision_trains_exactly_the_preregistered_candidates():
    from resqcity.tools.tasks import V5_RUNS
    from resqcity.vision.eval.select_v5 import STAGE_A, SWEEPS

    assert list(V5_RUNS) == STAGE_A and len(STAGE_A) == 12
    assert all((SWEEPS / f"{r}.yaml").exists() for r in STAGE_A)


def test_pre_v5_scope_drops_accident_clips():
    from resqcity.vision.datasets.annotations import pre_v5

    ann = pd.DataFrame({"source": ["ucf_crime", "accident", "cadp"], "split": "val"})
    assert pre_v5(ann)["source"].tolist() == ["ucf_crime", "cadp"]


def test_window_stream_reproduces_make_windows():
    from resqcity.vision.clip_classifier import WindowStream, make_windows

    rng = np.random.default_rng(3)
    for _ in range(20):  # irregular (variable-frame-rate) sampling
        times = np.cumsum(rng.uniform(0.05, 0.6, size=int(rng.integers(2, 80))))
        expected = [
            (w.start_s, w.end_s, list(range(w.first, w.last)))
            for w in make_windows(times, 3.0, 1.0)
        ]
        stream, got = WindowStream(3.0, 1.0), []
        for i, t in enumerate(times):
            got += stream.push(float(t), i)
        got += stream.finish()
        assert got == expected


def test_ua_detrac_records_and_normal_road_false_alarms(tmp_path):
    import cv2

    from resqcity.vision.datasets.annotations import FA_EVAL, detrac_records
    from resqcity.vision.eval.metrics import SECONDS_PER_HOUR, normal_road_summary

    seq = tmp_path / "ua_detrac" / "detrac_1_MVI_20011"
    seq.mkdir(parents=True)
    for i in range(50):  # 50 stored frames at 25 / 5 = 5 fps -> 10 s
        cv2.imwrite(str(seq / f"img{1 + 5 * i:05d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    rec = detrac_records(tmp_path, CFG)
    assert rec.iloc[0][["fps", "duration_s", "split", "label", "kind"]].tolist() == [
        5.0,
        10.0,
        FA_EVAL,
        "normal",
        "frames",
    ]
    assert rec.iloc[0]["video_id"] == "ua_detrac_1_MVI_20011"
    w = pd.DataFrame(
        {"video_id": "ua_detrac_1_MVI_20011", "end_s": [3.0, 4.0, 5.0, 9.0], "p": [0.9] * 4}
    )
    alarm = CFG.alarm.model_copy(update={"consecutive_windows": 2, "cooldown_s": 3.0})
    row = normal_road_summary(w, rec, "p", alarm, 0.5, 3.0, 1.0).iloc[0]
    assert row["road_false_alarms"] == 1  # at 4 s; after the cooldown only one window (9 s) is high
    assert row["road_hours"] == (10.0 - 4.0) / SECONDS_PER_HOUR
