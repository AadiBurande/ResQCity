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
