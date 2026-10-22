"""ACCIDENT split/sealing rules and UCA training intervals (no network, toy files)."""

import pandas as pd
import pytest

from resqcity.vision.config import load_vision_config
from resqcity.vision.datasets.accident import (
    HOLDOUT,
    accident_records,
    assign_accident_splits,
    source_id,
)
from resqcity.vision.datasets.uca import accident_intervals, agreement, parse_uca

CFG = load_vision_config()


def _meta(rows):
    cols = ["path", "split_in_distribution", "accident_time", "no_frames", "duration"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["accident_frame", "center_x", "center_y", "x1", "y1", "x2", "y2", "rollover"]:
        df[c] = 0.0
    df["type"], df["region"], df["quality"] = "rear-end", "World", "good"
    return df


TOY = _meta(
    [[f"real_videos/src{i:02d}_00.mp4", "train", 10.0 + i, 300, 10.0] for i in range(20)]
    + [
        ["real_videos/src00_01.mp4", "train", 5.0, 300, 10.0],  # same source as src00_00
        ["real_videos/both_00.mp4", "train", 4.0, 300, 10.0],  # source on both sides
        ["real_videos/both_01.mp4", "test", 6.0, 300, 10.0],
        ["real_videos/tst_00.mp4", "test", 7.0, 300, 10.0],
    ]
)


def test_source_id_strips_clip_index():
    assert source_id("real_videos/-2UPLUV7JLg_00.mp4") == "-2UPLUV7JLg"


def test_accident_splits_group_sources_and_exclude_straddlers():
    split = assign_accident_splits(TOY, CFG.datasets.accident)
    by_path = dict(zip(TOY["path"], split, strict=True))
    assert by_path["real_videos/tst_00.mp4"] == HOLDOUT
    assert by_path["real_videos/both_01.mp4"] == HOLDOUT
    assert by_path["real_videos/both_00.mp4"] == "excluded"
    assert by_path["real_videos/src00_00.mp4"] == by_path["real_videos/src00_01.mp4"]
    assert set(split) == {"train", "val", HOLDOUT, "excluded"}
    assert (split == "val").sum() == round(20 * CFG.datasets.accident.val_fraction)


def test_holdout_labels_are_sealed(tmp_path):
    base = tmp_path / "accident"
    (base / "real_videos").mkdir(parents=True)
    for p in TOY["path"]:
        (base / p).write_bytes(b"x")
    TOY.to_csv(base / "metadata-real.csv", index=False)
    rows, holdout = accident_records(tmp_path, CFG.datasets.accident)
    hidden = rows[rows["split"] == HOLDOUT]
    assert len(hidden) == 2 and hidden["accident_onset_s"].isna().all()
    assert not hidden["has_onset"].any()
    assert set(holdout["video_id"]) == set(hidden["video_id"])
    assert holdout["accident_time"].notna().all()
    shown = rows[rows["split"] == "train"].iloc[0]
    assert shown["accident_offset_s"] == pytest.approx(
        shown["accident_onset_s"] + CFG.datasets.accident.offset_after_impact_s
    )
    assert shown["fps"] == pytest.approx(30.0)


def test_uca_intervals_take_the_earliest_accident_sentence(tmp_path):
    f = tmp_path / "UCFCrime_Train.txt"
    f.write_text(
        "RoadAccidents002_x264 00:00.0 00:07.9 ##Many cars are driving on the road\n"
        "RoadAccidents002_x264 00:09.2 00:11.0 ##A white bus hit the sidewalk\n"
        "RoadAccidents002_x264 00:12.0 00:15.0 ##A car crashed into the bus\n"
        "Normal_Videos_001_x264 00:00.0 00:30.0 ##A man hit a ball\n",
        encoding="utf-8",
    )
    uca = parse_uca([f])
    assert len(uca) == 4 and set(uca["uca_split"]) == {"train"}
    iv = accident_intervals(uca, CFG.datasets.uca.accident_keywords, "RoadAccidents")
    assert list(iv["video_id"]) == ["ucf_RoadAccidents002_x264"]
    assert iv.iloc[0][["train_onset_s", "train_offset_s"]].tolist() == [9.2, 11.0]


def test_uca_agreement_counts():
    iv = pd.DataFrame(
        {
            "video_id": ["ucf_a", "ucf_b"],
            "train_onset_s": [3.0, 20.0],
            "train_offset_s": [9.0, 25.0],
        }
    )
    ann = pd.DataFrame(
        {
            "video_id": ["ucf_a", "ucf_b"],
            "source": "ucf_crime",
            "label": "accident",
            "has_onset": True,
            "accident_onset_s": [5.0, 10.0],
            "accident_offset_s": [8.0, 12.0],
        }
    )
    a = agreement(iv, ann)
    assert a["uca_interval_overlaps_official"] == 1
    assert a["uca_interval_contains_official_onset"] == 1
    assert a["uca_start_minus_official_onset_s"]["median"] == pytest.approx(4.0)


def test_accident_alarm_rule_tolerance_false_and_late_alarms():
    from resqcity.vision.eval.metrics import SECONDS_PER_HOUR, accident_alarm_summary

    ends = [4.0, 5.0, 9.5, 25.0, 26.0]
    scores = [0.1, 0.9, 0.9, 0.9, 0.1]  # alarms at 5 (before t*-1), 9.5 (tolerance), 25 (late)
    windows = pd.DataFrame({"video_id": "acc_x", "end_s": ends, "p": scores})
    videos = pd.DataFrame(
        {"video_id": ["acc_x"], "source": ["accident"], "accident_onset_s": [10.0]}
    )
    alarm = CFG.alarm.model_copy(update={"consecutive_windows": 1, "cooldown_s": 0.0})
    row = accident_alarm_summary(windows, videos, "p", alarm, 0.5, 3.0, 1.0).iloc[0]
    assert row["detected"] and row["delay_s"] == pytest.approx(-0.5)
    assert row["road_false_alarms"] == 1 and row["late_alarms"] == 1
    assert row["road_hours"] == pytest.approx((10.0 - 1.0 - 3.0) / SECONDS_PER_HOUR)
