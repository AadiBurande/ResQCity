"""Alarm state machine and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from resqcity.vision.alarm import Alarm, alarm_times, first_alarm_time
from resqcity.vision.config import AlarmConfig
from resqcity.vision.eval.metrics import (
    auroc,
    binary_metrics,
    reliability,
    summarise_alarms,
    video_alarm_summary,
)

CFG = AlarmConfig(threshold=0.6, consecutive_windows=2, cooldown_s=10.0)


def test_alarm_needs_consecutive_windows():
    alarm = Alarm(CFG)
    assert [alarm.update(t, c) for t, c in enumerate([0.7, 0.2, 0.7, 0.8])] == [
        False,
        False,
        False,
        True,
    ]


def test_alarm_cooldown():
    times = [0, 1, 2, 3, 4, 13, 14]
    conf = [0.9] * len(times)
    assert alarm_times(times, conf, CFG, 0.6) == [1, 14]


def test_first_alarm_time_none_when_never_confident():
    assert first_alarm_time([0, 1, 2], [0.1, 0.2, 0.3], CFG, 0.6) is None


def test_auroc_matches_definition():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_binary_metrics():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (m.precision, m.recall, m.f1) == (0.5, 0.5, 0.5)


def test_reliability_perfectly_calibrated_has_zero_ece():
    p = np.repeat([0.05, 0.95], 100)
    y = np.concatenate([np.r_[np.ones(5), np.zeros(95)], np.r_[np.ones(95), np.zeros(5)]])
    assert reliability(y, p).attrs["ece"] == pytest.approx(0.0, abs=1e-12)


def test_video_alarm_summary_delay_and_false_alarms():
    videos = pd.DataFrame(
        [
            {
                "video_id": "a",
                "source": "s",
                "label": "accident",
                "has_onset": True,
                "accident_onset_s": 10.0,
                "duration_s": 30.0,
            },
            {
                "video_id": "n",
                "source": "s",
                "label": "normal",
                "has_onset": False,
                "accident_onset_s": None,
                "duration_s": 3600.0,
            },
        ]
    )
    windows = pd.DataFrame(
        {
            "video_id": ["a"] * 6 + ["n"] * 4,
            "end_s": [3, 4, 15, 16, 17, 18, 100, 101, 2000, 2001],  # cooldown 10 s after t=4
            "p": [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9],
        }
    )
    per = video_alarm_summary(windows, videos, "p", CFG, 0.6)
    acc = per.set_index("video_id").loc["a"]
    assert acc["false_alarms"] == 1 and acc["detected"] and acc["delay_s"] == pytest.approx(6.0)
    assert per.set_index("video_id").loc["n", "false_alarms"] == 2
    summary = summarise_alarms(per)
    hours = 10 / 3600 + 1.0
    assert summary["false_alarms_per_camera_hour"] == pytest.approx(3 / hours)
    assert summary["video_recall"] == 1.0 and summary["delay_median_s"] == pytest.approx(6.0)


def test_intervals():
    from resqcity.vision.eval.metrics import poisson_rate_interval, wilson_interval

    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = poisson_rate_interval(0, 1.0)  # zero events in 1 h: upper bound ~3.69/h
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=1e-3)
