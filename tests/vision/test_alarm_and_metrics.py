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


def _videos():
    rows = [
        ("a", "accident", True, 10.0, 12.0, 60.0),  # onset 10 s, offset 12 s
        ("late", "accident", True, 10.0, 12.0, 60.0),
        ("weak", "accident", False, None, None, 60.0),
        ("n", "normal", False, None, None, 3600.0),
    ]
    cols = ["video_id", "label", "has_onset", "accident_onset_s", "accident_offset_s", "duration_s"]
    return pd.DataFrame(rows, columns=cols).assign(source="s")


def test_video_alarm_summary_classifies_alarms_by_time():
    windows = pd.DataFrame(
        {
            "video_id": ["a"] * 4 + ["late"] * 2 + ["weak"] * 2 + ["n"] * 4,
            # a: alarm at 4 (pre-onset, road FA), cooldown 10 s, alarm at 16 (in [10, 22]: hit)
            # late: alarm at 31 (> offset + 10 s grace: late, not a hit)
            "end_s": [3, 4, 15, 16, 30, 31, 5, 6, 100, 101, 2000, 2001],
            "p": [0.9] * 12,
        }
    )
    per = video_alarm_summary(windows, _videos(), "p", CFG, 0.6).set_index("video_id")
    a = per.loc["a"]
    assert a["detected"] and a["delay_s"] == pytest.approx(6.0) and a["road_false_alarms"] == 1
    assert a["road_hours"] == pytest.approx(10 / 3600)
    late = per.loc["late"]
    assert not late["detected"] and late["late_alarms"] == 1 and late["road_false_alarms"] == 0
    assert per.loc["weak", "any_alarm"] and not per.loc["weak", "detected"]
    assert per.loc["n", "nonroad_false_alarms"] == 2 and per.loc["n", "road_false_alarms"] == 0

    s = summarise_alarms(per.reset_index())
    assert (s["onset_detected"], s["onset_videos"]) == (1, 2)
    assert s["any_alarm_rate"] == 1.0 and s["late_alarms"] == 1
    assert s["road_false_alarms"] == 1 and s["road_hours"] == pytest.approx(20 / 3600)
    assert s["nonroad_false_alarms_per_hour"] == pytest.approx(2.0)
    assert s["fp_video_rate"] == pytest.approx(2 / 3)  # a (road FA) and n (non-road) of 3 negatives
    assert s["youden_j"] == pytest.approx(0.5 - 2 / 3)
    assert (
        s["road_fa_cluster_ci95_lo"]
        <= s["road_false_alarms_per_hour"]
        <= s["road_fa_cluster_ci95_hi"]
    )


def test_per_video_auroc():
    from resqcity.vision.eval.metrics import per_video_auroc

    w = pd.DataFrame(
        {
            "video_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
            "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # c has one class only: skipped
            "p": [0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1, 0.5, 0.5],
        }
    )
    mean, (lo, hi), n = per_video_auroc(w, "p")
    assert (mean, n) == (0.5, 2) and lo <= mean <= hi


def test_intervals():
    from resqcity.vision.eval.metrics import poisson_rate_interval, wilson_interval

    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = poisson_rate_interval(0, 1.0)  # zero events in 1 h: upper bound ~3.69/h
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=1e-3)
