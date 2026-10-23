"""Exploratory v5 analyses (D-027 re-audit): tiny instances checked by brute force."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from resqcity.vision.eval import final_v5
from resqcity.vision.eval.explore_v5 import (
    armed_time,
    chance_detection,
    content_free_detected,
    detection_time,
    rate_summary,
    uniform_share_within,
)
from resqcity.vision.eval.final_v5 import export, temporal_score
from resqcity.vision.eval.metrics import (
    ACCIDENT_PRE_TOLERANCE_S,
    DETECTION_GRACE_S,
    SECONDS_PER_HOUR,
)


def _armed_brute(alarms: list[float], start: float, end: float, cooldown: float) -> float:
    """Armed seconds by a 1 ms grid: armed unless within [a, a + cooldown) of an earlier alarm."""
    grid = np.arange(start, end, 0.001)
    silent = np.zeros_like(grid, dtype=bool)
    for a in alarms:
        if start <= a < end:
            silent |= (grid >= a) & (grid < a + cooldown)
    return float((~silent).sum()) * 0.001


@pytest.mark.parametrize(
    ("alarms", "start", "end", "cooldown"),
    [
        ([], 6.0, 20.0, 60.0),
        ([6.0], 6.0, 20.0, 60.0),
        ([8.0, 70.0], 6.0, 100.0, 60.0),
        ([7.0, 9.5], 0.0, 15.0, 2.0),
        ([3.0], 6.0, 20.0, 60.0),  # before start: not counted
    ],
)
def test_armed_time_matches_brute_force(
    alarms: list[float], start: float, end: float, cooldown: float
) -> None:
    n, armed = armed_time(alarms, start, end, cooldown)
    assert n == sum(start <= a < end for a in alarms)
    assert armed == pytest.approx(_armed_brute(alarms, start, end, cooldown), abs=0.01)


def test_detection_time_first_alarm_in_interval() -> None:
    onset = 10.0
    lo, hi = onset - ACCIDENT_PRE_TOLERANCE_S, onset + DETECTION_GRACE_S
    assert detection_time([lo - 0.5, lo, hi], onset) == lo
    assert detection_time([hi + 0.1], onset) is None
    assert detection_time([], onset) is None


def test_content_free_baseline() -> None:
    warm_up = 6.0
    onset = np.array([3.0, 6.5, 7.0, 7.5, 20.0, 5.0])
    duration = np.array([15.0, 15.0, 15.0, 15.0, 30.0, 5.5])
    got = content_free_detected(onset, duration, warm_up)
    want = [
        (o - ACCIDENT_PRE_TOLERANCE_S <= warm_up <= o + DETECTION_GRACE_S) and d >= warm_up
        for o, d in zip(onset, duration, strict=True)
    ]
    assert got.tolist() == want
    assert got.tolist() == [True, True, True, False, False, False]


def test_uniform_share_within_by_simulation() -> None:
    rng = np.random.default_rng(0)
    for armed_s in (1.0, 3.0, 10.0):
        t = rng.uniform(0.0, armed_s, 200_000)  # seconds before t* - 1 s
        sim = np.mean(t + ACCIDENT_PRE_TOLERANCE_S <= 3.0)
        assert uniform_share_within(3.0, armed_s) == pytest.approx(sim, abs=0.005)


def test_rate_summary_units() -> None:
    units = pd.DataFrame(
        {"alarms": [2, 0, 1], "armed_s": [1800.0, 3600.0, 1800.0], "alarm_at_warm_up": [1, 0, 0]}
    )
    r = rate_summary(units, cooldown=60.0)
    assert r["alarms"] == 3 and r["armed_hours"] == pytest.approx(2.0)
    assert r["rate_per_armed_hour"] == pytest.approx(1.5)
    assert r["units_with_alarm"] == 2 and r["units_with_2plus_alarms"] == 1
    assert r["units_alarming_at_first_possible_instant"] == 1
    lam = 1.5
    assert r["long_run_per_camera_hour_with_cooldown"] == pytest.approx(
        lam / (1 + lam * 60.0 / SECONDS_PER_HOUR)
    )


def test_constant_guess_temporal_score() -> None:
    truth = np.array([4.0, 5.0, 6.0])
    got = temporal_score(np.full_like(truth, 5.0), truth, 1.0)
    assert got == pytest.approx((2 * np.exp(-0.5) + 1) / 3)


def test_export_only_puts_simulator_defaults_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final_v5, "PROJECT_ROOT", tmp_path)
    out = tmp_path / "v5"
    out.mkdir()
    block = {"onset_detected": 6, "onset_videos": 10}
    summary = {
        **block,
        "onset_detection_rate": 0.6,
        "onset_detection_rate_ci95_lo": 0.3,
        "onset_detection_rate_ci95_hi": 0.8,
        "n_delays": 6,
        "delay_median_s": 2.0,
        "road_false_alarms": 4,
        "road_hours": 0.1,
        "road_false_alarms_per_hour": 40.0,
    }
    road = {
        "road_false_alarms": 4,
        "road_hours": 0.1,
        "road_false_alarms_per_hour": 40.0,
        "road_fa_poisson_ci95_lo": 10.0,
        "road_fa_poisson_ci95_hi": 90.0,
        "road_fa_cluster_ci95_lo": 12.0,
        "road_fa_cluster_ci95_hi": 80.0,
    }
    report = {
        "deployed": {"operating_point": {"threshold": 0.3}, "verifier": "none", "tau": None},
        "detection_rule": "rule",
        "accident_holdout": summary,
        "road_false_alarms_pooled": road,
        "e2b_ua_detrac_normal_road": {"road_false_alarms_per_hour": 30.0},
        "e3b_delay_with_lead_time": {"delay_median_s": 2.5},
        "by_region": {"us": block, "non_us": {"onset_detected": 1, "onset_videos": 3}},
    }
    rates = {
        "rate_per_armed_hour": 50.0,
        "cluster_bootstrap_ci95": [30.0, 70.0],
    }
    explore = {
        "simulator_defaults": {"detection_probability_when_armed": 0.5},
        "false_alarms_armed": {
            "definition": "d",
            "accident_pre_impact": rates,
            "ua_detrac_normal_traffic": rates,
            "accident_pre_impact_reading": "r",
        },
        "detection_clustered": {"cluster_bootstrap_ci95": [0.4, 0.7]},
    }
    (out / "exploratory.json").write_text(json.dumps(explore), encoding="utf-8")
    (out / "stacker.json").write_text("{}", encoding="utf-8")
    export(tmp_path, out, report, "v5_vjepa_mil_all", "")
    doc = yaml.safe_load((tmp_path / "sim_inputs.yaml").read_text(encoding="utf-8"))
    assert next(iter(doc)) == "simulator_defaults"
    assert doc["false_alarm_model"]["default"] == "ua_detrac_normal_traffic"
    assert "3/13 = 23.1% outside the US" in doc["caveat"]
    assert (tmp_path / "deployed_detector.json").exists()


def test_chance_detection_by_simulation() -> None:
    rng = np.random.default_rng(1)
    rate = 55.0
    window_h = (ACCIDENT_PRE_TOLERANCE_S + DETECTION_GRACE_S) / SECONDS_PER_HOUR
    sim = np.mean(rng.poisson(rate * window_h, 200_000) > 0)
    assert chance_detection(rate) == pytest.approx(sim, abs=0.005)
    assert chance_detection(0.0) == 0.0
