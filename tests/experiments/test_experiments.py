"""Tests for the experiments module (phase 10)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from resqcity.experiments.metrics import (
    _gini,
    aggregate_metrics,
    compute_incident_metrics,
)
from resqcity.experiments.stats import (
    holm_bonferroni,
    non_inferiority_test,
    paired_test,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

IST = UTC  # use UTC for test simplicity

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=IST)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _make_row(
    incident_id: str = "I1",
    policy: str = "B0",
    replication: int = 0,
    onset: datetime | None = None,
    first_det: datetime | None = None,
    first_call: datetime | None = None,
    decision: datetime | None = None,
    arrival_amb: datetime | None = None,
    arrival_pol: datetime | None = None,
    arrival_fire: datetime | None = None,
    last_arr: datetime | None = None,
    ward_id: str = "W01",
    inc_type: str = "accident",
    is_false_alarm: bool = False,
    in_warmup: bool = False,
    shortfall: int = 0,
) -> dict:
    return {
        "incident_id": incident_id,
        "policy": policy,
        "replication": replication,
        "onset_ts": onset,
        "first_detection_ts": first_det,
        "first_call_ts": first_call,
        "decision_ts": decision,
        "arrival_ambulance_ts": arrival_amb,
        "arrival_police_ts": arrival_pol,
        "arrival_fire_ts": arrival_fire,
        "last_required_arrival_ts": last_arr,
        "ward_id": ward_id,
        "type": inc_type,
        "is_false_alarm": is_false_alarm,
        "in_warmup": in_warmup,
        "shortfall_units": shortfall,
    }


# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------


class TestGini:
    def test_equal_values(self) -> None:
        assert _gini(np.array([10.0, 10.0, 10.0, 10.0])) == pytest.approx(0.0)

    def test_max_inequality(self) -> None:
        g = _gini(np.array([0.0, 0.0, 0.0, 100.0]))
        assert g > 0.5

    def test_empty(self) -> None:
        assert _gini(np.array([])) == 0.0


class TestIncidentMetrics:
    def test_basic_timing(self) -> None:
        row = _make_row(
            onset=T0,
            first_call=_t(60),
            decision=_t(120),
            arrival_amb=_t(300),
            arrival_pol=_t(360),
            last_arr=_t(360),
        )
        m = compute_incident_metrics(row)
        assert m.detection_delay_s == pytest.approx(60.0)
        assert m.detection_to_dispatch_s == pytest.approx(60.0)
        assert m.response_time_ambulance_s == pytest.approx(300.0)
        assert m.response_time_police_s == pytest.approx(360.0)
        assert m.time_to_last_required_arrival_s == pytest.approx(360.0)
        assert m.within_8min is True  # 360 s < 480 s

    def test_camera_before_call(self) -> None:
        row = _make_row(onset=T0, first_det=_t(10), first_call=_t(60))
        m = compute_incident_metrics(row)
        assert m.detection_delay_s == pytest.approx(10.0)

    def test_no_onset(self) -> None:
        row = _make_row(onset=None, first_call=_t(60))
        m = compute_incident_metrics(row)
        assert m.detection_delay_s is None

    def test_false_alarm(self) -> None:
        row = _make_row(is_false_alarm=True)
        m = compute_incident_metrics(row)
        assert m.is_false_alarm is True

    def test_first_to_last_gap(self) -> None:
        row = _make_row(
            onset=T0,
            arrival_amb=_t(200),
            arrival_pol=_t(350),
            arrival_fire=_t(500),
            last_arr=_t(500),
        )
        m = compute_incident_metrics(row)
        assert m.first_to_last_gap_s == pytest.approx(300.0)

    def test_above_15min(self) -> None:
        row = _make_row(onset=T0, last_arr=_t(1000))
        m = compute_incident_metrics(row)
        assert m.within_8min is False
        assert m.within_15min is False


class TestAggregateMetrics:
    def _make_df(self, n: int = 10, policy: str = "B0") -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append(
                {
                    "incident_id": f"I{i}",
                    "policy": policy,
                    "replication": i % 3,
                    "ward_id": f"W{i % 5:02d}",
                    "incident_type": "accident",
                    "is_false_alarm": False,
                    "in_warmup": False,
                    "time_to_last_required_arrival_s": 300.0 + i * 10,
                    "detection_delay_s": 30.0 + i,
                    "detection_to_dispatch_s": 20.0 + i,
                    "response_time_ambulance_s": 200.0 + i * 5,
                    "response_time_police_s": 250.0 + i * 5,
                    "response_time_fire_s": None,
                    "first_to_last_gap_s": 50.0 + i * 2,
                    "expected_survival": 0.85 - i * 0.01,
                    "within_8min": i < 7,
                    "within_15min": True,
                    "shortfall_units": 0,
                }
            )
        return pd.DataFrame(rows)

    def test_basic_aggregation(self) -> None:
        df = self._make_df()
        agg = aggregate_metrics(df, "normal", "B0")
        assert agg.n_incidents == 10
        assert agg.primary_mean_s > 0
        assert agg.pct_within_15min == pytest.approx(100.0)
        assert len(agg.per_rep_primary_means) > 0

    def test_excludes_warmup_and_false_alarms(self) -> None:
        df = self._make_df(n=5)
        df.loc[0, "in_warmup"] = True
        df.loc[1, "is_false_alarm"] = True
        agg = aggregate_metrics(df, "normal", "B0")
        assert agg.n_incidents == 3


# ---------------------------------------------------------------------------
# stats tests
# ---------------------------------------------------------------------------


class TestPairedTest:
    def test_identical_values(self) -> None:
        a = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        r = paired_test(a, a, "primary", "B0", "FULL")
        assert r.mean_diff == pytest.approx(0.0)

    def test_clear_difference(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(500, 10, 30)
        b = rng.normal(400, 10, 30)
        r = paired_test(a, b, "primary", "B0", "FULL")
        assert r.mean_diff > 50
        assert r.p_value < 0.01

    def test_too_few(self) -> None:
        a = np.array([100.0, 200.0])
        b = np.array([90.0, 190.0])
        r = paired_test(a, b, "primary", "B0", "FULL")
        assert "Too few" in r.reading


class TestHolmBonferroni:
    def test_correction(self) -> None:
        rng = np.random.default_rng(42)
        results = []
        for name_a, name_b in [("B0", "B1"), ("B0", "B2"), ("B1", "B2")]:
            a = rng.normal(500, 20, 30)
            b = rng.normal(480, 20, 30)
            r = paired_test(a, b, "primary", name_a, name_b)
            results.append(r)
        corrected = holm_bonferroni(results)
        for r in corrected:
            assert r.p_adjusted >= r.p_value or r.p_adjusted == pytest.approx(r.p_value)


class TestNonInferiority:
    def test_clearly_non_inferior(self) -> None:
        full = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.10, 0.11, 0.10])
        base = np.array([0.12, 0.13, 0.11, 0.12, 0.14, 0.12, 0.13, 0.12])
        r = non_inferiority_test(full, base, delta=0.02)
        assert r.non_inferior is True

    def test_clearly_inferior(self) -> None:
        full = np.array([0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31, 0.30])
        base = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.10, 0.11, 0.10])
        r = non_inferiority_test(full, base, delta=0.02)
        assert r.non_inferior is False
