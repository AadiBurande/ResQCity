"""Forecasters: determinism, persistence and the phase-10 noise knob (phase 5, D-030).

The brief's acceptance criterion for this phase is that a model "reloads deterministically", so
the LightGBM test below asserts predictions are bit-identical after a save/load round trip.
"""

import numpy as np
import pandas as pd
import pytest

from resqcity.forecasting.config import load_forecasting_config
from resqcity.forecasting.metrics import poisson_deviance, split, top_k_hit_rate
from resqcity.forecasting.models import (
    Climatology,
    LightGbmPoisson,
    NoisyForecaster,
    UniformRate,
    feature_columns,
)

CFG = load_forecasting_config()
WARDS = ["W01", "W02", "W03"]
TYPES = CFG.targets.types


def _panel(hours: int = 24 * 40, seed: int = 0) -> pd.DataFrame:
    """A small panel with the same columns as sim.panel, and a real ward x hour signal."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-02", tz="UTC")
    ts = pd.date_range(start, periods=hours, freq="h")
    rows = pd.MultiIndex.from_product([WARDS, ts, TYPES], names=["ward", "ts", "type"])
    frame = rows.to_frame(index=False)
    local = frame["ts"].dt.tz_convert("Asia/Kolkata")
    frame["hour"] = local.dt.hour.astype(np.int8)
    frame["weekday"] = local.dt.weekday.astype(np.int8)
    frame["month"] = local.dt.month.astype(np.int8)
    frame["festival"] = False
    ward_rate = {"W01": 0.05, "W02": 0.02, "W03": 0.005}
    rate = frame["ward"].map(ward_rate) * (1.0 + 0.8 * np.sin(frame["hour"] / 24 * 2 * np.pi))
    frame["intensity"] = rate.astype(np.float32)
    frame["count"] = rng.poisson(rate).astype(np.int16)
    frame["rain_mm_h"] = rng.gamma(1.0, 0.2, len(frame)).astype(np.float32)
    frame["rain_fc"] = (frame["rain_mm_h"] * rng.lognormal(0, 0.3, len(frame))).astype(np.float32)
    frame["hotspot_mass"] = frame["ward"].map(ward_rate).astype(np.float32)
    for k in CFG.features.ward_lags_h:
        frame[f"lag_{k}h"] = rng.poisson(0.05, len(frame)).astype(np.float32)
    frame[f"lag_city_{CFG.features.citywide_lag_h}h"] = rng.poisson(2.0, len(frame)).astype(
        np.float32
    )
    frame["warmup"] = False
    for col in ("ward", "type"):
        frame[col] = frame[col].astype("category")
    return frame


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_uniform_knows_only_the_mean(panel):
    model = UniformRate.fit(panel)
    pred = model.predict(panel)
    assert np.allclose(pred, panel["count"].mean())
    assert pred.sum() == pytest.approx(panel["count"].sum())


def test_climatology_matches_a_hand_computed_rate(panel):
    model = Climatology.fit(panel)
    pred = model.predict(panel)
    ward, kind = panel["ward"].astype(str), panel["type"].astype(str)
    how = panel["weekday"].astype(int) * 24 + panel["hour"].astype(int)
    row = 0
    want = (
        panel["count"][(ward == ward.iloc[row]) & (kind == kind.iloc[row])].mean()
        * panel["count"][(how == how.iloc[row]) & (kind == kind.iloc[row])].mean()
        / panel["count"][kind == kind.iloc[row]].mean()
    )
    assert pred[row] == pytest.approx(want)
    assert pred.sum() == pytest.approx(panel["count"].sum(), rel=0.02)  # total is preserved


def test_climatology_beats_uniform_at_ranking_wards(panel):
    """The point of the baseline: it knows which wards are busy."""
    uniform, clim = UniformRate.fit(panel), Climatology.fit(panel)
    assert top_k_hit_rate(panel, clim.predict(panel), 1) > top_k_hit_rate(
        panel, uniform.predict(panel), 1
    )


def test_lightgbm_reloads_deterministically(panel, tmp_path):
    """Phase-5 acceptance criterion: a saved model must reload to identical predictions."""
    parts = {"train": panel.iloc[: len(panel) // 2], "valid": panel.iloc[len(panel) // 2 :]}
    model = LightGbmPoisson.fit(CFG, "realistic", parts["train"], parts["valid"])
    before = model.predict(panel)
    model.save(tmp_path / model.name)
    reloaded = LightGbmPoisson.load(tmp_path / model.name)
    assert np.array_equal(before, reloaded.predict(panel))  # bit-identical, not approx
    assert reloaded.columns == model.columns
    assert reloaded.best_iteration == model.best_iteration
    again = LightGbmPoisson.fit(CFG, "realistic", parts["train"], parts["valid"])
    assert np.array_equal(before, again.predict(panel))  # and refitting is deterministic too


def test_lightgbm_uses_only_the_columns_its_feature_set_allows():
    realistic = feature_columns(CFG, "realistic")
    informed = feature_columns(CFG, "informed")
    assert "hotspot_mass" not in realistic and "rain_mm_h" not in realistic
    assert "rain_fc" in realistic
    assert "hotspot_mass" in informed and "rain_mm_h" in informed
    assert all(f"lag_{k}h" in realistic for k in CFG.features.ward_lags_h)


def test_the_noise_knob_is_reproducible_and_mean_preserving(panel):
    inner = Climatology.fit(panel)
    base = inner.predict(panel)
    quiet = NoisyForecaster(inner, 0.0, 7).predict(panel)
    assert np.array_equal(quiet, base)  # sigma 0 leaves the forecaster untouched
    noisy = NoisyForecaster(inner, 0.8, 7)
    assert np.array_equal(noisy.predict(panel), noisy.predict(panel))  # reproducible
    assert not np.allclose(noisy.predict(panel), base)
    assert NoisyForecaster(inner, 0.8, 7).predict(panel).mean() == pytest.approx(
        base.mean(), rel=0.05
    )  # lognormal is centred so the total demand is unchanged
    worse = NoisyForecaster(inner, 1.5, 7).predict(panel)
    assert poisson_deviance(panel["count"].to_numpy(float), worse) > poisson_deviance(
        panel["count"].to_numpy(float), base
    )


def test_the_oracle_returns_the_generators_own_intensity(panel):
    from resqcity.sim.oracle_forecast import OracleForecast

    oracle = OracleForecast()
    assert np.array_equal(oracle.predict(panel), panel["intensity"].to_numpy(float))
    with pytest.raises(KeyError):
        oracle.predict(panel.drop(columns=["intensity"]))


def test_split_is_time_based_and_contiguous(panel):
    cfg = CFG.model_copy(
        update={"split": CFG.split.model_copy(update={"test_weeks": 1, "validation_weeks": 1})}
    )
    parts = split(panel, cfg)
    assert parts["train"]["ts"].max() < parts["valid"]["ts"].min()
    assert parts["valid"]["ts"].max() < parts["test"]["ts"].min()
    assert sum(len(p) for p in parts.values()) == len(panel)
