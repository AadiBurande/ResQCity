"""Ward-hour-type panel for the phase-5 forecaster.

The pure functions are checked against brute force on tiny inputs; the checks that need the real
Pune network are skipped when data/processed is absent.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from resqcity.common.config import load_base_config
from resqcity.common.schemas import EventType, GeoPoint, SeverityFeatures
from resqcity.forecasting.config import load_forecasting_config
from resqcity.geo.config import load_geo_config
from resqcity.sim.groundtruth import GroundTruthIncident
from resqcity.sim.panel import lagged, observed_counts, panel_path

IST = ZoneInfo("Asia/Kolkata")
FC = load_forecasting_config()
BASE, GEO = load_base_config(), load_geo_config()
PROCESSED = BASE.paths.resolve("data_processed")


def test_lags_match_brute_force_and_never_see_the_present():
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.4, size=(2, 40, 3)).astype(np.int32)
    out = lagged(counts, [1, 3, 5], citywide_h=4)
    for w in range(counts.shape[0]):
        for t in range(counts.shape[2]):
            for k in (1, 3, 5):
                got = out[f"lag_{k}h"][w, :, t]
                for h in range(counts.shape[1]):
                    want = counts[w, max(0, h - k) : h, t].sum() if h >= k else np.nan
                    if np.isnan(want):
                        assert np.isnan(got[h]), (w, t, k, h)
                    else:
                        assert got[h] == want, (w, t, k, h)
    city = counts.sum(axis=(0, 2))
    for h in range(4, counts.shape[1]):
        assert out["lag_city_4h"][0, h, 0] == city[h - 4 : h].sum()
    assert np.isnan(out["lag_city_4h"][0, 3, 0])


def test_a_spike_only_shows_up_in_later_hours():
    """A leakage guard: an incident at hour h must not appear in any feature of hour h."""
    counts = np.zeros((1, 10, 1), dtype=np.int32)
    counts[0, 5, 0] = 7
    out = lagged(counts, [1, 3], citywide_h=2)
    assert out["lag_1h"][0, 5, 0] == 0 and out["lag_3h"][0, 5, 0] == 0
    assert out["lag_city_2h"][0, 5, 0] == 0
    assert out["lag_1h"][0, 6, 0] == 7
    assert out["lag_3h"][0, 8, 0] == 7 and out["lag_3h"][0, 9, 0] == 0


def _incident(k: int, ward: str, ts: datetime, etype: EventType) -> GroundTruthIncident:
    return GroundTruthIncident(
        id=f"GT-P-{k:04d}",
        onset_ts=ts,
        type=etype,
        node=1,
        location=GeoPoint(lat=18.5, lon=73.85),
        ward_id=ward,
        severity_features=SeverityFeatures(),
        severity_score=0.3,
        severity_class="minor",
        requirements={"ambulance": 0, "police": 1, "fire": 0},
        scenario="history_3y",
        replication=0,
    )


def test_counts_place_every_incident_in_the_right_cell():
    start = datetime(2023, 1, 2, tzinfo=IST)
    hours = [start + timedelta(hours=h) for h in range(5)]
    wards = ["W01", "W02"]
    incidents = [
        _incident(0, "W01", start + timedelta(minutes=10), EventType.ACCIDENT),
        _incident(1, "W01", start + timedelta(minutes=59), EventType.ACCIDENT),
        _incident(2, "W02", start + timedelta(hours=3), EventType.FIRE_SMOKE),
        _incident(3, "W01", start + timedelta(hours=2), EventType.CROWD),  # folds into "other"
        _incident(4, "W03", start + timedelta(hours=1), EventType.ACCIDENT),  # ward not in panel
        _incident(5, "W01", start + timedelta(hours=99), EventType.ACCIDENT),  # beyond horizon
    ]
    counts = observed_counts(incidents, hours, wards, FC)
    types = FC.targets.types
    assert counts.sum() == 4
    assert counts[0, 0, types.index("accident")] == 2
    assert counts[1, 3, types.index("fire_smoke")] == 1
    assert counts[0, 2, types.index("other")] == 1


pytestmark_real = pytest.mark.skipif(
    not (PROCESSED / "synthetic" / panel_path(FC).name).exists(),
    reason="panel not built (uv run python -m resqcity.sim.panel)",
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_parquet(panel_path(FC))


@pytestmark_real
def test_panel_shape_and_dtypes(panel):
    assert set(panel["type"].cat.categories) == set(FC.targets.types)
    assert panel["ts"].dt.tz is not None
    assert len(panel) == panel["ward"].nunique() * panel["ts"].nunique() * len(FC.targets.types)
    assert panel["warmup"].sum() == (
        FC.features.max_lag_h * panel["ward"].nunique() * len(FC.targets.types)
    )


@pytestmark_real
def test_the_analytic_intensity_matches_the_sampled_counts(panel):
    """The oracle column is derived from the generator's parameters, the counts are sampled from
    them; over three years they must agree to within sampling error."""
    live = panel[~panel["warmup"]]
    observed, expected = live["count"].sum(), live["intensity"].sum()
    assert expected == pytest.approx(observed, rel=0.05)
    assert abs(observed - expected) < 3.0 * np.sqrt(expected)  # Poisson tolerance
    for name, group in live.groupby("type", observed=True):
        assert group["intensity"].sum() == pytest.approx(group["count"].sum(), rel=0.12), name


@pytestmark_real
def test_the_rain_forecast_is_informative_but_wrong(panel):
    hourly = panel.groupby("ts", observed=True)[["rain_mm_h", "rain_fc"]].first()
    assert hourly["rain_fc"].corr(hourly["rain_mm_h"]) > 0.5  # useful
    assert not np.allclose(hourly["rain_fc"], hourly["rain_mm_h"])  # but not the truth
    dry = hourly["rain_mm_h"] == 0
    assert (hourly.loc[dry, "rain_fc"] > 0).mean() > 0.01  # sometimes predicts rain on a dry hour
    assert (hourly.loc[~dry, "rain_fc"] == 0).mean() > 0.01  # and misses real rain


@pytestmark_real
def test_the_realistic_feature_set_excludes_the_generators_own_inputs():
    realistic = set(FC.feature_sets["realistic"])
    assert "hotspot_mass" not in realistic and "rain_true" not in realistic
    assert "rain_forecast" in realistic
    assert "hotspot_mass" in FC.feature_sets["informed"]
