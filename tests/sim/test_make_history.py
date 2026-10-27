"""Generated history scenario for the phase-5 forecaster."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from resqcity.forecasting.config import load_forecasting_config
from resqcity.geo.scenario import load_scenario
from resqcity.sim.make_history import (
    _days_in_month,
    build,
    daily_rain_mm,
    festival_windows,
)

IST = ZoneInfo("Asia/Kolkata")
HISTORY = load_forecasting_config().history
MONTHLY_MM = HISTORY.monthly_mm.value
START = datetime.combine(HISTORY.start, datetime.min.time(), tzinfo=IST)
DAYS = HISTORY.days


def test_daily_rain_is_reproducible_and_matches_the_monthly_profile():
    a = daily_rain_mm(HISTORY, START, DAYS, 7)
    assert a == daily_rain_mm(HISTORY, START, DAYS, 7)
    assert a != daily_rain_mm(HISTORY, START, DAYS, 8)
    dates = [START.date() + timedelta(days=d) for d in range(DAYS)]
    for month in (1, 7, 12):  # a dry, a peak-monsoon and a dry month
        for year in (2023, 2024, 2025):
            days = [i for i, d in enumerate(dates) if (d.year, d.month) == (year, month)]
            # the first and last months of the horizon are partial, so the target is pro-rata
            target = MONTHLY_MM[month] * len(days) / _days_in_month(year, month)
            got = sum(a[i] for i in days)
            # a dry month may draw no wet day at all and then stays completely dry
            assert got == pytest.approx(target, rel=1e-3, abs=0.02) or got == 0.0, (year, month)
    annual = sum(a) / (DAYS / 365.25)
    assert annual == pytest.approx(sum(MONTHLY_MM.values()), rel=0.05)


def test_rain_is_not_a_function_of_the_calendar():
    """Otherwise a model given only the date would know the rain exactly and the phase-5
    informed-vs-realistic comparison would collapse."""
    a = np.asarray(daily_rain_mm(HISTORY, START, DAYS, 11))
    dates = [START.date() + timedelta(days=d) for d in range(DAYS)]
    july = {
        y: a[[i for i, d in enumerate(dates) if (d.year, d.month) == (y, 7)]] for y in (2023, 2024)
    }
    assert not np.array_equal(july[2023][:30], july[2024][:30])


def test_dry_season_is_drier_than_the_monsoon():
    a = np.asarray(daily_rain_mm(HISTORY, START, DAYS, 3))
    dates = [START.date() + timedelta(days=d) for d in range(DAYS)]
    monsoon = a[[i for i, d in enumerate(dates) if d.month in (6, 7, 8, 9)]]
    dry = a[[i for i, d in enumerate(dates) if d.month in (12, 1, 2, 3)]]
    assert monsoon.sum() > 20 * dry.sum()


def test_festival_windows_once_per_year_and_inside_the_horizon():
    windows = festival_windows(HISTORY, START, DAYS)
    effects = {w["effect"] for w in windows}
    assert effects == {"slowed", "closed"}
    all_days = [d for w in windows for d in w["days"]]
    assert all(0 <= d < DAYS for d in all_days)
    assert len(all_days) == len(set(all_days))
    closed = sorted(d for w in windows if w["effect"] == "closed" for d in w["days"])
    assert len(closed) == 6  # a two-day immersion in each of the three years
    years = {(START.date() + timedelta(days=d)).year for d in closed}
    assert years == {2023, 2024, 2025}


def test_the_profile_comes_from_config_not_from_code():
    """The climate and festival assumptions must be in configs/forecasting.yaml, so that
    `find_unverified` lists them (research integrity rule 3). Guards the phase-5 audit fix."""
    import resqcity.sim.make_history as mh

    paths = {"IST", "OUT", "SOURCE_SCENARIO", "CONFIG_DIR"}  # locations, not parameters
    assert not [n for n in dir(mh) if n.isupper() and n not in paths]
    assert HISTORY.days == DAYS and START.weekday() == 0


def test_days_in_month():
    assert _days_in_month(2024, 2) == 29  # leap year
    assert _days_in_month(2023, 2) == 28
    assert _days_in_month(2023, 12) == 31


def test_build_is_loadable_and_hour_shape_is_normalised():
    doc = build(HISTORY, datetime(2023, 1, 2, tzinfo=IST), 364, 5)
    assert sum(doc["rain"]["hourly_mm_h"]) == pytest.approx(1.0, abs=1e-4)
    assert len(doc["rain"]["day_factors"]) == 364
    assert doc["synthetic"] is True


def test_committed_scenario_loads_and_is_synthetic():
    sc = load_scenario("history_3y")
    assert sc.synthetic and sc.days == DAYS
    assert len(sc.rain.day_factors) == sc.days  # not a 365-day cycle
    assert sc.start.date() == date(2023, 1, 2) and sc.start.weekday() == 0
    assert sc.festival is not None and sc.flooding is None
