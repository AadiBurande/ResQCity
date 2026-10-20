"""Base incident rate / hour profile derivation and agreement with configs/incidents.yaml."""

import json

import pytest

from resqcity.common.config import CONFIG_DIR, load_base_config, load_model
from resqcity.sim.base_rate import BaseRateConfig, derive, fire_calls_per_year, hourly_profile
from resqcity.sim.generator import load_incidents_config

CFG = load_model(BaseRateConfig, CONFIG_DIR / "base_rate.yaml")
BASE = load_base_config()
EXTRACT = {
    "pune_million_plus_city": {"accidents_2022": 300, "killed_2022": 30},
    "all_million_plus_cities": {"accidents_2022": 1000, "killed_2022": 200},
    "pune_weather_2022": {"rainy": 30},
    "maharashtra_time_of_day_2022": {
        "accidents_by_interval": {
            "06-09": 3,
            "09-12": 3,
            "12-15": 3,
            "15-18": 3,
            "18-21": 6,
            "21-24": 3,
            "00-03": 3,
            "03-06": 0,
        }
    },
}


def _csv(tmp_path):
    p = tmp_path / "fire.csv"
    p.write_text(
        "S. No.,Type of Call,2018,2019,2020\n1,Fire,100,300,5\n2,Garbage,50,50,50\n",
        encoding="utf-8",
    )
    return p


def test_fire_calls_mean_over_selected_years(tmp_path):
    assert fire_calls_per_year(_csv(tmp_path), ["Fire"], [2018, 2019]) == 200.0


def test_hourly_profile_spreads_buckets_and_has_mean_one():
    prof = hourly_profile(EXTRACT["maharashtra_time_of_day_2022"]["accidents_by_interval"])
    assert len(prof) == 24 and sum(prof) / 24 == pytest.approx(1.0, abs=1e-3)
    assert prof[18] == prof[19] == prof[20] == pytest.approx(2 * prof[6], abs=1e-3)
    assert prof[3] == prof[4] == prof[5] == 0.0


def test_extra_share_is_added_on_top(tmp_path):
    cfg = CFG.model_copy(
        update={"fire_calls": CFG.fire_calls.model_copy(update={"years": [2018, 2019]})}
    )
    r = derive(cfg, _csv(tmp_path), EXTRACT)
    assert r["accident_plus_fire_per_day"] == pytest.approx((200 + 300) / 365)
    assert r["incidents_per_day"] * (1 - 0.03) == pytest.approx(r["accident_plus_fire_per_day"])
    total = r["type_share_accident"] + r["type_share_fire"] + r["type_share_crowd_other"]
    assert total == pytest.approx(1)
    assert r["pune_killed_per_100_accidents"] == pytest.approx(10.0)
    assert r["million_plus_cities_killed_per_100_accidents"] == pytest.approx(20.0)


def test_incidents_config_matches_derivation():
    out = BASE.paths.resolve("results") / "calibration" / "base_rate.json"
    if not out.exists():
        pytest.skip("results/calibration/base_rate.json not generated")
    r = json.loads(out.read_text(encoding="utf-8"))
    inc = load_incidents_config()
    assert inc.base_rate_per_day.value == pytest.approx(r["incidents_per_day"], abs=0.005)
    assert inc.type_mix.value["accident"] == pytest.approx(r["type_share_accident"], abs=0.001)
    assert inc.type_mix.value["fire_smoke"] == pytest.approx(r["type_share_fire"], abs=0.001)
    assert inc.hour_profile.value == pytest.approx(r["hour_profile"], abs=1e-4)
