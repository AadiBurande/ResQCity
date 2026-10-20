"""Base incident rate derivation (D-018 / D-025) and its agreement with configs/incidents.yaml."""

import pytest

from resqcity.common.config import CONFIG_DIR, load_base_config, load_model
from resqcity.sim.base_rate import BaseRateConfig, derive, fire_calls_per_year
from resqcity.sim.generator import load_incidents_config

CFG = load_model(BaseRateConfig, CONFIG_DIR / "base_rate.yaml")


def _csv(tmp_path):
    p = tmp_path / "fire.csv"
    p.write_text(
        "S. No.,Type of Call,2018,2019,2020\n1,Fire,100,300,5\n2,Garbage,50,50,50\n",
        encoding="utf-8",
    )
    return p


def test_fire_calls_mean_over_selected_years(tmp_path):
    assert fire_calls_per_year(_csv(tmp_path), ["Fire"], [2018, 2019]) == 200.0


def test_extra_share_is_added_on_top(tmp_path):
    cfg = CFG.model_copy(
        update={"fire_calls": CFG.fire_calls.model_copy(update={"years": [2018, 2019]})}
    )
    r = derive(cfg, _csv(tmp_path))
    accidents = 334 / 0.365
    assert r["accident_plus_fire_per_day"] == pytest.approx((200 + accidents) / 365)
    assert r["incidents_per_day"] * (1 - 0.03) == pytest.approx(r["accident_plus_fire_per_day"])
    assert r["type_share_accident"] + r["type_share_fire"] + r[
        "type_share_crowd_other"
    ] == pytest.approx(1)


def test_incidents_config_matches_derivation():
    csv_path = load_base_config().paths.resolve("data_raw") / CFG.fire_calls.raw_file
    if not csv_path.exists():
        pytest.skip("PMC fire-call CSV not downloaded (run resqcity.sim.base_rate)")
    r = derive(CFG, csv_path)
    inc = load_incidents_config()
    assert inc.base_rate_per_day.value == pytest.approx(r["incidents_per_day"], abs=0.005)
    assert inc.type_mix.value["accident"] == pytest.approx(r["type_share_accident"], abs=0.001)
    assert inc.type_mix.value["fire_smoke"] == pytest.approx(r["type_share_fire"], abs=0.001)
