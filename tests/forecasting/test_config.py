"""configs/forecasting.yaml: the guards that keep the phase-5 comparison honest.

The feature-set guards matter because the whole point of the `realistic` model is that it is NOT
handed the generator's own inputs (D-030). The `find_unverified` test matters because the
synthetic history's climate and festival parameters are assumptions, and the project's integrity
rule is that every assumption is listed, not that it is right.
"""

import pytest
from pydantic import ValidationError

from resqcity.common.config import SourcedValue, find_unverified
from resqcity.forecasting.config import ForecastingConfig, load_forecasting_config

CFG = load_forecasting_config()


def test_every_assumed_parameter_is_flagged() -> None:
    """Exact list, so adding a parameter without a source cannot slip in unnoticed."""
    assert set(find_unverified(CFG)) == {
        "history.monthly_mm",
        "history.wet_day",
        "history.hourly_shape",
        "history.festival",
        "rain_forecast.sigma",
        "rain_forecast.p_mixup",
        "noise_knob.sigma",
    }


def test_illustrative_is_not_a_citation() -> None:
    """ILLUSTRATIVE must be reported like UNVERIFIED: the festival date is not sourced."""
    assert CFG.history.festival.source == "ILLUSTRATIVE"
    assert not CFG.history.festival.verified
    assert SourcedValue[int](value=1, source="Smith et al. 2020").verified


def test_the_history_profile_lives_in_config_not_in_code() -> None:
    """make_history must read these, so a reviewer can change them without touching Python."""
    assert CFG.history.days == 1092  # whole weeks, so weekday structure is balanced
    assert sorted(CFG.history.monthly_mm.value) == list(range(1, 13))
    assert len(CFG.history.hourly_shape.value) == 24
    monsoon = sum(CFG.history.monthly_mm.value[m] for m in (6, 7, 8, 9))
    assert monsoon / sum(CFG.history.monthly_mm.value.values()) > 0.75  # seasonal, not flat


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hourly_shape", [1.0] * 23, "24 entries"),
        ("monthly_mm", dict.fromkeys(range(1, 12), 1.0), "one entry per month"),
    ],
)
def test_a_malformed_history_profile_is_rejected(field: str, value: object, message: str) -> None:
    raw = CFG.model_dump()
    raw["history"][field]["value"] = value
    with pytest.raises(ValidationError, match=message):
        ForecastingConfig.model_validate(raw)


def test_the_realistic_model_may_not_see_the_generators_own_inputs() -> None:
    for leak, message in (("hotspot_mass", "hotspot mass"), ("rain_true", "true rain")):
        raw = CFG.model_dump()
        raw["feature_sets"]["realistic"] = [*CFG.feature_sets["realistic"], leak]
        with pytest.raises(ValidationError, match=message):
            ForecastingConfig.model_validate(raw)


def test_the_informed_model_is_the_one_that_sees_them() -> None:
    informed = set(CFG.feature_sets["informed"])
    assert {"rain_true", "hotspot_mass"} <= informed
    assert "rain_forecast" not in informed


def test_lags_cannot_reach_the_hour_being_predicted() -> None:
    raw = CFG.model_dump()
    raw["features"]["ward_lags_h"] = [0, 24]
    with pytest.raises(ValidationError, match="at least 1 hour"):
        ForecastingConfig.model_validate(raw)


def test_the_config_declares_itself_synthetic() -> None:
    assert CFG.synthetic
    raw = CFG.model_dump()
    raw["synthetic"] = False
    with pytest.raises(ValidationError, match="synthetic must be true"):
        ForecastingConfig.model_validate(raw)
