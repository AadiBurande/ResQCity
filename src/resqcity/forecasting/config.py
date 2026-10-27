"""Typed config of the ward-hour demand forecaster (configs/forecasting.yaml; phase 5, D-030)."""

from datetime import date
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from resqcity.common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from resqcity.common.schemas import EventType

OTHER = "other"
HOURS_PER_DAY = 24


class Targets(ConfigModel):
    types: list[str] = Field(min_length=1)
    fold_into_other: list[EventType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if OTHER not in self.types:
            raise ValueError(f"targets.types must contain {OTHER!r} to absorb the rare types")
        known = {t.value for t in EventType}
        if unknown := set(self.types) - known:
            raise ValueError(f"unknown target types: {sorted(unknown)}")
        return self

    def of(self, event_type: EventType) -> str:
        """The forecast target an incident of this type counts towards."""
        if event_type in self.fold_into_other or event_type.value not in self.types:
            return OTHER
        return event_type.value


class WetDay(ConfigModel):
    p_per_mm: float = Field(gt=0)
    p_min: float = Field(gt=0, lt=1)
    p_max: float = Field(gt=0, le=1)
    gamma_shape: float = Field(gt=0)


class Festival(ConfigModel):
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    evenings: int = Field(ge=1)


class History(ConfigModel):
    """Parameters of the synthetic history written by ``resqcity.sim.make_history``."""

    start: date
    weeks: int = Field(ge=1)
    rain_seed: int
    monthly_mm: SourcedValue[dict[int, float]]
    wet_day: SourcedValue[WetDay]
    hourly_shape: SourcedValue[list[float]]
    festival: SourcedValue[Festival]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if sorted(self.monthly_mm.value) != list(range(1, 13)):
            raise ValueError("history.monthly_mm needs one entry per month 1-12")
        if len(self.hourly_shape.value) != HOURS_PER_DAY:
            raise ValueError(f"history.hourly_shape needs {HOURS_PER_DAY} entries")
        if self.weeks * 7 % 7:  # pragma: no cover - whole weeks by construction
            raise ValueError("history must span whole weeks")
        wet = self.wet_day.value
        if wet.p_min > wet.p_max:
            raise ValueError("history.wet_day.p_min must not exceed p_max")
        return self

    @property
    def days(self) -> int:
        return self.weeks * 7


class Split(ConfigModel):
    test_weeks: int = Field(ge=1)
    validation_weeks: int = Field(ge=1)


class Features(ConfigModel):
    ward_lags_h: list[int] = Field(min_length=1)
    citywide_lag_h: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if min(self.ward_lags_h) < 1:
            raise ValueError("every lag must be at least 1 hour, or the target leaks")
        return self

    @property
    def max_lag_h(self) -> int:
        return max(max(self.ward_lags_h), self.citywide_lag_h)


class RainForecast(ConfigModel):
    sigma: SourcedValue[float]
    p_mixup: SourcedValue[float]
    seed: int


class LightGbm(ConfigModel):
    objective: str
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(ge=2)
    min_data_in_leaf: int = Field(ge=1)
    feature_fraction: float = Field(gt=0, le=1)
    bagging_fraction: float = Field(gt=0, le=1)
    bagging_freq: int = Field(ge=0)
    num_boost_round: int = Field(ge=1)
    early_stopping_rounds: int = Field(ge=1)
    seed: int
    num_threads: int = Field(ge=1)

    def params(self) -> dict[str, object]:
        """Booster parameters (everything but the loop controls)."""
        skip = {"num_boost_round", "early_stopping_rounds"}
        return {k: v for k, v in self.model_dump().items() if k not in skip} | {"verbosity": -1}


class NoiseKnob(ConfigModel):
    sigma: SourcedValue[float]


class ForecastingConfig(ConfigModel):
    synthetic: bool
    history_scenario: str
    history: History
    targets: Targets
    split: Split
    features: Features
    feature_sets: dict[str, list[str]]
    rain_forecast: RainForecast
    lightgbm: LightGbm
    noise_knob: NoiseKnob

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.synthetic:
            raise ValueError("forecasting.yaml trains on synthetic history; synthetic must be true")
        if set(self.feature_sets) != {"realistic", "informed"}:
            raise ValueError("feature_sets must define exactly 'realistic' and 'informed'")
        if "hotspot_mass" in self.feature_sets["realistic"]:
            raise ValueError(
                "the realistic model must not see the generator's own ward hotspot mass"
            )
        if "rain_true" in self.feature_sets["realistic"]:
            raise ValueError("the realistic model must see a rain forecast, not the true rain")
        return self


def load_forecasting_config(path: Path | None = None) -> ForecastingConfig:
    return load_model(ForecastingConfig, path or CONFIG_DIR / "forecasting.yaml")
