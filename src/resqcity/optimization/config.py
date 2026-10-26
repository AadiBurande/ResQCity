"""Typed configs of the phase-6 OR models (configs/fleet.yaml, configs/optimization.yaml)."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from resqcity.common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from resqcity.common.schemas import Agency

SEVERITY_CLASSES = ("minor", "serious", "critical")


# -- fleet.yaml ---------------------------------------------------------------------------------


class AmbulanceBases(ConfigModel):
    prefer_capable: bool
    max_bases: SourcedValue[int]


class Placement(ConfigModel):
    rule: str
    max_units_per_station: SourcedValue[int]
    ambulance_bases: AmbulanceBases
    seed: int

    @model_validator(mode="after")
    def _known_rule(self) -> Self:
        if self.rule != "demand_weighted":
            raise ValueError("the only implemented placement rule is 'demand_weighted'")
        if self.max_units_per_station.value < 1:
            raise ValueError("max_units_per_station must be at least 1")
        return self


class FleetConfig(ConfigModel):
    synthetic_roster: bool
    counts: dict[Agency, SourcedValue[int]]
    placement: Placement

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.synthetic_roster:
            raise ValueError("the roster is synthetic; synthetic_roster must be true")
        if set(self.counts) != set(Agency):
            raise ValueError(f"counts must cover every agency: {sorted(a.value for a in Agency)}")
        if any(c.value < 0 for c in self.counts.values()):
            raise ValueError("unit counts must not be negative")
        return self

    def total_units(self) -> int:
        return sum(c.value for c in self.counts.values())


# -- optimization.yaml --------------------------------------------------------------------------


class GreenCorridor(ConfigModel):
    rho: SourcedValue[float]
    note: str = ""

    @model_validator(mode="after")
    def _in_range(self) -> Self:
        if not 0 < self.rho.value <= 1:
            raise ValueError("rho is a speed-up factor on ambulance travel time, in (0, 1]")
        return self


class RoutingConfig(ConfigModel):
    hour_buckets: int = Field(ge=1, le=24)
    profile: str
    green_corridor: GreenCorridor


class SolverConfig(ConfigModel):
    time_limit_s: float = Field(gt=0)
    workers: int = Field(ge=1)
    greedy_fallback: bool


class DispatchConfig(ConfigModel):
    severity_weights: SourcedValue[dict[str, float]]
    lambda_coverage: SourcedValue[float]
    mu_shortfall: SourcedValue[float]
    epsilon_ambulance: SourcedValue[float]
    note_epsilon: str = ""
    solver: SolverConfig

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.severity_weights.value) != set(SEVERITY_CLASSES):
            raise ValueError(f"severity_weights needs exactly {SEVERITY_CLASSES}")
        if any(v <= 0 for v in self.severity_weights.value.values()):
            raise ValueError("severity weights must be positive")
        smallest = min(self.severity_weights.value.values())
        if self.epsilon_ambulance.value >= smallest:
            raise ValueError(
                "epsilon_ambulance must be far below the smallest severity weight, or the "
                "secondary ambulance term could change the last-arrival time it is meant to "
                "leave untouched"
            )
        return self

    def weight(self, severity_class: str) -> float:
        return self.severity_weights.value[severity_class]


class PolicyConfig(ConfigModel):
    tau_low: SourcedValue[float]
    tau_high: SourcedValue[float]
    t_review_s: SourcedValue[float]
    escalate_on_timeout: bool
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not 0 <= self.tau_low.value <= self.tau_high.value <= 1:
            raise ValueError("thresholds must satisfy 0 <= tau_low <= tau_high <= 1")
        if self.t_review_s.value < 0:
            raise ValueError("t_review_s must not be negative")
        return self


class Fairness(ConfigModel):
    enabled: bool
    alpha: SourcedValue[float]
    t_ward_s: SourcedValue[float]

    @model_validator(mode="after")
    def _in_range(self) -> Self:
        if not 0 <= self.alpha.value <= 1:
            raise ValueError("alpha is a probability")
        return self


class Relocation(ConfigModel):
    max_moves: SourcedValue[int]
    max_total_s: SourcedValue[float]


class PrepositionConfig(ConfigModel):
    every_minutes: int = Field(ge=1)
    busy_fraction: SourcedValue[float]
    coverage_radius_s: SourcedValue[float]
    fairness: Fairness
    relocation: Relocation

    @model_validator(mode="after")
    def _in_range(self) -> Self:
        if not 0 <= self.busy_fraction.value < 1:
            raise ValueError("busy_fraction is the MEXCLP q, in [0, 1)")
        return self


class PlacementConfig(ConfigModel):
    coverage_radius_m: SourcedValue[float]
    k_max: int = Field(ge=1)
    note: str = ""


class ErlangConfig(ConfigModel):
    service_time_s: SourcedValue[dict[Agency, float]]
    target_wait_s: SourcedValue[float]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.service_time_s.value) != set(Agency):
            raise ValueError("service_time_s must cover every agency")
        return self


class OptimizationConfig(ConfigModel):
    synthetic: bool
    routing: RoutingConfig
    dispatch: DispatchConfig
    policy: PolicyConfig
    preposition: PrepositionConfig
    placement: PlacementConfig
    erlang: ErlangConfig


def load_fleet_config(path: Path | None = None) -> FleetConfig:
    return load_model(FleetConfig, path or CONFIG_DIR / "fleet.yaml")


def load_optimization_config(path: Path | None = None) -> OptimizationConfig:
    return load_model(OptimizationConfig, path or CONFIG_DIR / "optimization.yaml")
