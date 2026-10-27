"""Typed config of the discrete-event simulator (configs/sim.yaml, configs/survival.yaml).

Phase 7, D-032. Two rules are enforced here rather than left to discipline:

1. **Background workload cannot be silently switched off.** It exists because D-031 measured the
   modelled city at 0.3 % utilisation (L30). Disabling it is a legitimate phase-10 ablation, so
   it is allowed, but the target must then be explicitly zero -- ``enabled: false`` with a
   non-zero target is a contradiction and is rejected.
2. **Survival is never the primary outcome.** ``survival.yaml`` declares
   ``primary_outcome: time_to_last_required_arrival`` and the validator refuses anything else,
   because the evidence for a survival-vs-time curve in road trauma is inconsistent (see the
   file's own header and ``docs/plans/phase-7.md``).

Background SERVICE times are deliberately not in sim.yaml: they are read from
``configs/optimization.yaml`` -> ``erlang.service_time_s`` so the simulator and the Erlang-C
cross-check of phase 6 cannot drift apart.
"""

import math
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from resqcity.common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from resqcity.common.schemas import Agency, EventType

HOURS_PER_DAY = 24
TIME_OUTCOME = "time_to_last_required_arrival"
POLICIES = ("B0", "B1", "B2", "B3", "FULL")


class RunConfig(ConfigModel):
    horizon_days: int = Field(ge=1)
    warmup_hours: float = Field(ge=0)
    replications: int = Field(ge=1)
    policies: list[str] = Field(min_length=1)
    dispatch_interval_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if unknown := set(self.policies) - set(POLICIES):
            raise ValueError(f"unknown policies {sorted(unknown)}; known: {list(POLICIES)}")
        if self.warmup_hours >= self.horizon_days * HOURS_PER_DAY:
            raise ValueError("warmup_hours must leave some of the horizon to measure")
        return self

    @property
    def horizon_s(self) -> float:
        return self.horizon_days * HOURS_PER_DAY * 3600.0

    @property
    def warmup_s(self) -> float:
        return self.warmup_hours * 3600.0


class OnSceneConfig(ConfigModel):
    median_s: SourcedValue[dict[Agency, float]]
    false_alarm_s: SourcedValue[dict[Agency, float]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if missing := {a for a in Agency} - set(self.median_s.value):
            raise ValueError(f"on_scene.median_s needs every agency; missing {missing}")
        if any(v <= 0 for v in self.median_s.value.values()):
            raise ValueError("on_scene.median_s must be positive")
        return self


class TransportConfig(ConfigModel):
    p_transport: SourcedValue[dict[EventType, float]]
    handover_median_s: SourcedValue[float]
    no_capable_hospital: str
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not all(0.0 <= p <= 1.0 for p in self.p_transport.value.values()):
            raise ValueError("transport.p_transport values are probabilities")
        if self.no_capable_hospital != "log_and_release":
            raise ValueError("the only supported behaviour is log_and_release")
        return self


class SecondWaveConfig(ConfigModel):
    enabled: bool
    assessment_s: SourcedValue[float]
    note: str = ""


class BackgroundConfig(ConfigModel):
    """Out-of-scope workload that makes units genuinely busy (D-031 item 3, L30)."""

    enabled: bool
    target_utilisation: SourcedValue[dict[Agency, float]]
    preempt: bool
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        targets = self.target_utilisation.value
        if missing := {a for a in Agency} - set(targets):
            raise ValueError(f"background.target_utilisation needs every agency; missing {missing}")
        if not all(0.0 <= v < 1.0 for v in targets.values()):
            raise ValueError("target utilisation must be in [0, 1): 1 or more never clears")
        if self.preempt:
            raise ValueError("pre-empting a background job is not modelled (see the config note)")
        if not self.enabled and any(v > 0 for v in targets.values()):
            raise ValueError(
                "background.enabled is false but a target utilisation is non-zero. The "
                "capacity-blind ablation must say so explicitly: set every target to 0."
            )
        return self


class LogsConfig(ConfigModel):
    incident_columns: list[str] = Field(min_length=1)
    write_parquet: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(set(self.incident_columns)) != len(self.incident_columns):
            raise ValueError("logs.incident_columns has duplicates")
        required = {"incident_id", "onset_ts", "last_required_arrival_ts", "policy", "replication"}
        if missing := required - set(self.incident_columns):
            raise ValueError(f"logs.incident_columns must include {sorted(missing)}")
        return self


class ErlangCheckConfig(ConfigModel):
    agency: Agency
    arrival_per_day: float = Field(gt=0)
    units: int = Field(ge=1)
    horizon_days: int = Field(ge=1)
    tolerance_utilisation: float = Field(gt=0)
    tolerance_wait_s: float = Field(gt=0)


class ValidationConfig(ConfigModel):
    erlang_check: ErlangCheckConfig


class SimConfig(ConfigModel):
    synthetic: bool
    run: RunConfig
    on_scene: OnSceneConfig
    transport: TransportConfig
    second_wave: SecondWaveConfig
    background: BackgroundConfig
    logs: LogsConfig
    validation: ValidationConfig

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.synthetic:
            raise ValueError("the simulator runs on synthetic incidents; synthetic must be true")
        return self


# ------------------------------------------------------------------------------------------
# survival.yaml -- a SECONDARY outcome, and the validator keeps it that way
# ------------------------------------------------------------------------------------------


class SurvivalCurve(ConfigModel):
    s0: float = Field(gt=0, le=1)
    s_floor: float = Field(ge=0, le=1)
    tau_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.s_floor > self.s0:
            raise ValueError("s_floor cannot exceed s0: the curve must be non-increasing in time")
        return self

    def at(self, seconds: float) -> float:
        """S(t) = s_floor + (s0 - s_floor) * exp(-t / tau)."""
        if seconds < 0:
            raise ValueError("time since onset cannot be negative")
        return self.s_floor + (self.s0 - self.s_floor) * math.exp(-seconds / self.tau_s)


class SurvivalSensitivity(ConfigModel):
    tau_multipliers: SourcedValue[list[float]]
    flat_curve_ablation: SourcedValue[bool]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if any(m <= 0 for m in self.tau_multipliers.value):
            raise ValueError("tau multipliers must be positive")
        if 1.0 not in self.tau_multipliers.value:
            raise ValueError("the sensitivity sweep must include 1.0, the configured curve")
        return self


class SurvivalSource(ConfigModel):
    id: str
    citation: str
    url: str
    checked: date
    supports: str
    doi: str = ""


class SurvivalConfig(ConfigModel):
    synthetic: bool
    primary_outcome: str
    model: str
    curves: dict[EventType, SourcedValue[SurvivalCurve]]
    sensitivity: SurvivalSensitivity
    sources: list[SurvivalSource] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.primary_outcome != TIME_OUTCOME:
            raise ValueError(
                f"primary_outcome must stay {TIME_OUTCOME!r}: expected survival is a secondary "
                "outcome because the prehospital-time-to-trauma-mortality evidence is "
                "inconsistent (see the config header and docs/plans/phase-7.md)"
            )
        if self.model != "exponential_decay":
            raise ValueError("the only implemented survival model is exponential_decay")
        if missing := {t for t in EventType} - set(self.curves):
            raise ValueError(f"survival.curves needs every incident type; missing {missing}")
        if any(c.verified for c in self.curves.values()):
            raise ValueError(
                "a survival curve claiming a citation would let a conclusion rest on it; every "
                "curve must be UNVERIFIED or ILLUSTRATIVE until a source is confirmed"
            )
        return self

    def curve(self, event_type: EventType) -> SurvivalCurve:
        return self.curves[event_type].value


def load_sim_config(path: Path | None = None) -> SimConfig:
    return load_model(SimConfig, path or CONFIG_DIR / "sim.yaml")


def load_survival_config(path: Path | None = None) -> SurvivalConfig:
    return load_model(SurvivalConfig, path or CONFIG_DIR / "survival.yaml")
