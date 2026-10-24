"""Typed configs of the call process, call triage and fusion (configs/calls.yaml,
call_templates.yaml, triage.yaml, fusion.yaml; phase 4, D-028)."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from resqcity.common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from resqcity.common.schemas import EventType, Language

SEVERITY_CLASSES = ("minor", "serious", "critical")
CUES = ("person_down", "overturned", "fire_visible", "multi_vehicle")


def _check_classes(name: str, value: dict[str, float]) -> None:
    if set(value) != set(SEVERITY_CLASSES):
        raise ValueError(f"{name} needs exactly the severity classes {SEVERITY_CLASSES}")


# -- calls.yaml -------------------------------------------------------------------------------


class Reporting(ConfigModel):
    p_unreported: SourcedValue[dict[str, float]]
    first_delay_median_s: SourcedValue[float]
    first_delay_sigma: SourcedValue[float]
    extra_calls_mean: SourcedValue[dict[str, float]]
    extra_gap_mean_s: SourcedValue[float]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_classes("p_unreported", self.p_unreported.value)
        _check_classes("extra_calls_mean", self.extra_calls_mean.value)
        if not all(0 <= p <= 1 for p in self.p_unreported.value.values()):
            raise ValueError("p_unreported values must be probabilities")
        if self.first_delay_median_s.value <= 0 or self.first_delay_sigma.value <= 0:
            raise ValueError("first-call delay median and sigma must be positive")
        if self.extra_gap_mean_s.value <= 0:
            raise ValueError("extra_gap_mean_s must be positive")
        return self


class LocationMode(ConfigModel):
    median_error_m: SourcedValue[float]
    sigma: SourcedValue[float]


class Location(ConfigModel):
    mode: str
    modes: dict[str, LocationMode]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.mode not in self.modes:
            raise ValueError(f"location mode {self.mode!r} is not defined under modes")
        return self

    @property
    def active(self) -> LocationMode:
        return self.modes[self.mode]


class CallerCues(ConfigModel):
    p_mention_present: SourcedValue[float]
    p_mention_absent: SourcedValue[float]
    p_type_confusion: SourcedValue[float]


class FalseCalls(ConfigModel):
    rate_per_day: SourcedValue[float]
    note: str = ""


class CallsConfig(ConfigModel):
    synthetic: bool
    reporting: Reporting
    location: Location
    language_mix: SourcedValue[dict[Language, float]]
    caller_cues: CallerCues
    false_calls: FalseCalls

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.synthetic:
            raise ValueError("calls.yaml describes a synthetic process; synthetic must be true")
        if abs(sum(self.language_mix.value.values()) - 1.0) > 1e-9:
            raise ValueError("language_mix must sum to 1")
        return self


# -- call_templates.yaml ----------------------------------------------------------------------


class PhraseSet(ConfigModel):
    types: dict[EventType, list[str]]
    cues: dict[str, list[str]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.types) != set(EventType) or set(self.cues) != set(CUES):
            raise ValueError("every phrase set needs every event type and every cue")
        if any(not v for v in [*self.types.values(), *self.cues.values()]):
            raise ValueError("every type and cue needs at least one phrasing")
        return self


class LanguageTemplates(ConfigModel):
    location: list[str] = Field(min_length=1)
    location_unknown: str
    vehicles: dict[str, str]
    main: PhraseSet
    held_out: PhraseSet


class TemplatesConfig(ConfigModel):
    synthetic: bool
    share_held_out: float = Field(ge=0, le=1)
    languages: dict[Language, LanguageTemplates]


# -- triage.yaml ------------------------------------------------------------------------------


class RuleLexicon(ConfigModel):
    types: dict[EventType, list[str]]
    cues: dict[str, list[str]]
    vehicle_count_pattern: str
    vehicles: dict[str, str]


class RulesConfig(ConfigModel):
    type_priority: list[EventType]
    fallback_type: EventType
    confidence_matched: SourcedValue[float]
    confidence_fallback: SourcedValue[float]
    languages: dict[Language, RuleLexicon]


class LlmConfig(ConfigModel):
    enabled: bool = False
    model: str
    revision: str
    quantization: str = "nf4"
    max_new_tokens: int = Field(ge=1)


class TriageConfig(ConfigModel):
    rules: RulesConfig
    llm: LlmConfig


# -- fusion.yaml ------------------------------------------------------------------------------


class Gate(ConfigModel):
    base_distance_m: SourcedValue[float]
    k_location_error: SourcedValue[float]
    time_window_s: SourcedValue[float]


class CameraModel(ConfigModel):
    coverage_radius_m: SourcedValue[float]
    detected_types: list[EventType]
    cue_flip_probability: SourcedValue[float]
    persistent_share: SourcedValue[float]
    persistent_rate_multiplier: SourcedValue[float]


class Review(ConfigModel):
    sensitivity: SourcedValue[float]
    specificity: SourcedValue[float]
    duration_s: SourcedValue[float]


class SeverityWeights(ConfigModel):
    camera_weight: SourcedValue[float]
    call_weight: SourcedValue[float]


class Update(ConfigModel):
    posterior_epsilon: float = Field(gt=0, lt=1)
    expiry_s: float = Field(gt=0)
    posterior_floor: float = Field(gt=0, lt=0.5)


class FusionConfig(ConfigModel):
    detector_inputs: Path
    gate: Gate
    camera: CameraModel
    review: Review
    severity: SeverityWeights
    update: Update


def load_calls_config(path: Path | None = None) -> CallsConfig:
    return load_model(CallsConfig, path or CONFIG_DIR / "calls.yaml")


def load_templates(path: Path | None = None) -> TemplatesConfig:
    return load_model(TemplatesConfig, path or CONFIG_DIR / "call_templates.yaml")


def load_triage_config(path: Path | None = None) -> TriageConfig:
    return load_model(TriageConfig, path or CONFIG_DIR / "triage.yaml")


def load_fusion_config(path: Path | None = None) -> FusionConfig:
    return load_model(FusionConfig, path or CONFIG_DIR / "fusion.yaml")


# -- measured detector model (results/vision/sim_inputs.yaml, simulator_defaults) -----------------


class DetectorModel(ConfigModel):
    """The camera detector as the simulator and fusion see it (D-027 simulator defaults; the
    confidence model is exploratory, D-028)."""

    p_detect: float = Field(gt=0, le=1)
    delay_s: float = Field(ge=0)
    false_rate_per_armed_h: float = Field(gt=0)
    cooldown_s: float = Field(gt=0)
    bin_edges: list[float] = Field(min_length=2)
    real_shares: list[float]
    false_shares: list[float]
    real_confidences: list[float] = Field(min_length=1)
    false_confidences: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        n = len(self.bin_edges) - 1
        if len(self.real_shares) != n or len(self.false_shares) != n:
            raise ValueError("one real and one false share per confidence bin")
        if any(s <= 0 for s in self.false_shares):
            raise ValueError("false-alarm shares must be positive (likelihood ratio)")
        return self

    @property
    def long_run_false_rate_per_h(self) -> float:
        """False alarms per camera-hour with the cooldown: lambda / (1 + lambda * cooldown)."""
        lam = self.false_rate_per_armed_h
        return lam / (1.0 + lam * self.cooldown_s / 3600.0)

    def bin_of(self, confidence: float) -> int:
        edges = self.bin_edges
        for j in range(len(edges) - 2, -1, -1):
            if confidence >= edges[j]:
                return j
        return 0


def load_detector_model(path: Path) -> DetectorModel:
    import yaml

    d = yaml.safe_load(path.read_text(encoding="utf-8"))["simulator_defaults"]
    cm = d["confidence_model"]
    return DetectorModel(
        p_detect=d["detection_probability_when_armed"],
        delay_s=d["delay_median_s"],
        false_rate_per_armed_h=d["false_alarm_rate_per_armed_hour"],
        cooldown_s=d["cooldown_s"],
        bin_edges=cm["bin_edges"],
        real_shares=cm["real_armed"]["real_share_per_bin"],
        false_shares=cm["false_share_per_bin"],
        real_confidences=cm["samples"]["real_armed_confidence"],
        false_confidences=cm["samples"]["false_confidence"],
    )
