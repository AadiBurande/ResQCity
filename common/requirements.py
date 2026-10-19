"""Severity score and required services from observable incident cues.

Shared by the vision pipeline (estimates from video) and the simulator (ground truth), driven by
``configs/requirements.yaml``. Rules only test for cues being *present* (or minimum values) and
are combined by element-wise maximum, so adding a cue can never reduce the services required.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from common.schemas import Agency, EventType, ServiceCounts, SeverityFeatures


class SeverityWeights(ConfigModel):
    person_down: float = Field(ge=0)
    overturned: float = Field(ge=0)
    fire_visible: float = Field(ge=0)
    heavy_vehicle: float = Field(ge=0)
    multi_vehicle: float = Field(ge=0)


class SeverityClass(ConfigModel):
    name: str
    below: float


class SeverityConfig(ConfigModel):
    base: SourcedValue[dict[EventType, float]]
    weights: SourcedValue[SeverityWeights]
    multi_vehicle_min: int = Field(ge=1)
    heavy_vehicle_classes: list[str]
    classes: list[SeverityClass] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(EventType) - set(self.base.value)
        if missing:
            raise ValueError(f"severity.base missing types: {sorted(missing)}")
        bounds = [c.below for c in self.classes]
        if bounds != sorted(bounds) or bounds[-1] <= 1.0:
            raise ValueError("severity classes must be ascending and the last must exceed 1.0")
        return self


class RuleCondition(ConfigModel):
    """All given conditions must hold.

    Flags may only be ``true`` (presence required), which keeps the rule table monotone.
    """

    types: list[EventType] = Field(min_length=1)
    person_down: bool | None = None
    overturned: bool | None = None
    fire_visible: bool | None = None
    heavy_vehicle: bool | None = None
    min_vehicles: int | None = Field(default=None, ge=0)
    min_severity: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _presence_only(self) -> Self:
        for flag in ("person_down", "overturned", "fire_visible", "heavy_vehicle"):
            if getattr(self, flag) is False:
                raise ValueError(f"rule condition {flag}: false is not allowed (non-monotone)")
        return self

    @property
    def is_type_only(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "person_down",
                "overturned",
                "fire_visible",
                "heavy_vehicle",
                "min_vehicles",
                "min_severity",
            )
        )


class Rule(ConfigModel):
    name: str
    when: RuleCondition
    require: ServiceCounts


class RequirementsConfig(ConfigModel):
    provenance: SourcedValue[str]
    severity: SeverityConfig
    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def _every_type_has_a_base_rule(self) -> Self:
        covered = {
            t
            for r in self.rules
            if r.when.is_type_only and r.require.total > 0
            for t in r.when.types
        }
        missing = set(EventType) - covered
        if missing:
            raise ValueError(f"no unconditional base rule for types: {sorted(missing)}")
        return self


def load_requirements_config(path: Path | None = None) -> RequirementsConfig:
    return load_model(RequirementsConfig, path or CONFIG_DIR / "requirements.yaml")


def has_heavy_vehicle(features: SeverityFeatures, heavy_classes: list[str]) -> bool:
    return any(c in heavy_classes for c in features.vehicle_classes)


def severity_score(event_type: EventType, features: SeverityFeatures, cfg: SeverityConfig) -> float:
    w = cfg.weights.value
    score = cfg.base.value[event_type]
    score += w.person_down * features.person_down
    score += w.overturned * features.overturned
    score += w.fire_visible * features.fire_visible
    score += w.heavy_vehicle * has_heavy_vehicle(features, cfg.heavy_vehicle_classes)
    score += w.multi_vehicle * (features.num_vehicles >= cfg.multi_vehicle_min)
    return min(1.0, max(0.0, score))


def severity_class(score: float, cfg: SeverityConfig) -> str:
    for cls in cfg.classes:
        if score < cls.below:
            return cls.name
    return cfg.classes[-1].name


def _matches(
    cond: RuleCondition,
    event_type: EventType,
    features: SeverityFeatures,
    score: float,
    heavy_classes: list[str],
) -> bool:
    if event_type not in cond.types:
        return False
    if cond.person_down and not features.person_down:
        return False
    if cond.overturned and not features.overturned:
        return False
    if cond.fire_visible and not features.fire_visible:
        return False
    if cond.heavy_vehicle and not has_heavy_vehicle(features, heavy_classes):
        return False
    if cond.min_vehicles is not None and features.num_vehicles < cond.min_vehicles:
        return False
    return cond.min_severity is None or score >= cond.min_severity


def matching_rules(
    event_type: EventType, features: SeverityFeatures, cfg: RequirementsConfig
) -> list[Rule]:
    score = severity_score(event_type, features, cfg.severity)
    heavy = cfg.severity.heavy_vehicle_classes
    return [r for r in cfg.rules if _matches(r.when, event_type, features, score, heavy)]


def required_services(
    event_type: EventType, features: SeverityFeatures, cfg: RequirementsConfig
) -> ServiceCounts:
    """Element-wise maximum of the requirements of every matching rule."""
    rules = matching_rules(event_type, features, cfg)
    return ServiceCounts(
        **{a.value: max((r.require.get(a) for r in rules), default=0) for a in Agency}
    )
