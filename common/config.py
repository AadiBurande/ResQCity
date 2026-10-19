"""Typed configuration: YAML files in ``configs/`` loaded into pydantic models.

Research-integrity rule 3: any parameter taken from the literature (medical, epidemiological,
behavioural) is a :class:`SourcedValue` carrying a ``source`` citation. A source of
``UNVERIFIED`` means it has not been confirmed yet; :func:`find_unverified` lists such values so
they can be flagged in docs and results.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Generic, Self, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
UNVERIFIED = "UNVERIFIED"

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class ConfigModel(BaseModel):
    """Base for config sections: unknown keys are errors (catches typos in YAML)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourcedValue(ConfigModel, Generic[T]):
    """A parameter value together with the citation it comes from."""

    value: T
    source: str = Field(min_length=1, description=f"Citation, or '{UNVERIFIED}'.")
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.source.strip().upper() != UNVERIFIED


# ---------------------------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------------------------


class ProjectSection(ConfigModel):
    name: str
    timezone: str

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return value


class PathsSection(ConfigModel):
    """Paths relative to the project root."""

    data_raw: Path
    data_processed: Path
    results: Path
    configs: Path

    def resolve(self, name: str) -> Path:
        return PROJECT_ROOT / getattr(self, name)


class RandomSection(ConfigModel):
    seed_base: int = Field(ge=0)


class BBox(ConfigModel):
    """Geographic bounding box in degrees."""

    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (self.west < self.east and self.south < self.north):
            raise ValueError("bbox must satisfy west < east and south < north")
        return self


class StudyAreaSection(ConfigModel):
    place: str
    bbox: BBox | None = Field(
        default=None, description="If set, used instead of the place polygon (e.g. + PCMC)."
    )
    network_type: str


class BaseConfig(ConfigModel):
    project: ProjectSection
    paths: PathsSection
    random: RandomSection
    study_area: StudyAreaSection


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (mappings merge, everything else replaces)."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_model(model: type[M], *paths: Path) -> M:
    """Load one or more YAML files (later files override earlier ones) into ``model``."""
    if not paths:
        raise ValueError("at least one config path is required")
    data: dict[str, Any] = {}
    for path in paths:
        data = deep_merge(data, load_yaml(path))
    return model.model_validate(data)


def load_base_config(path: Path | None = None) -> BaseConfig:
    return load_model(BaseConfig, path or CONFIG_DIR / "base.yaml")


def _walk(obj: object, prefix: str) -> Iterator[tuple[str, SourcedValue[Any]]]:
    if isinstance(obj, SourcedValue):
        yield prefix, obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from _walk(getattr(obj, name), f"{prefix}.{name}" if prefix else name)
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _walk(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            yield from _walk(value, f"{prefix}[{i}]")


def find_unverified(config: BaseModel) -> list[str]:
    """Dotted paths of every :class:`SourcedValue` in ``config`` whose source is UNVERIFIED."""
    return [path for path, sv in _walk(config, "") if not sv.verified]
