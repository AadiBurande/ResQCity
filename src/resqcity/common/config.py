"""Typed configuration: YAML files in ``configs/`` loaded into pydantic models.

Research-integrity rule 3: any parameter taken from the literature (medical, epidemiological,
behavioural) is a :class:`SourcedValue` carrying a ``source`` citation. A source of
``UNVERIFIED`` means it has not been confirmed yet; :func:`find_unverified` lists such values so
they can be flagged in docs and results.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _find_project_root(start: Path) -> Path:
    """Nearest ancestor holding pyproject.toml (the repo root, where configs/ and data/ live)."""
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"no pyproject.toml above {start}; run from a ResQCity checkout")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
CONFIG_DIR = PROJECT_ROOT / "configs"
UNVERIFIED = "UNVERIFIED"
ILLUSTRATIVE = "ILLUSTRATIVE"
# Neither word is a citation, so both must be reported by find_unverified. ILLUSTRATIVE marks a
# value chosen to have the right shape rather than the right magnitude (a fixed festival date, a
# seasonal profile); treating it as sourced would hide it from the assumption list.
UNSOURCED = frozenset({UNVERIFIED, ILLUSTRATIVE})

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class ConfigModel(BaseModel):
    """Base for config sections: unknown keys are errors (catches typos in YAML)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourcedValue(ConfigModel, Generic[T]):
    """A parameter value together with the citation it comes from."""

    value: T
    source: str = Field(
        min_length=1, description=f"Citation, or one of {sorted(UNSOURCED)} when there is none."
    )
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.source.strip().upper() not in UNSOURCED


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
    """Disjoint seed bases per purpose, so tuned parameters are never evaluated on their own data.

    ``seed_base`` is the evaluation set (confirmatory experiments); threshold/parameter tuning and
    forecast-model training use their own bases.
    """

    seed_base: int = Field(ge=0)
    tuning_seed_base: int = Field(ge=0)
    forecast_training_seed_base: int = Field(ge=0)

    @model_validator(mode="after")
    def _disjoint(self) -> "RandomSection":
        seeds = [self.seed_base, self.tuning_seed_base, self.forecast_training_seed_base]
        if len(set(seeds)) != len(seeds):
            raise ValueError("evaluation, tuning and forecast-training seed bases must differ")
        return self


class BaseConfig(ConfigModel):
    project: ProjectSection
    paths: PathsSection
    random: RandomSection


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
