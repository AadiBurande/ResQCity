"""Tests for typed configuration loading."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.config import (
    CONFIG_DIR,
    BaseConfig,
    ConfigModel,
    SourcedValue,
    deep_merge,
    find_unverified,
    load_base_config,
    load_model,
    load_yaml,
)


def _write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_config_loads():
    cfg = load_base_config()
    assert isinstance(cfg, BaseConfig)
    assert cfg.project.timezone == "Asia/Kolkata"
    assert cfg.paths.resolve("configs") == CONFIG_DIR


def test_unknown_key_is_an_error(tmp_path: Path):
    data = load_yaml(CONFIG_DIR / "base.yaml")
    data["random"]["sead_base"] = 1  # typo
    with pytest.raises(ValidationError):
        load_model(BaseConfig, _write(tmp_path / "bad.yaml", data))


def test_bad_timezone_rejected(tmp_path: Path):
    data = load_yaml(CONFIG_DIR / "base.yaml")
    data["project"]["timezone"] = "Asia/Poona"
    with pytest.raises(ValidationError, match="timezone"):
        load_model(BaseConfig, _write(tmp_path / "bad.yaml", data))


def test_later_files_override_earlier(tmp_path: Path):
    override = _write(tmp_path / "override.yaml", {"random": {"seed_base": 7}})
    cfg = load_model(BaseConfig, CONFIG_DIR / "base.yaml", override)
    assert cfg.random.seed_base == 7
    assert cfg.project.name == "ResQCity"  # untouched keys survive the merge


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    override = {"a": {"b": 10}, "d": [3]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": [3]}
    assert base == {"a": {"b": 1, "c": 2}, "d": [1, 2]}


def test_empty_yaml_is_empty_mapping(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml(path) == {}


def test_non_mapping_yaml_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(_write(tmp_path / "list.yaml", [1, 2]))


def test_load_model_requires_a_path():
    with pytest.raises(ValueError):
        load_model(BaseConfig)


class _Survival(ConfigModel):
    half_life_min: SourcedValue[float]
    curves: dict[str, SourcedValue[float]]
    extras: list[SourcedValue[int]]


def test_find_unverified_walks_nested_structures():
    cfg = _Survival.model_validate(
        {
            "half_life_min": {"value": 10.0, "source": "UNVERIFIED"},
            "curves": {
                "cardiac": {"value": 0.1, "source": "Doe et al. 2020, Resuscitation"},
                "trauma": {"value": 0.05, "source": " unverified "},
            },
            "extras": [{"value": 3, "source": "UNVERIFIED"}],
        }
    )
    assert find_unverified(cfg) == ["half_life_min", "curves.trauma", "extras[0]"]
    assert cfg.curves["cardiac"].verified


def test_sourced_value_requires_a_source():
    with pytest.raises(ValidationError):
        SourcedValue[float].model_validate({"value": 1.0, "source": ""})
