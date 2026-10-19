"""Tests for the pydantic -> TypeScript generator."""

import re
from pathlib import Path

import pytest

from resqcity.common.schemas import SCHEMA_MODELS
from resqcity.tools import gen_ts_types
from resqcity.tools.gen_ts_types import DEFAULT_OUT, render, ts_type


def test_every_model_becomes_an_interface():
    out = render()
    for model in SCHEMA_MODELS:
        assert f"export interface {model.__name__} {{" in out


def test_enums_become_string_unions():
    out = render()
    assert 'export type Agency = "ambulance" | "police" | "fire";' in out
    assert 'export const AgencyValues = ["ambulance", "police", "fire"] as const;' in out


def test_field_types():
    out = render()
    event = re.search(r"export interface Event \{(.*?)\n\}", out, re.S)
    assert event is not None
    body = event.group(1)
    assert "  ts: string;" in body
    assert "  type: EventType;" in body
    assert "  confidence: number;" in body
    assert "  clip_ref: string | null;" in body
    assert "  confidence_components: Record<string, number>;" in body
    assert "  required_services: ServiceCounts;" in body


def test_defaulted_fields_are_required_in_output():
    # Serialized payloads always include defaulted fields, so TS must not mark them optional.
    assert "?:" not in render()


def test_output_is_deterministic():
    assert render() == render()


def test_committed_file_is_up_to_date():
    assert DEFAULT_OUT.read_text(encoding="utf-8") == render()


def test_check_mode_detects_stale_file(tmp_path: Path):
    out = tmp_path / "schemas.ts"
    out.write_text("// stale\n", encoding="utf-8")
    assert gen_ts_types.main(["--check", "--out", str(out)]) == 1
    assert gen_ts_types.main(["--out", str(out)]) == 0
    assert gen_ts_types.main(["--check", "--out", str(out)]) == 0


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "integer"}, "number"),
        ({"type": "array", "items": {"type": "string"}}, "Array<string>"),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "string | null"),
        ({"$ref": "#/$defs/Unit"}, "Unit"),
        ({"const": "x"}, '"x"'),
        ({"type": "object"}, "Record<string, unknown>"),
    ],
)
def test_ts_type_fragments(schema: dict, expected: str):
    assert ts_type(schema) == expected


@pytest.mark.parametrize(
    "schema",
    [{"type": "object", "properties": {"a": {"type": "string"}}}, {"$ref": "other.json#/X"}, {}],
)
def test_unsupported_fragments_raise(schema: dict):
    with pytest.raises(ValueError):
        ts_type(schema)
