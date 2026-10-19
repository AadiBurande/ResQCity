# Phase 1 — Scaffolding and shared schemas

Status: implemented · Date: 2026-10-20

## Goal
A monorepo where `make setup && make lint && make test` pass, the shared schemas round-trip through
JSON, and the dashboard TypeScript types are generated from the pydantic models.

## Steps
1. `git init`, `.gitignore`, `.gitattributes` (LF line endings, so the generated TS file is byte-stable).
2. `pyproject.toml` (uv, hatchling, Python 3.11): runtime deps `pydantic>=2`, `pyyaml`; dev deps
   `ruff`, `pytest`, `hypothesis`, `pre-commit`. Later phases add their own dependencies when needed.
3. Package skeletons (`__init__.py` with a docstring) for `common geo vision streams fusion forecasting
   optimization sim backend experiments paper tools`.
4. `common/schemas.py`: every model from master-plan §4. Shared base `Schema` (`extra="forbid"`,
   defaults marked required in the serialization JSON schema, so TS fields are non-optional).
   Enums are `StrEnum`. `SCHEMA_MODELS` lists the exported models in one place.
5. `common/config.py`: `SourcedValue` (value + source + verified), `BaseConfig` typed model,
   `load_yaml`, `load_config(path)`, `deep_merge`, `find_unverified(model)`.
   `configs/base.yaml` holds the project, paths, random seeds, study area and time zone.
6. `tools/gen_ts_types.py`: pydantic JSON schema → TypeScript (no npm dependency). `--check` mode
   fails if the committed file is stale. Output: `dashboard/src/types/schemas.ts`.
7. `tools/tasks.py`: cross-platform task runner (GNU make is not on this Windows machine). The
   `Makefile` targets delegate to it. Unimplemented targets print "not implemented yet (phase N)".
8. `.pre-commit-config.yaml` (local hooks: ruff check, ruff format, TS types check).
9. `.github/workflows/ci.yml`: uv sync → ruff check → ruff format --check → pytest → TS check.
10. Tests: schema JSON round-trip for every model (hand-built + Hypothesis), validation bounds,
    config loading / deep merge / UNVERIFIED detection, TS generator (golden structure, stale check).
11. `docs/decisions.md`, `docs/data-sources.md` templates; record D-001…D-00n.

## Acceptance
- `make setup && make lint && make test` (or `uv run python -m resqcity.tools.tasks setup|lint|test`) pass.
- `tools/gen_ts_types.py --check` passes, and `tsc --noEmit --strict` accepts the generated file.
- Commit: "phase 1: scaffolding and schemas".
