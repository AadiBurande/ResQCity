# Decision log

Design choices made where the brief was ambiguous or silent. Newest last. Each entry records the
context, the decision, alternatives considered, and the consequences. If a decision is reversed,
add a new entry that supersedes the old one; do not edit history.

Template:

```
## D-NNN Title  (YYYY-MM-DD, phase N)
Context: ...
Decision: ...
Alternatives: ...
Consequences: ...
```

---

## D-001 Cross-platform task runner behind the Makefile  (2026-10-20, phase 1)
Context: the development machine is Windows without GNU make; CLAUDE.md specifies `make` targets.
Decision: the `Makefile` stays the documented interface. Every target delegates to
`tools/tasks.py`, which also runs as `uv run python -m tools.tasks <target>`.
Targets of later phases exit with status 3 ("not implemented yet (phase N)") instead of 0.
Alternatives: PowerShell `make.ps1` (not portable to CI/Linux); only stubs that print and succeed
(a silent no-op `make reproduce` would be a research-integrity hazard).
Consequences: one implementation per task; CI calls the same runner.

## D-002 Schema additions beyond the brief's field list  (2026-10-20, phase 1)
Context: later phases need fields that the Phase 1 brief does not list.
Decision: added `Camera.stream_url`, `Event.confidence_components` (explanation panel),
`Call.language`, `Call.triage_confidence`, `Call.synthetic`, `Incident.type`, `Incident.updated_ts`,
`Unit.position`, `Station.{name, location, ward_id, source}`, `Hospital.{name, location,
capabilities_synthetic}`, `Assignment.hospital_id`, `DispatchDecision.{id, created_ts, policy,
status}` (`Explanation.decision_id` needs a decision id). Value objects `GeoPoint`,
`ServiceCounts`, `SourceRef`, `Factor`.
Consequences: every addition has a default or is needed for referential integrity. All models
forbid unknown fields.

## D-003 Units and time conventions  (2026-10-20, phase 1)
Decision: durations in seconds, distances in metres, probabilities in [0, 1], timestamps are
timezone-aware (`AwareDatetime`); project time zone `Asia/Kolkata` from `configs/base.yaml`.
Consequences: naive datetimes are rejected at validation, so sim and live code cannot silently
mix time zones.

## D-004 TypeScript types generated without npm  (2026-10-20, phase 1)
Context: `pydantic2ts` needs the npm package `json-schema-to-typescript`; pnpm is not installed yet.
Decision: `tools/gen_ts_types.py` converts the pydantic serialization JSON schema to TypeScript
in pure Python. Unsupported constructs raise an error. A `--check` mode guards against stale
files in pre-commit and CI. CI type-checks the output with `tsc --strict`.
Consequences: no Node dependency for the Python toolchain; supports only the JSON-schema subset
our models use (which is enforced by tests).

## D-005 Literature parameters carry their source  (2026-10-20, phase 1)
Decision: config values from the literature use `SourcedValue {value, source, note}`.
`source: UNVERIFIED` marks unconfirmed values; `common.config.find_unverified` lists them so
experiment metadata and docs can flag them automatically (research-integrity rule 3).
