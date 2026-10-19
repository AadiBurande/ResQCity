# ResQCity

Research prototype and study: **CCTV-triggered, confidence-aware, joint multi-agency dispatch**
(ambulance, police, fire) on the real Pune road network.

> Status: phase 1 (scaffolding + shared schemas). See [docs/plans/master-plan.md](docs/plans/master-plan.md).
> Incident, call and hospital-capability data in this project are **synthetic** unless stated otherwise.

## Quick start

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Node 20+ (dashboard, later phases).

```bash
make setup     # uv sync, generate TS types, install pre-commit hooks
make lint
make test
```

No GNU make (e.g. Windows)? Every target also runs as:

```bash
uv sync
uv run python -m tools.tasks setup   # or: lint | test | types | sim --scenario monsoon ...
```

## Layout

See [CLAUDE.md](CLAUDE.md) for module responsibilities and the research-integrity rules, and
[docs/decisions.md](docs/decisions.md) for design decisions.
