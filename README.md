# ResQCity

Research prototype and study: **CCTV-triggered, confidence-aware, joint multi-agency dispatch**
(ambulance, police, fire) on the real Pune road network.

> Status: phase 2 (Pune road network, wards, facilities, scenarios, synthetic incidents).
> See [docs/plans/master-plan.md](docs/plans/master-plan.md).
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
uv run python -m resqcity.tools.tasks setup   # or: lint | test | types | sim --scenario monsoon ...
```

## Data

```bash
make data      # downloads PMC wards + OSM roads/rivers/facilities, builds data/processed, sanity check
uv run python -m resqcity.sim.generator --scenario monsoon --replication 0   # synthetic incidents
```

Map data © OpenStreetMap contributors (ODbL). Ward boundaries © DataMeet Trust, CC BY-SA 2.5 IN.
See [docs/data-sources.md](docs/data-sources.md).

## Layout

```
src/resqcity/   Python package (common, geo, sim, tools; more modules added per phase)
configs/        YAML configuration (scenarios, speeds, incident generator, rule tables)
data/           raw downloads and processed caches (gitignored; rebuilt by `make data`)
results/        generated tables and figures, each with a .meta.json provenance sidecar
notebooks/      analysis notebooks
dashboard/      React dashboard (generated TypeScript types for now)
tests/          pytest suite mirroring src/resqcity
docs/           plans, decisions, data sources, limitations; docs/kit = original build prompts
```

See [CLAUDE.md](CLAUDE.md) for module responsibilities and the research-integrity rules, and
[docs/decisions.md](docs/decisions.md) for design decisions.
