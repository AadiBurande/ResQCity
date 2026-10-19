# ResQCity — Project memory for Claude Code

## What this project is
A research prototype + study for **CCTV-triggered, confidence-aware, joint multi-agency dispatch**
(ambulance, police, fire brigade) on the real **Pune road network**. It is a university Research
Methodology project, so it must be BOTH a working demo AND a reproducible experiment.

Research question: Does CCTV-triggered, confidence-aware joint dispatch reduce time-to-full-response
versus today's call-based, agency-by-agency dispatch, without increasing inequity across wards?

## Architecture (one line per module)
All Python code is one package in `src/resqcity/` (import as `resqcity.<module>`, run as
`python -m resqcity.<module>.<script>`). Python modules below live in `src/resqcity/`; `dashboard/`,
`configs/`, `data/`, `results/`, `notebooks/`, `tests/` and `docs/` are at the repo root.
A module folder is created in the phase that implements it.
- `common/`        Shared pydantic schemas + config loading, RNG streams, provenance sidecars
- `geo/`           Pune road network (OSMnx), wards, facilities, camera sites, scenarios, travel times
- `vision/`        CCTV pipeline: YOLO + ByteTrack -> accident / fire-smoke / severity -> `Event` with confidence
- `streams/`       Simulated live cameras: MediaMTX + ffmpeg looping clips as RTSP, pinned to real junctions
- `fusion/`        Bayesian fusion of CCTV events + simulated 112 calls -> incident with posterior + severity
- `forecasting/`   LightGBM ward-hour incident risk (rain, hour, weekday, festival, hotspot features)
- `optimization/`  OR models: joint dispatch MILP, confidence-aware policy, pre-positioning (fairness),
                   camera placement (MCLP), Erlang-C fleet sizing, time-dependent routing
- `sim/`           SimPy discrete-event simulator (digital twin) with pluggable dispatch policies
- `backend/`       FastAPI + WebSocket event bus, operator approve/override, audit log
- `dashboard/`     React + Vite + TypeScript + Leaflet live map, camera tiles, explanations, KPIs
- `experiments/`   Configs, runners, statistics, figures for the research results
- `tools/`         Task runner behind the Makefile, TypeScript type generation
- `docs/`          Methodology, model formulations (LaTeX), data sources, decisions, limitations, ethics;
                   `docs/kit/` holds the original build prompts and setup guide

## Tech conventions
- Python 3.11, dependency management with `uv`; lint/format with `ruff`; type hints everywhere; `pytest`.
- Node 20 + pnpm for `dashboard/`.
- Config via YAML in `configs/`, loaded into pydantic models. No magic numbers in code.
- Shared schemas live in `src/resqcity/common/schemas.py`: Camera, Event, Call, Incident, Unit, Station, Hospital,
  DispatchDecision, Explanation. Dashboard TypeScript types are generated from these.
- OR solvers: OR-Tools (CP-SAT / MIP) first, PuLP + CBC as fallback. Every model has a tiny-instance
  unit test checked against brute force.
- Road graph: OSMnx, cached to `data/processed/pune_drive.graphml`. Tests never hit the network.
- GPU-heavy training (vision) must also run on Google Colab via a notebook in `notebooks/vision/`.

## Research integrity rules (non-negotiable)
1. Never invent or hand-type a result. Every number in docs/figures comes from a file in `results/`
   produced by a committed script + config + seed.
2. Synthetic data is labelled synthetic in file names, metadata and docs.
3. Any medical or epidemiological parameter (e.g. survival-vs-time curve) lives in config with a
   `source:` field citing the paper. If the source is not confirmed, mark it `UNVERIFIED` and flag it.
4. Vision splits are by video, never by frame (prevents leakage). Report results per dataset.
5. Experiments use common random numbers (same seed set across policies) and >= 30 replications.
6. Privacy by design: faces and number plates are blurred at the edge; only event metadata leaves `vision/`.

## Commands (create these in the Makefile)
- `make setup` · `make test` · `make lint`
- `make streams` start simulated RTSP cameras · `make api` backend · `make ui` dashboard
- `make demo` end-to-end live demo · `make sim SCENARIO=monsoon` · `make experiments` · `make figures`
- `make reproduce` regenerates every result and figure from scratch

## Working style for Claude
- Start each phase in plan mode; save the plan to `docs/plans/phase-N.md` before coding.
- Build in small vertical slices with tests; commit at the end of each phase.
- If a requirement is ambiguous and expensive to redo, ask. Otherwise pick a sensible default and
  record it in `docs/decisions.md`.
- Use the project skills: `or-model-builder`, `run-experiment`, `vision-eval`, `research-writeup`.
- Before marking a phase done, run `/verify` and ask the `or-model-reviewer` or `research-auditor`
  subagent to review the phase where relevant.
