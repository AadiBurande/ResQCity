# Phase 8 — Backend API (Live Mode)

Status: implementing · Date: 2026-11-20 · Decisions: docs/decisions.md

## Goal

Build a production-grade asynchronous FastAPI backend (`src/resqcity/backend/`) that powers live operations
and the control-room dashboard (Phase 9). The backend provides event-driven processing (CCTV events + calls →
Bayesian fusion → OR dispatch decision), a real-time unit simulator moving emergency vehicles along the Pune
road network, RESTful endpoints, live WebSocket streaming, and a legally compliant operator audit log.

## Architectural Boundaries

Per `tests/test_architecture.py`, `backend` is strictly forbidden from importing `sim` or `experiments` (`FORBIDDEN["backend"] = {"sim", "experiments"}`).
All live operational logic must use:
- `resqcity.common` (schemas, bus, config, rng)
- `resqcity.geo` (network, facilities, wards, travel times)
- `resqcity.fusion` (association, Bayesian log-odds posterior, requirement estimation)
- `resqcity.optimization` (joint dispatch CP-SAT solver, confidence policy, routing)
- `resqcity.forecasting` (ward risk predictor)

Real-time unit movement is handled by a lightweight live-mode discrete simulator (`backend/unit_sim.py`) that operates on wall-clock time rather than SimPy virtual ticks.

## System Components

```
src/resqcity/backend/
├── __init__.py           Package initialization
├── app.py                FastAPI application, CORS, lifespan context, WebSocket /ws
├── state.py              In-memory operational state (cameras, incidents, units, decisions, KPIs)
├── bus.py                Async bridge between InProcessBus and WebSocket broadcast
├── services.py           Vision ingestion, Bayesian fusion worker, auto/assist dispatch worker
├── unit_sim.py           Real-time vehicle movement engine along OSMnx graph paths
├── audit.py              Append-only operator audit log (who, when, what changed, reason)
└── routes/
    ├── __init__.py
    ├── cameras.py        GET /cameras, GET /cameras/{id}
    ├── incidents.py      GET /incidents, GET /incidents/{id}, POST /incidents/trigger
    ├── units.py          GET /units, GET /units/{id}
    ├── stations.py       GET /stations
    ├── hospitals.py      GET /hospitals
    ├── decisions.py      GET /decisions/{id}, POST /decisions/{id}/approve, POST /decisions/{id}/override
    ├── scenario.py       POST /scenario/start, POST /scenario/stop, GET /scenario/status
    └── kpis.py           GET /kpis
```

## Functional Requirements

### 1. State Management & Initialization (`state.py`)
- Loads Pune road network, wards, facilities (stations, hospitals), cameras from processed data.
- Initializes fleet roster (ambulances, police patrol cruisers, fire engines) positioned at real stations.
- Tracks active and archived `Incident` objects, `DispatchDecision` records, and real-time `Unit` telemetry.
- Thread-safe access via asyncio locks.

### 2. Services & Event Pipeline (`services.py`)
- **Vision Ingest Service:** Ingests `Event` models (from simulated loops, files, or API triggers).
- **Fusion Service:** Associates camera events with civilian 112 calls within spatial-temporal radii; computes updated Bayesian log-odds posterior.
- **Dispatch Service:**
  - Evaluates confidence policy (C2).
  - If high confidence: triggers `optimization.joint_dispatch` (C1) CP-SAT solver to generate `DispatchDecision` with `Explanation`.
  - In `auto` mode: automatically executes the dispatch decision.
  - In `assist` mode: places decision in `PENDING_APPROVAL` for human operator review.
- **Pre-Alert & Green Corridor:** Emits simulated hospital notification and green-corridor traffic clearance events.

### 3. Real-Time Unit Simulator (`unit_sim.py`)
- Background `asyncio.Task` ticking at 10 Hz (configurable simulation speed multiplier: 1x, 5x, 10x).
- Moves assigned units along shortest-path nodes from current position to incident coordinates.
- Manages state transitions: `IDLE` → `DISPATCHED` → `ON_SCENE` → `TRANSPORTING` → `AT_HOSPITAL` → `RETURNING` → `IDLE`.
- Broadcasts updated unit GPS coordinates (`GeoPoint`) to all connected WebSocket clients.

### 4. Operator Audit Log (`audit.py`)
- Every operator intervention (`approve`, `override`, `mode change`) is appended to `results/audit/operator_actions.jsonl`.
- Schema: `{timestamp, operator_id, decision_id, action, previous_units, final_units, override_reason, client_ip}`.
- Immutable, append-only file log adhering to DPDP 2023 accountability rules.

### 5. WebSocket Protocol (`/ws`)
- Client receives structured JSON events:
  - `{"type": "init", "data": {...}}`
  - `{"type": "incident_update", "data": Incident}`
  - `{"type": "decision_ready", "data": DispatchDecision}`
  - `{"type": "units_tick", "data": list[Unit]}`
  - `{"type": "kpis", "data": KPIs}`
  - `{"type": "audit", "data": AuditEntry}`

### 6. REST API Endpoints
- `GET /cameras`, `GET /cameras/{id}`
- `GET /incidents`, `GET /incidents/{id}`, `POST /incidents/trigger`
- `GET /units`, `GET /units/{id}`
- `GET /stations`, `GET /hospitals`
- `GET /decisions`, `GET /decisions/{id}`
- `POST /decisions/{id}/approve`
- `POST /decisions/{id}/override` (payload: `{reason: str, units: dict}`)
- `POST /scenario/start`, `POST /scenario/stop`, `GET /scenario/status`
- `GET /kpis`

## Acceptance Criteria
- `make api` starts the backend on port 8000.
- `POST /incidents/trigger` triggers an accident, computes Bayesian posterior, executes joint CP-SAT dispatch, and broadcasts decision and moving units over `/ws`.
- Approvals and overrides log to the audit file.
- All unit and integration tests in `tests/backend/` pass.
- `test_architecture.py` confirms no boundaries are violated.
