"""FastAPI application entry point for ResQCity backend (Phase 8)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from resqcity.backend.bus import broadcaster
from resqcity.backend.routes.cameras import router as cameras_router
from resqcity.backend.routes.decisions import router as decisions_router
from resqcity.backend.routes.hospitals import router as hospitals_router
from resqcity.backend.routes.incidents import router as incidents_router
from resqcity.backend.routes.kpis import router as kpis_router
from resqcity.backend.routes.scenario import router as scenario_router
from resqcity.backend.routes.stations import router as stations_router
from resqcity.backend.routes.units import router as units_router
from resqcity.backend.state import state
from resqcity.backend.unit_sim import unit_simulator

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Starting ResQCity Backend services...")
    await state.initialize()
    unit_simulator.start(state, broadcaster)
    yield
    log.info("Shutting down ResQCity Backend services...")
    await unit_simulator.stop()


app = FastAPI(
    title="ResQCity Control Room API",
    description="Live CCTV-triggered emergency dispatch decision-support backend for Pune.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(cameras_router)
app.include_router(incidents_router)
app.include_router(units_router)
app.include_router(stations_router)
app.include_router(hospitals_router)
app.include_router(decisions_router)
app.include_router(scenario_router)
app.include_router(kpis_router)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "resqcity-backend",
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": state.mode,
        "scenario": state.scenario,
    }


@app.get("/wards")
async def get_wards() -> dict[str, Any]:
    return state.wards_geojson


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await broadcaster.connect(websocket)
    try:
        # Send initial snapshot to client
        snapshot = {
            "type": "init",
            "data": {
                "cameras": [c.model_dump(mode="json") for c in state.cameras.values()],
                "units": [u.model_dump(mode="json") for u in state.units.values()],
                "stations": [s.model_dump(mode="json") for s in state.stations.values()],
                "hospitals": [h.model_dump(mode="json") for h in state.hospitals.values()],
                "kpis": await state.get_kpis(),
            },
        }
        await websocket.send_text(json.dumps(snapshot, default=str))

        while True:
            # Keep connection alive; handle incoming client pings
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        await broadcaster.disconnect(websocket)
    except Exception as e:
        log.warning("WebSocket exception: %s", e)
        await broadcaster.disconnect(websocket)
