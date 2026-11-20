"""Scenario control endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from resqcity.backend.bus import broadcaster
from resqcity.backend.state import state

router = APIRouter(prefix="/scenario", tags=["scenario"])


class StartScenarioRequest(BaseModel):
    scenario: str = "normal"  # "normal", "monsoon", "ganeshotsav"
    mode: str = "assist"  # "assist" or "auto"


@router.post("/start")
async def start_scenario(req: StartScenarioRequest) -> dict[str, Any]:
    state.scenario = req.scenario
    state.mode = req.mode
    state.scenario_running = True
    await broadcaster.broadcast(
        "scenario_status",
        {
            "scenario": state.scenario,
            "mode": state.mode,
            "running": state.scenario_running,
        },
    )
    return {
        "status": "started",
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
    }


@router.post("/stop")
async def stop_scenario() -> dict[str, Any]:
    state.scenario_running = False
    await broadcaster.broadcast(
        "scenario_status",
        {
            "scenario": state.scenario,
            "mode": state.mode,
            "running": state.scenario_running,
        },
    )
    return {
        "status": "stopped",
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
    }


@router.get("/status")
async def scenario_status() -> dict[str, Any]:
    return {
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
    }
