"""Scenario control endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from resqcity.backend.bus import broadcaster
from resqcity.backend.scenario_runner import scenario_runner
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
    scenario_runner.start(state)

    rain = 24.5 if state.scenario == "monsoon" else 0.0
    await broadcaster.broadcast(
        "scenario_status",
        {
            "scenario": state.scenario,
            "mode": state.mode,
            "running": state.scenario_running,
            "rain_mm_h": rain,
        },
    )
    return {
        "status": "started",
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
        "rain_mm_h": rain,
    }


@router.post("/stop")
async def stop_scenario() -> dict[str, Any]:
    state.scenario_running = False
    scenario_runner.stop(state)
    await broadcaster.broadcast(
        "scenario_status",
        {
            "scenario": state.scenario,
            "mode": state.mode,
            "running": state.scenario_running,
            "rain_mm_h": 0.0,
        },
    )
    return {
        "status": "stopped",
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
        "rain_mm_h": 0.0,
    }


@router.get("/status")
async def scenario_status() -> dict[str, Any]:
    rain = 24.5 if state.scenario == "monsoon" else 0.0
    return {
        "scenario": state.scenario,
        "mode": state.mode,
        "running": state.scenario_running,
        "rain_mm_h": rain,
    }
