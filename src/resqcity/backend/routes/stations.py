"""Emergency station endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from resqcity.backend.state import state
from resqcity.common.schemas import Station

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("", response_model=list[Station])
async def list_stations() -> list[Station]:
    return list(state.stations.values())


@router.get("/{station_id}", response_model=Station)
async def get_station(station_id: str) -> Station:
    if station_id not in state.stations:
        raise HTTPException(status_code=404, detail="Station not found")
    return state.stations[station_id]
