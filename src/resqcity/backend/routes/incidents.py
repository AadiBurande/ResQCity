"""Incident endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from resqcity.backend.services import trigger_incident
from resqcity.backend.state import state
from resqcity.common.schemas import Incident

router = APIRouter(prefix="/incidents", tags=["incidents"])


class TriggerIncidentRequest(BaseModel):
    camera_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    severity: float = 0.8
    incident_type: str = "accident"


@router.get("", response_model=list[Incident])
async def list_incidents() -> list[Incident]:
    return await state.list_incidents()


@router.get("/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str) -> Incident:
    inc = await state.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.post("/trigger", response_model=Incident)
async def trigger_new_incident(req: TriggerIncidentRequest) -> Incident:
    return await trigger_incident(
        state=state,
        camera_id=req.camera_id,
        lat=req.lat,
        lon=req.lon,
        severity=req.severity,
        incident_type=req.incident_type,
    )
