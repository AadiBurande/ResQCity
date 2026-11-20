"""Hospital endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from resqcity.backend.state import state
from resqcity.common.schemas import Hospital

router = APIRouter(prefix="/hospitals", tags=["hospitals"])


@router.get("", response_model=list[Hospital])
async def list_hospitals() -> list[Hospital]:
    return list(state.hospitals.values())


@router.get("/{hospital_id}", response_model=Hospital)
async def get_hospital(hospital_id: str) -> Hospital:
    if hospital_id not in state.hospitals:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return state.hospitals[hospital_id]
