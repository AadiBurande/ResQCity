"""Emergency fleet unit endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from resqcity.backend.state import state
from resqcity.common.schemas import Unit

router = APIRouter(prefix="/units", tags=["units"])


@router.get("", response_model=list[Unit])
async def list_units() -> list[Unit]:
    return await state.list_units()


@router.get("/{unit_id}", response_model=Unit)
async def get_unit(unit_id: str) -> Unit:
    unit = await state.get_unit(unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")
    return unit
