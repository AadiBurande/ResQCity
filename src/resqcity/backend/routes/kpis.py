"""Key Performance Indicator (KPI) endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from resqcity.backend.state import state

router = APIRouter(prefix="/kpis", tags=["kpis"])


@router.get("")
async def get_kpis() -> dict[str, Any]:
    return await state.get_kpis()
