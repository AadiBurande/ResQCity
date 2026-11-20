"""Camera endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from resqcity.backend.state import state
from resqcity.common.schemas import Camera

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[Camera])
async def list_cameras() -> list[Camera]:
    return list(state.cameras.values())


@router.get("/{camera_id}", response_model=Camera)
async def get_camera(camera_id: str) -> Camera:
    if camera_id not in state.cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    return state.cameras[camera_id]
