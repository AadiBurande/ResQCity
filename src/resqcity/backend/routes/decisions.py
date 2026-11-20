"""Dispatch decision endpoints (operator in the loop)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from resqcity.backend.services import execute_decision
from resqcity.backend.state import state
from resqcity.common.schemas import DispatchDecision, Explanation

router = APIRouter(prefix="/decisions", tags=["decisions"])


class OverrideDecisionRequest(BaseModel):
    reason: str
    assignments: list[dict[str, Any]] | None = None
    operator_id: str = "operator_1"


class ApproveDecisionRequest(BaseModel):
    operator_id: str = "operator_1"


@router.get("", response_model=list[DispatchDecision])
async def list_decisions() -> list[DispatchDecision]:
    return await state.list_decisions()


@router.get("/{decision_id}", response_model=DispatchDecision)
async def get_decision(decision_id: str) -> DispatchDecision:
    dec = await state.get_decision(decision_id)
    if not dec:
        raise HTTPException(status_code=404, detail="Decision not found")
    return dec


@router.get("/{decision_id}/explanation", response_model=Explanation)
async def get_decision_explanation(decision_id: str) -> Explanation:
    expl = await state.get_explanation(decision_id)
    if not expl:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return expl


@router.post("/{decision_id}/approve", response_model=DispatchDecision)
async def approve_decision(
    decision_id: str,
    req: ApproveDecisionRequest,
    raw_request: Request,
) -> DispatchDecision:
    client_ip = raw_request.client.host if raw_request.client else None
    dec = await execute_decision(
        state=state,
        decision_id=decision_id,
        operator_id=req.operator_id,
        overrides=None,
        override_reason="",
        client_ip=client_ip,
    )
    if not dec:
        raise HTTPException(status_code=404, detail="Decision not found")
    return dec


@router.post("/{decision_id}/override", response_model=DispatchDecision)
async def override_decision(
    decision_id: str,
    req: OverrideDecisionRequest,
    raw_request: Request,
) -> DispatchDecision:
    client_ip = raw_request.client.host if raw_request.client else None
    overrides = {"assignments": req.assignments} if req.assignments else None
    dec = await execute_decision(
        state=state,
        decision_id=decision_id,
        operator_id=req.operator_id,
        overrides=overrides,
        override_reason=req.reason,
        client_ip=client_ip,
    )
    if not dec:
        raise HTTPException(status_code=404, detail="Decision not found")
    return dec
