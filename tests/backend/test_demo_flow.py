"""End-to-end integration test for Phase 9 demo flow:
Trigger accident -> decision appears -> approve -> units move.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from resqcity.backend.app import app
from resqcity.backend.state import state
from resqcity.backend.unit_sim import unit_simulator
from resqcity.common.schemas import IncidentStatus, UnitState


def test_full_e2e_demo_loop():
    """Verify trigger accident -> decision appears -> approve -> units move."""
    with TestClient(app) as client:
        # 1. Trigger accident
        trig_resp = client.post(
            "/incidents/trigger",
            json={
                "lat": 18.5204,
                "lon": 73.8567,
                "severity": 0.8,
                "incident_type": "accident",
            },
        )
        assert trig_resp.status_code == 200
        incident = trig_resp.json()
        inc_id = incident["id"]
        assert incident["status"] == "confirmed"

        # 2. Decision appears
        decs_resp = client.get("/decisions")
        assert decs_resp.status_code == 200
        decisions = decs_resp.json()
        matching = [d for d in decisions if d["incident_id"] == inc_id]
        assert len(matching) >= 1
        decision = matching[0]
        dec_id = decision["id"]
        assert decision["status"] == "proposed"
        assert len(decision["assignments"]) >= 1

        assigned_unit_ids = [a["unit_id"] for a in decision["assignments"]]

        # 3. Approve decision
        appr_resp = client.post(
            f"/decisions/{dec_id}/approve",
            json={"operator_id": "operator_pune_eoc"},
        )
        assert appr_resp.status_code == 200
        assert appr_resp.json()["status"] == "approved"

        # Verify incident transitioned to DISPATCHED
        inc_check = client.get(f"/incidents/{inc_id}").json()
        assert inc_check["status"] == IncidentStatus.DISPATCHED.value

        # Verify assigned units immediately transition to EN_ROUTE
        units_resp_initial = client.get("/units").json()
        units_map_initial = {u["id"]: u for u in units_resp_initial}
        for uid in assigned_unit_ids:
            assert uid in units_map_initial
            assert units_map_initial[uid]["state"] == UnitState.EN_ROUTE.value
            assert units_map_initial[uid]["position"] is not None

        # 4. Units move (advance simulator discrete ticks)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(unit_simulator.tick(state, delta_s=1.0))
        finally:
            loop.close()

        # Check moving units
        units_resp_moving = client.get("/units").json()
        units_map_moving = {u["id"]: u for u in units_resp_moving}
        active_states = {
            UnitState.EN_ROUTE.value,
            UnitState.ON_SCENE.value,
            UnitState.RETURNING.value,
        }
        assert any(units_map_moving[uid]["state"] in active_states for uid in assigned_unit_ids)
