"""Integration tests for FastAPI backend routes, WebSocket, and audit logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from resqcity.backend.app import app
from resqcity.backend.audit import AuditLogger


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan executed."""
    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "resqcity-backend"


def test_list_and_get_cameras(client: TestClient):
    resp = client.get("/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert len(cameras) >= 1
    first_id = cameras[0]["id"]

    resp_single = client.get(f"/cameras/{first_id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["id"] == first_id

    resp_404 = client.get("/cameras/non_existent_camera")
    assert resp_404.status_code == 404


def test_list_and_get_stations(client: TestClient):
    resp = client.get("/stations")
    assert resp.status_code == 200
    stations = resp.json()
    assert len(stations) >= 1
    first_id = stations[0]["id"]

    resp_single = client.get(f"/stations/{first_id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["id"] == first_id


def test_list_and_get_hospitals(client: TestClient):
    resp = client.get("/hospitals")
    assert resp.status_code == 200
    hospitals = resp.json()
    assert len(hospitals) >= 1
    first_id = hospitals[0]["id"]

    resp_single = client.get(f"/hospitals/{first_id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["id"] == first_id


def test_list_and_get_units(client: TestClient):
    resp = client.get("/units")
    assert resp.status_code == 200
    units = resp.json()
    assert len(units) >= 1
    first_id = units[0]["id"]

    resp_single = client.get(f"/units/{first_id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["id"] == first_id


def test_get_kpis(client: TestClient):
    resp = client.get("/kpis")
    assert resp.status_code == 200
    kpis = resp.json()
    assert "total_units" in kpis
    assert "fleet_utilisation" in kpis


def test_scenario_start_and_stop(client: TestClient):
    resp_start = client.post("/scenario/start", json={"scenario": "monsoon", "mode": "assist"})
    assert resp_start.status_code == 200
    data = resp_start.json()
    assert data["scenario"] == "monsoon"
    assert data["running"] is True

    resp_status = client.get("/scenario/status")
    assert resp_status.status_code == 200
    assert resp_status.json()["scenario"] == "monsoon"

    resp_stop = client.post("/scenario/stop")
    assert resp_stop.status_code == 200
    assert resp_stop.json()["running"] is False


def test_trigger_incident_and_decision_lifecycle(client: TestClient, tmp_path: Path):
    # Setup custom audit log to verify logging
    audit_file = tmp_path / "test_audit.jsonl"
    test_logger = AuditLogger(audit_file)
    from resqcity.backend import audit

    original_logger = audit.audit_logger
    audit.audit_logger = test_logger

    try:
        # 1. Trigger incident via camera
        cams_resp = client.get("/cameras")
        cam_id = cams_resp.json()[0]["id"]

        trig_resp = client.post(
            "/incidents/trigger",
            json={
                "camera_id": cam_id,
                "severity": 0.85,
                "incident_type": "accident",
            },
        )
        assert trig_resp.status_code == 200
        incident = trig_resp.json()
        inc_id = incident["id"]
        assert inc_id.startswith("inc_")
        assert incident["status"] == "confirmed"

        # 2. Verify incident in list
        inc_list = client.get("/incidents").json()
        assert any(i["id"] == inc_id for i in inc_list)

        # 3. Verify generated decision
        dec_list = client.get("/decisions").json()
        matching_decs = [d for d in dec_list if d["incident_id"] == inc_id]
        assert len(matching_decs) >= 1
        decision = matching_decs[0]
        dec_id = decision["id"]
        assert decision["status"] == "proposed"
        assert len(decision["assignments"]) >= 1

        # 4. Approve decision
        appr_resp = client.post(
            f"/decisions/{dec_id}/approve",
            json={"operator_id": "test_operator"},
        )
        assert appr_resp.status_code == 200
        assert appr_resp.json()["status"] == "approved"

        # 5. Trigger another incident and override it
        trig2 = client.post(
            "/incidents/trigger",
            json={
                "lat": 18.5204,
                "lon": 73.8567,
                "severity": 0.5,
                "incident_type": "accident",
            },
        ).json()
        dec2 = next(d for d in client.get("/decisions").json() if d["incident_id"] == trig2["id"])

        ovr_resp = client.post(
            f"/decisions/{dec2['id']}/override",
            json={
                "operator_id": "test_operator",
                "reason": "Traffic congestion on primary arterial",
                "assignments": [
                    {
                        "unit_id": dec2["assignments"][0]["unit_id"],
                        "eta_s": 420.0,
                        "role": "respond",
                    }
                ],
            },
        )
        assert ovr_resp.status_code == 200
        assert ovr_resp.json()["status"] == "overridden"

        # 6. Verify audit file written
        entries = test_logger.read_entries()
        assert len(entries) >= 2
        actions = [e.action for e in entries]
        assert "approve" in actions
        assert "override" in actions
    finally:
        audit.audit_logger = original_logger


def test_websocket_connection(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["type"] == "init"
        assert "cameras" in data["data"]
        assert "units" in data["data"]

        # Test ping/pong
        ws.send_text(json.dumps({"type": "ping"}))
        pong = ws.receive_text()
        assert json.loads(pong)["type"] == "pong"
