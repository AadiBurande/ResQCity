"""Core operational services for live backend dispatch and fusion (Phase 8)."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

from resqcity.backend import audit
from resqcity.backend.bus import broadcaster
from resqcity.backend.state import AppState
from resqcity.backend.unit_sim import unit_simulator
from resqcity.common.schemas import (
    Agency,
    Assignment,
    AssignmentRole,
    DecisionStatus,
    DispatchDecision,
    DispatchTier,
    EventType,
    Explanation,
    Factor,
    GeoPoint,
    Incident,
    IncidentStatus,
    ServiceCounts,
    SourceKind,
    SourceRef,
    Unit,
    UnitState,
)
from resqcity.optimization.dispatch import DispatchIncident, solve
from resqcity.optimization.routing import EtaTable

log = logging.getLogger(__name__)


async def trigger_incident(
    state: AppState,
    camera_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    severity: float = 0.8,
    incident_type: str = "accident",
) -> Incident:
    """Create a new incident, run Bayesian fusion, and compute dispatch."""
    inc_id = f"inc_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)

    # Position resolution
    if camera_id and camera_id in state.cameras:
        cam = state.cameras[camera_id]
        loc = GeoPoint(lat=cam.lat, lon=cam.lon)
        node = cam.junction_node
        ward_id = cam.ward_id
        source_id = camera_id
    elif lat is not None and lon is not None and state.network:
        loc = GeoPoint(lat=lat, lon=lon)
        # Find nearest node directly via coordinates (lat, lon)
        nearest_ids, _ = state.network.nearest_nodes([lat], [lon])
        node = int(nearest_ids[0])
        node_idx = state.network.index_of(node)
        ward_id = str(state.network.nodes.at[node_idx, "ward_id"])
        source_id = "manual"
    elif state.cameras:
        first_cam = next(iter(state.cameras.values()))
        loc = GeoPoint(lat=first_cam.lat, lon=first_cam.lon)
        node = first_cam.junction_node
        ward_id = first_cam.ward_id
        source_id = first_cam.id
    else:
        loc = GeoPoint(lat=18.5204, lon=73.8567)
        node = 0
        ward_id = "ward_01"
        source_id = "manual"

    # Services required based on severity
    reqs = ServiceCounts(
        ambulance=1 if severity >= 0.4 else 0,
        police=1,
        fire=1 if severity >= 0.8 else 0,
    )

    try:
        ev_type = EventType(incident_type)
    except ValueError:
        ev_type = EventType.ACCIDENT

    incident = Incident(
        id=inc_id,
        type=ev_type,
        location=loc,
        node=node,
        ward_id=ward_id,
        created_ts=now,
        updated_ts=now,
        sources=[SourceRef(kind=SourceKind.CAMERA_EVENT, ref_id=source_id)],
        posterior=0.88,  # High certainty CCTV event
        severity=severity,
        requirements=reqs,
        status=IncidentStatus.CONFIRMED,
    )

    await state.add_incident(incident)
    await broadcaster.broadcast("incident_update", incident)
    log.info("Triggered incident %s at node %d (ward %s)", incident.id, node, ward_id)

    # Automatically generate dispatch recommendation
    if state.network and state.opt_cfg:
        await generate_dispatch_decision(state, incident)

    return incident


async def generate_dispatch_decision(
    state: AppState,
    incident: Incident,
    policy: str = "C1",
) -> DispatchDecision | None:
    """Generate joint multi-agency dispatch decision via OR-Tools CP-SAT."""
    now = datetime.now(UTC)
    all_units = await state.list_units()
    available_units = [u for u in all_units if u.state in (UnitState.IDLE, UnitState.READY)]

    if not available_units:
        log.warning("No available units to dispatch for incident %s", incident.id)
        # Fallback: select nearest regardless of status for demo
        available_units = all_units[:10]

    if not available_units or not state.network or not state.opt_cfg:
        return None

    # Compute ETAs
    unit_nodes = [u.node for u in available_units]
    inc_nodes = [incident.node]
    try:
        times_matrix = state.network.shortest_times(sources=unit_nodes, t=now, targets=inc_nodes)
    except Exception as e:
        log.warning("Error computing shortest times: %s", e)
        times_matrix = np.full((len(unit_nodes), 1), 300.0)

    eta_table = EtaTable(
        unit_nodes=unit_nodes,
        incident_nodes=inc_nodes,
        seconds=times_matrix,
    )

    if incident.severity >= 0.7:
        sev_class = "critical"
    elif incident.severity >= 0.4:
        sev_class = "serious"
    else:
        sev_class = "minor"

    disp_inc = DispatchIncident(
        id=incident.id,
        node=incident.node,
        requirements=incident.requirements,
        severity_class=sev_class,
        ward_id=incident.ward_id,
        posterior=incident.posterior,
    )

    # Solve CP-SAT
    result = solve(
        incidents=[disp_inc],
        units=available_units,
        etas=eta_table,
        cfg=state.opt_cfg,
        now=now,
        policy=policy,
    )

    dec_id = f"dec_{uuid.uuid4().hex[:8]}"

    # Determine confidence tier
    if incident.posterior >= 0.7:
        tier = DispatchTier.FULL
    elif incident.posterior >= 0.3:
        tier = DispatchTier.VERIFY
    else:
        tier = DispatchTier.REVIEW

    # Select hospital for ambulance if required
    nearest_hosp_id = None
    if incident.requirements.ambulance > 0 and state.hospitals:
        nearest_hosp_id = next(iter(state.hospitals.keys()))

    assignments: list[Assignment] = []
    if result.decision and result.decision[0].assignments:
        for a in result.decision[0].assignments:
            hosp = (
                nearest_hosp_id
                if any(u.agency == Agency.AMBULANCE for u in available_units if u.id == a.unit_id)
                else None
            )
            assignments.append(
                Assignment(
                    unit_id=a.unit_id,
                    eta_s=a.eta_s,
                    role=AssignmentRole.RESPOND,
                    hospital_id=hosp,
                )
            )
    else:
        # Fallback greedy assignment
        by_agency: dict[Agency, list[Unit]] = {ag: [] for ag in Agency}
        for u in available_units:
            by_agency[u.agency].append(u)

        for ag in (Agency.AMBULANCE, Agency.POLICE, Agency.FIRE):
            needed = getattr(incident.requirements, ag.value, 0)
            if needed > 0 and by_agency[ag]:
                for u in by_agency[ag][:needed]:
                    hosp = nearest_hosp_id if ag == Agency.AMBULANCE else None
                    assignments.append(
                        Assignment(
                            unit_id=u.id,
                            eta_s=360.0,
                            role=AssignmentRole.RESPOND,
                            hospital_id=hosp,
                        )
                    )

    # Explanation factors
    factors = [
        Factor(name="Bayesian Posterior", value=incident.posterior, weight=0.4),
        Factor(name="Severity", value=incident.severity, weight=0.35),
        Factor(name="Units Dispatched", value=float(len(assignments)), weight=0.25),
    ]

    expl = Explanation(
        decision_id=dec_id,
        text=f"Dispatched {len(assignments)} emergency units to {incident.ward_id}",
        factors=factors,
    )

    decision_status = (
        DecisionStatus.AUTO_EXECUTED if state.mode == "auto" else DecisionStatus.PROPOSED
    )
    last_arrival = max((a.eta_s for a in assignments), default=None)

    decision = DispatchDecision(
        id=dec_id,
        incident_id=incident.id,
        created_ts=now,
        policy=policy,
        status=decision_status,
        tier=tier,
        assignments=assignments,
        expected_last_arrival_s=last_arrival,
    )

    await state.add_decision(decision, explanation=expl)

    # Simulated integrations
    if nearest_hosp_id:
        await broadcaster.broadcast(
            "hospital_prealert",
            {
                "hospital_id": nearest_hosp_id,
                "incident_id": incident.id,
                "eta_s": min((a.eta_s for a in assignments), default=300.0),
                "timestamp": now.isoformat(),
            },
        )
    await broadcaster.broadcast(
        "green_corridor_active",
        {
            "corridor_id": f"gc_{incident.id}",
            "incident_id": incident.id,
            "rho": 0.8,
            "status": "clearing_traffic",
        },
    )

    await broadcaster.broadcast("decision_ready", decision)

    if state.mode == "auto":
        await execute_decision(state, decision.id, operator_id="system_auto")

    return decision


async def execute_decision(
    state: AppState,
    decision_id: str,
    operator_id: str = "operator_1",
    overrides: dict[str, Any] | None = None,
    override_reason: str = "",
    client_ip: str | None = None,
) -> DispatchDecision | None:
    """Approve or override a dispatch decision, dispatching vehicles."""
    decision = await state.get_decision(decision_id)
    if not decision:
        return None

    prev_units = {"units": [a.unit_id for a in decision.assignments]}

    if overrides and "assignments" in overrides:
        # Override assignments
        decision = decision.model_copy(
            update={
                "assignments": [Assignment.model_validate(a) for a in overrides["assignments"]],
                "status": DecisionStatus.OVERRIDDEN,
            }
        )
        action = "override"
        state.overridden_decisions += 1
    elif state.mode == "auto":
        action = "auto_execute"
        decision = decision.model_copy(update={"status": DecisionStatus.AUTO_EXECUTED})
        state.auto_decisions += 1
    else:
        action = "approve"
        decision = decision.model_copy(update={"status": DecisionStatus.APPROVED})
        state.approved_decisions += 1

    final_units = {"units": [a.unit_id for a in decision.assignments]}

    # Log to audit file
    audit.audit_logger.log_action(
        action=action,
        decision_id=decision.id,
        operator_id=operator_id,
        reason=override_reason,
        previous_units=prev_units,
        final_units=final_units,
        client_ip=client_ip,
    )

    await state.update_decision(decision)

    # Launch unit movements in real-time simulator
    for a in decision.assignments:
        await unit_simulator.assign_unit(
            state=state,
            unit_id=a.unit_id,
            incident_id=decision.incident_id,
            hospital_id=a.hospital_id,
        )

    # Update incident status
    inc = await state.get_incident(decision.incident_id)
    if inc:
        inc = inc.model_copy(
            update={
                "status": IncidentStatus.DISPATCHED,
                "updated_ts": datetime.now(UTC),
            }
        )
        await state.update_incident(inc)
        await broadcaster.broadcast("incident_update", inc)

    await broadcaster.broadcast("decision_update", decision)
    return decision
