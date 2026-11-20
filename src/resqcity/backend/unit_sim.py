"""Real-time unit movement simulator for live mode (Phase 8).

Moves emergency vehicles along shortest path routes on the Pune road network,
transitioning through EN_ROUTE, ON_SCENE, TRANSPORTING, and RETURNING states.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from resqcity.common.schemas import GeoPoint, Unit, UnitState

if TYPE_CHECKING:
    from resqcity.backend.bus import WebSocketBroadcaster
    from resqcity.backend.state import AppState

log = logging.getLogger(__name__)


@dataclass
class ActiveMission:
    unit_id: str
    incident_id: str
    home_station_id: str
    home_node: int
    home_location: GeoPoint
    hospital_id: str | None = None
    hospital_node: int | None = None

    phase: str = (
        "to_incident"  # "to_incident", "on_scene", "to_hospital", "at_hospital", "returning"
    )
    path_coords: list[tuple[float, float]] = field(default_factory=list)
    coord_index: int = 0
    phase_timer_s: float = 0.0
    service_duration_s: float = 6.0


def _nodes_to_coords(network: Any, node_ids: list[int]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for nid in node_ids:
        try:
            idx = network.index_of(nid)
            lat = float(network.nodes.at[idx, "lat"])
            lon = float(network.nodes.at[idx, "lon"])
            coords.append((lat, lon))
        except Exception:
            pass
    return coords


class RealTimeUnitSimulator:
    """Manages active vehicle journeys on the road network in live operations."""

    def __init__(self, speed_multiplier: float = 8.0) -> None:
        self.speed_multiplier = speed_multiplier
        self.missions: dict[str, ActiveMission] = {}
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def start(self, state: AppState, broadcaster: WebSocketBroadcaster) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(state, broadcaster))
        log.info("Started real-time unit simulator loop")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        log.info("Stopped real-time unit simulator loop")

    async def assign_unit(
        self,
        state: AppState,
        unit_id: str,
        incident_id: str,
        hospital_id: str | None = None,
    ) -> bool:
        """Assign unit to an incident and begin navigation."""
        unit = await state.get_unit(unit_id)
        incident = await state.get_incident(incident_id)
        if not unit or not incident or not state.network:
            return False

        home_st = state.stations.get(unit.home_station)
        if not home_st:
            return False

        now = datetime.now(UTC)
        try:
            path_nodes, _ = state.network.shortest_path(unit.node, incident.node, t=now)
        except Exception:
            path_nodes = [unit.node, incident.node]

        # Extract lat/lon coords for the route
        coords: list[tuple[float, float]] = []
        if len(path_nodes) >= 2:
            coords = _nodes_to_coords(state.network, path_nodes)
        if not coords:
            if unit.position:
                coords.append((unit.position.lat, unit.position.lon))
            coords.append((incident.location.lat, incident.location.lon))

        hosp_node = None
        if hospital_id and hospital_id in state.hospitals:
            hosp_node = state.hospitals[hospital_id].node

        mission = ActiveMission(
            unit_id=unit_id,
            incident_id=incident_id,
            home_station_id=unit.home_station,
            home_node=unit.node,
            home_location=home_st.location,
            hospital_id=hospital_id,
            hospital_node=hosp_node,
            phase="to_incident",
            path_coords=coords,
            coord_index=0,
            service_duration_s=6.0,
        )

        async with self._lock:
            self.missions[unit_id] = mission

        updated_unit = unit.model_copy(
            update={"state": UnitState.EN_ROUTE, "position": home_st.location}
        )
        await state.update_unit(updated_unit)
        return True

    async def _step(self, state: AppState, broadcaster: WebSocketBroadcaster, dt: float) -> None:
        async with self._lock:
            active_items = list(self.missions.items())

        if not active_items:
            return

        changed_units: list[Unit] = []
        finished_units: list[str] = []
        now = datetime.now(UTC)

        for unit_id, m in active_items:
            unit = await state.get_unit(unit_id)
            if not unit:
                continue

            if m.phase == "to_incident":
                # Advance along path coordinates
                step_size = max(1, int(self.speed_multiplier * dt * 2))
                m.coord_index += step_size
                if m.coord_index >= len(m.path_coords) - 1:
                    m.coord_index = len(m.path_coords) - 1
                    m.phase = "on_scene"
                    m.phase_timer_s = m.service_duration_s
                    dest_lat, dest_lon = m.path_coords[-1]
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.ON_SCENE,
                            "position": GeoPoint(lat=dest_lat, lon=dest_lon),
                        }
                    )
                else:
                    cur_lat, cur_lon = m.path_coords[m.coord_index]
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.EN_ROUTE,
                            "position": GeoPoint(lat=cur_lat, lon=cur_lon),
                        }
                    )
                changed_units.append(unit)

            elif m.phase == "on_scene":
                m.phase_timer_s -= dt * self.speed_multiplier
                if m.phase_timer_s <= 0:
                    if m.hospital_id and m.hospital_node and state.network:
                        # Route to hospital
                        inc = await state.get_incident(m.incident_id)
                        inc_node = inc.node if inc else unit.node
                        try:
                            p, _ = state.network.shortest_path(inc_node, m.hospital_node, t=now)
                        except Exception:
                            p = [inc_node, m.hospital_node]
                        fallback = [(unit.position.lat, unit.position.lon)] if unit.position else []
                        m.path_coords = _nodes_to_coords(state.network, p) or fallback
                        m.coord_index = 0
                        m.phase = "to_hospital"
                        unit = unit.model_copy(update={"state": UnitState.TRANSPORTING})
                    else:
                        # Return to home station
                        try:
                            p, _ = state.network.shortest_path(unit.node, m.home_node, t=now)
                        except Exception:
                            p = [unit.node, m.home_node]
                        fallback = [(unit.position.lat, unit.position.lon)] if unit.position else []
                        m.path_coords = _nodes_to_coords(state.network, p) or fallback
                        m.coord_index = 0
                        m.phase = "returning"
                        unit = unit.model_copy(update={"state": UnitState.RETURNING})
                    changed_units.append(unit)

            elif m.phase == "to_hospital":
                step_size = max(1, int(self.speed_multiplier * dt * 2))
                m.coord_index += step_size
                if m.coord_index >= len(m.path_coords) - 1:
                    m.coord_index = len(m.path_coords) - 1
                    m.phase = "at_hospital"
                    m.phase_timer_s = 4.0
                    dest_lat, dest_lon = m.path_coords[-1]
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.AT_HOSPITAL,
                            "position": GeoPoint(lat=dest_lat, lon=dest_lon),
                        }
                    )
                else:
                    cur_lat, cur_lon = m.path_coords[m.coord_index]
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.TRANSPORTING,
                            "position": GeoPoint(lat=cur_lat, lon=cur_lon),
                        }
                    )
                changed_units.append(unit)

            elif m.phase == "at_hospital":
                m.phase_timer_s -= dt * self.speed_multiplier
                if m.phase_timer_s <= 0:
                    if m.hospital_node and state.network:
                        try:
                            p, _ = state.network.shortest_path(m.hospital_node, m.home_node, t=now)
                        except Exception:
                            p = [m.hospital_node, m.home_node]
                        fallback = [(unit.position.lat, unit.position.lon)] if unit.position else []
                        m.path_coords = _nodes_to_coords(state.network, p) or fallback
                        m.coord_index = 0
                    m.phase = "returning"
                    unit = unit.model_copy(update={"state": UnitState.RETURNING})
                    changed_units.append(unit)

            elif m.phase == "returning":
                step_size = max(1, int(self.speed_multiplier * dt * 2))
                m.coord_index += step_size
                if m.coord_index >= len(m.path_coords) - 1:
                    # Mission complete! Return to idle at home station
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.IDLE,
                            "position": m.home_location,
                            "node": m.home_node,
                        }
                    )
                    finished_units.append(unit_id)
                else:
                    cur_lat, cur_lon = m.path_coords[m.coord_index]
                    unit = unit.model_copy(
                        update={
                            "state": UnitState.RETURNING,
                            "position": GeoPoint(lat=cur_lat, lon=cur_lon),
                        }
                    )
                changed_units.append(unit)

            await state.update_unit(unit)

        if finished_units:
            async with self._lock:
                for uid in finished_units:
                    self.missions.pop(uid, None)

        if changed_units:
            await broadcaster.broadcast("units_tick", changed_units)

    async def _run_loop(self, state: AppState, broadcaster: WebSocketBroadcaster) -> None:
        dt = 0.5
        while self._running:
            try:
                await self._step(state, broadcaster, dt)
            except Exception as e:
                log.error("Error in unit simulator loop: %s", e)
            await asyncio.sleep(dt)


unit_simulator = RealTimeUnitSimulator()
