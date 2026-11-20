"""In-memory operational state for ResQCity backend (Phase 8).

Maintains thread-safe registries for cameras, stations, hospitals, emergency units,
active incidents, and dispatch decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import yaml

from resqcity.common.config import PROJECT_ROOT, load_base_config
from resqcity.common.schemas import (
    Camera,
    DispatchDecision,
    Explanation,
    Hospital,
    Incident,
    Station,
    Unit,
    UnitState,
)
from resqcity.geo.config import GeoConfig, load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import (
    OptimizationConfig,
    load_fleet_config,
    load_optimization_config,
)
from resqcity.optimization.fleet import build_fleet

log = logging.getLogger(__name__)


@dataclass
class AppState:
    network: RoadNetwork | None = None
    geo_cfg: GeoConfig | None = None
    opt_cfg: OptimizationConfig | None = None

    cameras: dict[str, Camera] = field(default_factory=dict)
    stations: dict[str, Station] = field(default_factory=dict)
    hospitals: dict[str, Hospital] = field(default_factory=dict)
    wards_geojson: dict[str, Any] = field(default_factory=dict)

    units: dict[str, Unit] = field(default_factory=dict)
    incidents: dict[str, Incident] = field(default_factory=dict)
    decisions: dict[str, DispatchDecision] = field(default_factory=dict)
    explanations: dict[str, Explanation] = field(default_factory=dict)

    mode: str = "assist"  # "assist" or "auto"
    scenario: str = "normal"  # "normal", "monsoon", "ganeshotsav"
    scenario_running: bool = False

    # KPI counters
    total_incidents: int = 0
    approved_decisions: int = 0
    overridden_decisions: int = 0
    auto_decisions: int = 0

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def initialize(self) -> None:
        """Load graph, facilities, cameras, and initialize fleets."""
        async with self._lock:
            base = load_base_config()
            self.geo_cfg = load_geo_config()
            self.opt_cfg = load_optimization_config()
            data_proc = base.paths.resolve("data_processed")

            # Load road network
            log.info("Loading road network...")
            self.network = RoadNetwork.load(data_proc, self.geo_cfg)

            # Load facilities
            fac_path = data_proc / "facilities.json"
            if fac_path.exists():
                with fac_path.open("r", encoding="utf-8") as f:
                    fac_data = json.load(f)
                self.stations = {
                    s["id"]: Station.model_validate(s) for s in fac_data.get("stations", [])
                }
                self.hospitals = {
                    h["id"]: Hospital.model_validate(h) for h in fac_data.get("hospitals", [])
                }

            # Load cameras
            cam_path = PROJECT_ROOT / "configs" / "cameras.yaml"
            if cam_path.exists():
                with cam_path.open("r", encoding="utf-8") as f:
                    cam_data = yaml.safe_load(f)
                self.cameras = {
                    c["id"]: Camera.model_validate(c) for c in cam_data.get("cameras", [])
                }

            # Load wards geojson
            wards_path = data_proc / "wards.geojson"
            if wards_path.exists():
                with wards_path.open("r", encoding="utf-8") as f:
                    self.wards_geojson = json.load(f)

            # Build fleet
            fleet_cfg = load_fleet_config()
            demand = {s.ward_id: 1.0 for s in self.stations.values()}
            fleet_list = build_fleet(
                list(self.stations.values()),
                list(self.hospitals.values()),
                demand,
                fleet_cfg,
            )

            # Set unit initial positions to their home station coordinates
            self.units = {}
            for u in fleet_list:
                home_st = self.stations.get(u.home_station)
                pos = home_st.location if home_st else None
                u_pos = u.model_copy(update={"position": pos, "state": UnitState.IDLE})
                self.units[u.id] = u_pos

            log.info(
                "Initialized state: %d cameras, %d stations, %d hospitals, %d units",
                len(self.cameras),
                len(self.stations),
                len(self.hospitals),
                len(self.units),
            )

    async def add_incident(self, incident: Incident) -> None:
        async with self._lock:
            self.incidents[incident.id] = incident
            self.total_incidents += 1

    async def update_incident(self, incident: Incident) -> None:
        async with self._lock:
            self.incidents[incident.id] = incident

    async def get_incident(self, incident_id: str) -> Incident | None:
        async with self._lock:
            return self.incidents.get(incident_id)

    async def list_incidents(self) -> list[Incident]:
        async with self._lock:
            return list(self.incidents.values())

    async def add_decision(
        self, decision: DispatchDecision, explanation: Explanation | None = None
    ) -> None:
        async with self._lock:
            self.decisions[decision.id] = decision
            if explanation:
                self.explanations[decision.id] = explanation

    async def get_explanation(self, decision_id: str) -> Explanation | None:
        async with self._lock:
            return self.explanations.get(decision_id)

    async def update_decision(self, decision: DispatchDecision) -> None:
        async with self._lock:
            self.decisions[decision.id] = decision

    async def get_decision(self, decision_id: str) -> DispatchDecision | None:
        async with self._lock:
            return self.decisions.get(decision_id)

    async def list_decisions(self) -> list[DispatchDecision]:
        async with self._lock:
            return list(self.decisions.values())

    async def update_unit(self, unit: Unit) -> None:
        async with self._lock:
            self.units[unit.id] = unit

    async def get_unit(self, unit_id: str) -> Unit | None:
        async with self._lock:
            return self.units.get(unit_id)

    async def list_units(self) -> list[Unit]:
        async with self._lock:
            return list(self.units.values())

    async def get_kpis(self) -> dict[str, Any]:
        async with self._lock:
            busy_units = sum(
                1 for u in self.units.values() if u.state not in (UnitState.IDLE, UnitState.READY)
            )
            total_units = max(1, len(self.units))
            util = busy_units / total_units
            total_dec = self.approved_decisions + self.overridden_decisions + self.auto_decisions
            override_rate = (self.overridden_decisions / total_dec) if total_dec > 0 else 0.0
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "mode": self.mode,
                "scenario": self.scenario,
                "scenario_running": self.scenario_running,
                "total_incidents": self.total_incidents,
                "active_incidents": len(self.incidents),
                "total_units": len(self.units),
                "busy_units": busy_units,
                "fleet_utilisation": round(util, 3),
                "approved_decisions": self.approved_decisions,
                "overridden_decisions": self.overridden_decisions,
                "auto_decisions": self.auto_decisions,
                "decisions": {
                    "approved": self.approved_decisions,
                    "overridden": self.overridden_decisions,
                    "auto": self.auto_decisions,
                    "total": total_dec,
                    "override_rate": round(override_rate, 3),
                },
            }


state = AppState()
