"""Live scenario background simulation runner (Phase 8/9).

Generates real-time environmental stress, periodic CCTV incidents, and live dispatches
when a scenario is started from the control-room dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resqcity.backend.state import AppState

log = logging.getLogger(__name__)


class ScenarioRunner:
    """Manages active scenario loops, weather simulation, and synthetic incident feeds."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self, state: AppState) -> None:
        if self._running:
            self.stop()
        self._running = True
        state.scenario_running = True
        self._task = asyncio.create_task(self._run_loop(state))
        log.info("Started live scenario runner for %s (%s)", state.scenario, state.mode)

    def stop(self, state: AppState | None = None) -> None:
        self._running = False
        if state:
            state.scenario_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        log.info("Stopped live scenario runner")

    async def _run_loop(self, state: AppState) -> None:
        from resqcity.backend.bus import broadcaster

        try:
            # Trigger initial incident immediately on startup for instant visual feedback
            await self._trigger_scenario_incident(state)

            while self._running and state.scenario_running:
                # Interval between incidents (15-25 seconds)
                await asyncio.sleep(18.0)
                if not self._running or not state.scenario_running:
                    break
                await self._trigger_scenario_incident(state)

                # Broadcast status update
                rain = 24.5 if state.scenario == "monsoon" else 0.0
                await broadcaster.broadcast(
                    "scenario_status",
                    {
                        "scenario": state.scenario,
                        "mode": state.mode,
                        "running": True,
                        "rain_mm_h": rain,
                    },
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Error in scenario runner loop: %s", e)

    async def _trigger_scenario_incident(self, state: AppState) -> None:
        from resqcity.backend.services import trigger_incident

        if not state.cameras:
            return

        cams = list(state.cameras.values())
        cam = random.choice(cams)

        if state.scenario == "monsoon":
            incident_type = random.choice(["accident", "accident", "other"])
            severity = random.uniform(0.65, 0.95)
        elif state.scenario == "ganeshotsav":
            incident_type = random.choice(["crowd", "accident", "fire_smoke"])
            severity = random.uniform(0.70, 0.95)
        else:
            incident_type = random.choice(["accident", "accident", "fire_smoke"])
            severity = random.uniform(0.40, 0.85)

        try:
            await trigger_incident(
                state=state,
                camera_id=cam.id,
                severity=severity,
                incident_type=incident_type,
            )
            log.info(
                "Scenario runner triggered %s (%s) at camera %s (%s)",
                incident_type,
                state.scenario,
                cam.id,
                cam.ward_id,
            )
        except Exception as e:
            log.warning("Scenario runner failed to trigger incident: %s", e)


scenario_runner = ScenarioRunner()
