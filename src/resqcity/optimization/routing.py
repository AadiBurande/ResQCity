"""Travel times for dispatch: ETAs from units to incidents, and the green corridor (phase 6.1).

Formulation: ``docs/models/routing.md``.

The road graph already carries time-dependent edge speeds (``RoadNetwork.edge_times``) and a
scenario's rain, flooding and festival effects (``ScenarioState.edge_factors``), cached per hour
bucket. This module adds the two things dispatch needs:

1. **ETA matrices the cheap way round.** A batch has many units and few incidents, so one
   *reversed* search per incident gives the travel time from every node to that incident, and the
   unit rows are then read off. The forward alternative is one search per unit.
2. **The green corridor.** When a police unit is assigned to the same incident and can arrive no
   later than the ambulance, the ambulance's remaining travel time is multiplied by ``rho < 1``.
   The precedence is computed from the unmodified ETAs *before* the solver runs, so the
   assistance is data rather than a decision variable (master plan §6.2).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from resqcity.common.schemas import Agency
from resqcity.geo.network import RoadNetwork
from resqcity.geo.scenario import ScenarioState
from resqcity.optimization.config import OptimizationConfig

UNREACHABLE_S = float("inf")


@dataclass
class EtaTable:
    """Travel times (s) from each unit node to each incident node; shape (units, incidents)."""

    unit_nodes: list[int]
    incident_nodes: list[int]
    seconds: np.ndarray

    def __post_init__(self) -> None:
        if self.seconds.shape != (len(self.unit_nodes), len(self.incident_nodes)):
            raise ValueError("eta matrix shape does not match the node lists")

    def reachable(self, unit: int, incident: int) -> bool:
        return bool(np.isfinite(self.seconds[unit, incident]))


class Router:
    """ETAs on one network and scenario. Results depend on the departure time only through the
    hour bucket and the scenario state, which is what the underlying cache keys on."""

    def __init__(
        self,
        network: RoadNetwork,
        cfg: OptimizationConfig,
        scenario: ScenarioState | None = None,
    ) -> None:
        self.network, self.cfg, self.scenario = network, cfg, scenario
        self.profile = cfg.routing.profile

    def eta_table(self, unit_nodes: list[int], incident_nodes: list[int], t: datetime) -> EtaTable:
        """One reversed multi-source search over the incidents, then read off the unit rows."""
        if not incident_nodes:
            return EtaTable(unit_nodes, [], np.zeros((len(unit_nodes), 0)))
        to_incident = self.network.shortest_times(
            incident_nodes, t, self.scenario, self.profile, targets=unit_nodes, reverse=True
        )
        return EtaTable(unit_nodes, incident_nodes, np.asarray(to_incident, float).T)

    def green_corridor_etas(
        self, etas: EtaTable, agencies: list[Agency], assisted: Agency = Agency.AMBULANCE
    ) -> np.ndarray:
        """ETAs after the green corridor, as DATA for the solver.

        For each incident, the fastest police ETA is the earliest a corridor could start. An
        ambulance whose own ETA is at least that is eligible, and its time is scaled by ``rho``.
        An ambulance that would arrive before any police unit is not helped. The scaled time can
        never fall below the police ETA, which is when the corridor can first exist.
        """
        rho = self.cfg.routing.green_corridor.rho.value
        out = etas.seconds.copy()
        if rho >= 1.0:
            return out
        is_police = np.array([a == Agency.POLICE for a in agencies])
        is_assisted = np.array([a == assisted for a in agencies])
        if not is_police.any() or not is_assisted.any():
            return out
        for j in range(out.shape[1]):
            police = out[is_police, j]
            if not np.isfinite(police).any():
                continue
            first_police = float(np.nanmin(np.where(np.isfinite(police), police, np.nan)))
            rows = np.flatnonzero(is_assisted & np.isfinite(out[:, j]))
            eligible = rows[out[rows, j] >= first_police]
            out[eligible, j] = np.maximum(out[eligible, j] * rho, first_police)
        return out


def nearest_unit(
    etas: EtaTable, agencies: list[Agency], agency: Agency, incident: int
) -> int | None:
    """Index of the closest reachable unit of one agency to one incident (the B0/B1 rule)."""
    candidates = [
        u for u, a in enumerate(agencies) if a == agency and np.isfinite(etas.seconds[u, incident])
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda u: (etas.seconds[u, incident], u))
