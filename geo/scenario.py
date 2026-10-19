"""Scenario definitions (normal / monsoon / Ganeshotsav) and their effect on the network over time.

A scenario fixes the weather and events of a simulated period. ``ScenarioState`` answers, for any
time t: rain intensity, flood stage, the per-edge speed factor (0 = impassable) and the incident
multiplier near festival routes. Everything is piecewise constant per local hour, so results can be
cached by ``state_key(t)``.
"""

import itertools
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import AwareDatetime, Field, model_validator
from scipy.spatial import cKDTree

from common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_model
from common.schemas import EventType
from geo.network import RoadNetwork

log = logging.getLogger(__name__)
SCENARIO_DIR = CONFIG_DIR / "scenarios"
HOUR = timedelta(hours=1)


class RainConfig(ConfigModel):
    hourly_mm_h: list[float] = Field(min_length=24, max_length=24)
    day_factors: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _non_negative(self) -> Self:
        if min(self.hourly_mm_h) < 0 or min(self.day_factors) < 0:
            raise ValueError("rain intensities and day factors must be non-negative")
        return self


class FloodStage(ConfigModel):
    min_accum_mm: float = Field(ge=0)
    impassable_fraction: float = Field(ge=0, le=1)
    slowed_fraction: float = Field(ge=0, le=1)
    slow_multiplier: float = Field(gt=0, le=1)


class FloodingConfig(ConfigModel):
    buffer_m: float = Field(gt=0)
    window_h: int = Field(ge=1)
    vulnerability_seed: int
    stages: SourcedValue[list[FloodStage]]

    @model_validator(mode="after")
    def _monotone(self) -> Self:
        """More rain must never reopen or speed up a road."""
        if not self.stages.value:
            raise ValueError("flooding needs at least one stage")
        prev = None
        for s in self.stages.value:
            if s.impassable_fraction + s.slowed_fraction > 1:
                raise ValueError("impassable_fraction + slowed_fraction must be <= 1")
            if prev is not None and not (
                s.min_accum_mm > prev.min_accum_mm
                and s.impassable_fraction >= prev.impassable_fraction
                and s.impassable_fraction + s.slowed_fraction
                >= prev.impassable_fraction + prev.slowed_fraction
                and s.slow_multiplier <= prev.slow_multiplier
            ):
                raise ValueError("flood stages must be monotone in rain (see FloodingConfig)")
            prev = s
        return self


class ProcessionRoute(ConfigModel):
    name: str
    street_names: list[str] = Field(default_factory=list)
    polylines: list[list[tuple[float, float]]] = Field(default_factory=list)
    buffer_m: float = Field(gt=0)
    # Street names repeat across the city: only match edges within max_distance_m of center.
    center: tuple[float, float] | None = Field(default=None, description="(lat, lon)")
    max_distance_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _center_and_radius_together(self) -> Self:
        if (self.center is None) != (self.max_distance_m is None):
            raise ValueError("procession route: give both center and max_distance_m, or neither")
        return self


class FestivalWindow(ConfigModel):
    days: list[int] = Field(min_length=1)
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)
    effect: Literal["slowed", "closed"]

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_hour <= self.start_hour:
            raise ValueError("festival window must end after it starts (split windows at midnight)")
        return self


class FestivalConfig(ConfigModel):
    name: str
    routes: list[ProcessionRoute] = Field(min_length=1)
    windows: list[FestivalWindow] = Field(min_length=1)
    slowed_multiplier: SourcedValue[float]
    influence_radius_m: float = Field(ge=0)
    influence_speed_multiplier: SourcedValue[float]
    incident_multiplier_near: SourcedValue[float]
    type_mix_override: SourcedValue[dict[EventType, float]] | None = None

    @model_validator(mode="after")
    def _multipliers_in_range(self) -> Self:
        for name in ("slowed_multiplier", "influence_speed_multiplier"):
            if not 0 < getattr(self, name).value <= 1:
                raise ValueError(f"festival {name} must be in (0, 1]")
        return self


class ScenarioConfig(ConfigModel):
    name: str
    description: str
    synthetic: bool
    start: AwareDatetime
    days: int = Field(ge=1)
    incident_rate_multiplier: float = Field(gt=0)
    rain: RainConfig | None = None
    flooding: FloodingConfig | None = None
    festival: FestivalConfig | None = None

    @model_validator(mode="after")
    def _flooding_needs_rain(self) -> Self:
        if self.flooding is not None and self.rain is None:
            raise ValueError("flooding requires a rain profile")
        return self

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.days)


def load_scenario(name_or_path: str | Path) -> ScenarioConfig:
    path = Path(name_or_path)
    if path.suffix != ".yaml":
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    return load_model(ScenarioConfig, path)


def available_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def _densify(coords: list[tuple[float, float]], step: float) -> np.ndarray:
    """Points every ``step`` metres along a projected polyline (for KD-tree distance queries)."""
    pts = [np.asarray(coords[0], float)]
    for a, b in itertools.pairwise(coords):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, math.ceil(np.linalg.norm(b - a) / step))
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.vstack(pts)


class ScenarioState:
    """Time-dependent conditions of one scenario on one road network."""

    def __init__(
        self, cfg: ScenarioConfig, network: RoadNetwork, timezone: str = "Asia/Kolkata"
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.tz = ZoneInfo(timezone)
        self._local_start = cfg.start.astimezone(self.tz).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        n_edges = network.n_edges
        self._factor_cache: dict[tuple[object, ...], np.ndarray] = {}

        # Flooding: fixed vulnerability per ROAD (both directions share it), so the same roads
        # flood first in every run and a flooded two-way road is closed in both directions.
        self.flood_prone = np.zeros(n_edges, dtype=bool)
        self.vulnerability = np.ones(n_edges)
        if cfg.flooding is not None:
            pair_of_edge, first = network.undirected_pairs()
            river = network.edges["river_dist_m"].to_numpy(float)
            pair_river = np.full(len(first), np.inf)
            np.minimum.at(pair_river, pair_of_edge, river)
            self.flood_prone = pair_river[pair_of_edge] <= cfg.flooding.buffer_m
            rng = np.random.default_rng(cfg.flooding.vulnerability_seed)
            self.vulnerability = rng.random(len(first))[pair_of_edge]

        # Festival routes and their influence zone; match counts per configured street name.
        self.route_match_counts: dict[str, int] = {}
        self.route_edges = np.zeros(n_edges, dtype=bool)
        self.influence_edges = np.zeros(n_edges, dtype=bool)
        self.near_route_nodes = np.zeros(network.n_nodes, dtype=bool)
        if cfg.festival is not None:
            self._init_festival(cfg.festival)

    # -- time helpers -------------------------------------------------------------------------

    def local_hour(self, t: datetime) -> int:
        return t.astimezone(self.tz).hour

    def day_offset(self, t: datetime) -> int:
        """Whole local days since the scenario's start date (may be negative)."""
        local = t.astimezone(self.tz).replace(tzinfo=None)
        return (local.date() - self._local_start.replace(tzinfo=None).date()).days

    # -- rain and flooding -------------------------------------------------------------------

    def rain_mm_h(self, t: datetime) -> float:
        rain = self.cfg.rain
        if rain is None:
            return 0.0
        day = self.day_offset(t) % len(rain.day_factors)
        return rain.hourly_mm_h[self.local_hour(t)] * rain.day_factors[day]

    def rain_accum_mm(self, t: datetime, window_h: int) -> float:
        """Rain over the current and previous ``window_h - 1`` local hours (mm)."""
        return float(sum(self.rain_mm_h(t - k * HOUR) for k in range(window_h)))

    def flood_stage(self, t: datetime) -> int:
        """Index of the active flood stage, or -1 if none."""
        flooding = self.cfg.flooding
        if flooding is None:
            return -1
        accum = self.rain_accum_mm(t, flooding.window_h)
        stage = -1
        for i, s in enumerate(flooding.stages.value):
            if accum >= s.min_accum_mm:
                stage = i
        return stage

    def _flood_factors(self, stage: int) -> np.ndarray:
        factors = np.ones(self.network.n_edges)
        if stage < 0 or self.cfg.flooding is None:
            return factors
        s = self.cfg.flooding.stages.value[stage]
        u = self.vulnerability
        impassable = self.flood_prone & (u < s.impassable_fraction)
        slowed = self.flood_prone & ~impassable & (u < s.impassable_fraction + s.slowed_fraction)
        factors[slowed] = s.slow_multiplier
        factors[impassable] = 0.0
        return factors

    # -- festival -----------------------------------------------------------------------------

    def _init_festival(self, festival: FestivalConfig) -> None:
        net = self.network
        names = net.edges["name"].fillna("").str.lower()
        mids = net.edges[["mid_x", "mid_y"]].to_numpy(float)
        for route in festival.routes:
            in_area = np.ones(net.n_edges, dtype=bool)
            if route.center is not None and route.max_distance_m is not None:
                center = net.project(route.center[0], route.center[1])[0]
                in_area = np.linalg.norm(mids - center, axis=1) <= route.max_distance_m
            for street in route.street_names:
                matched = names.str.contains(street.lower(), regex=False).to_numpy() & in_area
                self.route_match_counts[f"{route.name}: {street}"] = int(matched.sum())
                if not matched.any():
                    log.warning("festival route %s: street %r matches no edge", route.name, street)
                self.route_edges |= matched
            for line in route.polylines:
                lat, lon = zip(*line, strict=True)
                pts = _densify([tuple(p) for p in net.project(lat, lon)], step=route.buffer_m / 2)
                dist, _ = cKDTree(pts).query(mids)
                self.route_edges |= dist <= route.buffer_m
        if not self.route_edges.any():
            return
        tree = cKDTree(mids[self.route_edges])
        radius = festival.influence_radius_m
        edge_dist, _ = tree.query(mids)
        self.influence_edges = (edge_dist <= radius) & ~self.route_edges
        node_dist, _ = tree.query(net.xy)
        self.near_route_nodes = node_dist <= radius

    def festival_effect(self, t: datetime) -> Literal["closed", "slowed"] | None:
        festival = self.cfg.festival
        if festival is None:
            return None
        day, hour = self.day_offset(t), self.local_hour(t)
        effects = {
            w.effect
            for w in festival.windows
            if day in w.days and w.start_hour <= hour < w.end_hour
        }
        if "closed" in effects:
            return "closed"
        return "slowed" if effects else None

    def _festival_factors(self, effect: str | None) -> np.ndarray:
        factors = np.ones(self.network.n_edges)
        festival = self.cfg.festival
        if effect is None or festival is None:
            return factors
        factors[self.influence_edges] = festival.influence_speed_multiplier.value
        slowed = festival.slowed_multiplier.value
        factors[self.route_edges] = 0.0 if effect == "closed" else slowed
        return factors

    def incident_node_multipliers(self, t: datetime) -> np.ndarray:
        """Per-node incident-rate multiplier from the festival (1 elsewhere / outside windows)."""
        mult = np.ones(self.network.n_nodes)
        if self.festival_effect(t) is not None and self.cfg.festival is not None:
            mult[self.near_route_nodes] = self.cfg.festival.incident_multiplier_near.value
        return mult

    def type_mix_near_route(self, t: datetime) -> dict[EventType, float] | None:
        festival = self.cfg.festival
        if self.festival_effect(t) is None or festival is None:
            return None
        return None if festival.type_mix_override is None else festival.type_mix_override.value

    # -- combined -----------------------------------------------------------------------------

    def state_key(self, t: datetime) -> tuple[object, ...]:
        """Everything (besides hour and profile) that changes edge times; used as a cache key."""
        rain_band = self.network.traffic.rain_multiplier(self.rain_mm_h(t))
        return (self.cfg.name, rain_band, self.flood_stage(t), self.festival_effect(t))

    def edge_factors(self, t: datetime) -> np.ndarray:
        """Scenario speed factor per edge (flood x festival); 0 means impassable."""
        key = (self.flood_stage(t), self.festival_effect(t))
        cached = self._factor_cache.get(key)
        if cached is None:
            cached = self._flood_factors(key[0]) * self._festival_factors(key[1])
            self._factor_cache[key] = cached
        return cached

    def impassable_edges(self, t: datetime) -> np.ndarray:
        return self.edge_factors(t) == 0.0
