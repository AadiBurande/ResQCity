"""Synthetic incident generator: non-homogeneous Poisson process, piecewise constant per node-hour.

Rate at in-city node n during hour h (local time):

    lambda_n(h) = Lambda/24 * s_n * hour[h] * weekday[d] * rain(h) * festival_n(h) * scenario_mult

``s_n`` is the node's hotspot weight normalised over in-city nodes (sum = 1), and ``hour`` and
``weekday`` are normalised to mean 1. Ward-hour rates are sums over a ward's nodes, so this is the
per-ward-hour NHPP of the design at node resolution. For each hour: N ~ Poisson(sum_n lambda_n),
nodes are drawn proportionally to lambda_n, onset times uniformly within the hour.

    uv run python -m sim.generator --scenario monsoon --replication 0
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from common.config import CONFIG_DIR, ConfigModel, SourcedValue, load_base_config, load_model
from common.provenance import write_meta
from common.requirements import (
    RequirementsConfig,
    load_requirements_config,
    required_services,
    severity_class,
    severity_score,
)
from common.rng import stream
from common.schemas import EventType, GeoPoint, SeverityFeatures
from geo.network import RoadNetwork
from geo.scenario import ScenarioState
from sim.groundtruth import GroundTruthIncident

HOUR = timedelta(hours=1)
# Sub-streams (common random numbers): arrivals per hour; marks (type, features) per incident.
# Changing a feature probability therefore never moves arrival times or locations.
ARRIVALS_STREAM = "incidents.arrivals"
MARKS_STREAM = "incidents.marks"


class RainEffect(ConfigModel):
    per_mm_h: SourcedValue[float]
    max_multiplier: SourcedValue[float]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_multiplier.value < 1:
            raise ValueError("rain_effect.max_multiplier must be >= 1")
        return self

    def multiplier(self, rain_mm_h: float) -> float:
        return min(self.max_multiplier.value, 1.0 + self.per_mm_h.value * rain_mm_h)


class HotspotConfig(ConfigModel):
    class_weight: SourcedValue[dict[str, float]]
    junction_bonus: SourcedValue[float]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.junction_bonus.value < 1 or min(self.class_weight.value.values()) < 0:
            raise ValueError("hotspot weights must be >= 0 and junction_bonus >= 1")
        return self


class AccidentFeatures(ConfigModel):
    extra_vehicles_mean: float = Field(ge=0)
    max_vehicles: int = Field(ge=1)
    p_person_down: float = Field(ge=0, le=1)
    p_overturned: float = Field(ge=0, le=1)
    p_overturned_heavy: float = Field(ge=0, le=1)
    p_fire_visible: float = Field(ge=0, le=1)


class FireFeatures(ConfigModel):
    p_vehicle_fire: float = Field(ge=0, le=1)
    p_person_down: float = Field(ge=0, le=1)


class SimpleFeatures(ConfigModel):
    p_person_down: float = Field(ge=0, le=1)


class FeatureModel(ConfigModel):
    heavy_vehicle_classes: list[str]
    vehicle_class_mix: SourcedValue[dict[str, float]]
    accident: SourcedValue[AccidentFeatures]
    fire_smoke: SourcedValue[FireFeatures]
    crowd: SourcedValue[SimpleFeatures]
    other: SourcedValue[SimpleFeatures]


def _normalised_mix(mix: dict[EventType, float]) -> dict[EventType, float]:
    total = sum(mix.values())
    if total <= 0 or any(v < 0 for v in mix.values()):
        raise ValueError("mix weights must be non-negative with a positive sum")
    return {k: v / total for k, v in mix.items()}


class IncidentsConfig(ConfigModel):
    synthetic: bool
    base_rate_per_day: SourcedValue[float]
    hour_profile: SourcedValue[list[float]]
    weekday_profile: SourcedValue[list[float]]
    hotspot: HotspotConfig
    rain_effect: RainEffect
    type_mix: SourcedValue[dict[EventType, float]]
    features: FeatureModel

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.synthetic:
            raise ValueError(
                "incidents.yaml describes a synthetic generator; synthetic must be true"
            )
        if len(self.hour_profile.value) != 24 or len(self.weekday_profile.value) != 7:
            raise ValueError("hour_profile needs 24 values and weekday_profile 7")
        if self.base_rate_per_day.value <= 0:
            raise ValueError("base_rate_per_day must be positive")
        _normalised_mix(self.type_mix.value)
        _normalised_mix(self.features.vehicle_class_mix.value)
        return self

    @property
    def hour_factors(self) -> np.ndarray:
        h = np.asarray(self.hour_profile.value, float)
        return h / h.mean()

    @property
    def weekday_factors(self) -> np.ndarray:
        d = np.asarray(self.weekday_profile.value, float)
        return d / d.mean()


def load_incidents_config(path: Path | None = None) -> IncidentsConfig:
    return load_model(IncidentsConfig, path or CONFIG_DIR / "incidents.yaml")


def hotspot_weights(network: RoadNetwork, cfg: HotspotConfig) -> np.ndarray:
    """Normalised hotspot weight per node (0 outside the city; sums to 1).

    Each road counts once, whether one-way or two-way (both directions share a pair id).
    """
    _, first = network.undirected_pairs()
    weights = cfg.class_weight.value
    class_w = network.edges["road_class"].iloc[first].map(weights).fillna(0.0).to_numpy(float)
    w = np.zeros(network.n_nodes)
    np.add.at(w, network.u_idx[first], class_w)
    np.add.at(w, network.v_idx[first], class_w)
    junction = network.distinct_degree() >= 3
    w = np.where(junction, w * cfg.junction_bonus.value, w)
    w = np.where(network.nodes["in_city"].to_numpy(bool), w, 0.0)
    if w.sum() <= 0:
        raise ValueError("no in-city node has positive hotspot weight")
    return w / w.sum()


class IncidentGenerator:
    def __init__(
        self,
        network: RoadNetwork,
        scenario: ScenarioState,
        cfg: IncidentsConfig,
        requirements: RequirementsConfig,
    ) -> None:
        self.network = network
        self.scenario = scenario
        self.cfg = cfg
        self.requirements = requirements
        self.node_share = hotspot_weights(network, cfg.hotspot)
        self._ward = network.nodes["ward_id"].to_numpy(str)
        self._lat = network.nodes["lat"].to_numpy(float)
        self._lon = network.nodes["lon"].to_numpy(float)

    # -- rates --------------------------------------------------------------------------------

    def hours(self) -> list[datetime]:
        start, n = self.scenario.cfg.start, self.scenario.cfg.days * 24
        return [start + k * HOUR for k in range(n)]

    def global_multiplier(self, t: datetime) -> float:
        s = self.scenario
        local = t.astimezone(s.tz)
        return float(
            self.cfg.base_rate_per_day.value
            / 24.0
            * self.cfg.hour_factors[local.hour]
            * self.cfg.weekday_factors[local.weekday()]
            * self.cfg.rain_effect.multiplier(s.rain_mm_h(t))
            * s.cfg.incident_rate_multiplier
        )

    def node_rates(self, t: datetime) -> np.ndarray:
        """Expected incidents per node during the hour starting at ``t``."""
        return (
            self.global_multiplier(t) * self.node_share * self.scenario.incident_node_multipliers(t)
        )

    def ward_hour_rates(self) -> pd.DataFrame:
        """Expected count per (ward, hour) over the scenario horizon: the NHPP intensity table."""
        rows = []
        for t in self.hours():
            rates = pd.Series(self.node_rates(t)).groupby(self._ward).sum()
            rows.append(
                pd.DataFrame(
                    {"hour_start": t, "ward_id": rates.index, "expected": rates.to_numpy()}
                )
            )
        return pd.concat(rows, ignore_index=True)

    # -- sampling -----------------------------------------------------------------------------

    def _sample_features(self, rng: np.random.Generator, event_type: EventType) -> SeverityFeatures:
        fm = self.cfg.features
        mix = fm.vehicle_class_mix.value
        classes = list(mix)
        p = np.asarray(list(mix.values()), float)
        p = p / p.sum()
        if event_type == EventType.ACCIDENT:
            a = fm.accident.value
            n = min(a.max_vehicles, 1 + int(rng.poisson(a.extra_vehicles_mean)))
            vehicles = [str(v) for v in rng.choice(classes, size=n, p=p)]
            heavy = any(v in fm.heavy_vehicle_classes for v in vehicles)
            return SeverityFeatures(
                vehicle_classes=vehicles,
                num_vehicles=n,
                person_down=bool(rng.random() < a.p_person_down),
                overturned=bool(rng.random() < (a.p_overturned_heavy if heavy else a.p_overturned)),
                fire_visible=bool(rng.random() < a.p_fire_visible),
            )
        if event_type == EventType.FIRE_SMOKE:
            f = fm.fire_smoke.value
            vehicles = [str(rng.choice(classes, p=p))] if rng.random() < f.p_vehicle_fire else []
            return SeverityFeatures(
                vehicle_classes=vehicles,
                num_vehicles=len(vehicles),
                fire_visible=True,
                person_down=bool(rng.random() < f.p_person_down),
            )
        simple = (fm.crowd if event_type == EventType.CROWD else fm.other).value
        return SeverityFeatures(person_down=bool(rng.random() < simple.p_person_down))

    def generate(self, seed_base: int, replication: int) -> list[GroundTruthIncident]:
        base_mix = _normalised_mix(self.cfg.type_mix.value)
        incidents: list[GroundTruthIncident] = []
        for h, t in enumerate(self.hours()):
            arrivals = stream(seed_base, replication, ARRIVALS_STREAM, h)
            rates = self.node_rates(t)
            total = rates.sum()
            count = int(arrivals.poisson(total))
            if count == 0:
                continue
            nodes = arrivals.choice(self.network.n_nodes, size=count, p=rates / total)
            offsets = np.sort(arrivals.random(count)) * 3600.0
            near_mix = self.scenario.type_mix_near_route(t)
            for k, (node, offset) in enumerate(zip(nodes, offsets, strict=True)):
                marks = stream(seed_base, replication, MARKS_STREAM, h, k)
                near = bool(self.scenario.near_route_nodes[node])
                mix = _normalised_mix(near_mix) if (near and near_mix) else base_mix
                types = list(mix)
                event_type = types[int(marks.choice(len(types), p=list(mix.values())))]
                features = self._sample_features(marks, event_type)
                score = severity_score(event_type, features, self.requirements.severity)
                incidents.append(
                    GroundTruthIncident(
                        id=f"GT-{self.scenario.cfg.name[:3].upper()}-{replication:03d}-{len(incidents):05d}",
                        onset_ts=t + timedelta(seconds=float(offset)),
                        type=event_type,
                        node=int(self.network.node_ids[node]),
                        location=GeoPoint(lat=float(self._lat[node]), lon=float(self._lon[node])),
                        ward_id=str(self._ward[node]),
                        severity_features=features,
                        severity_score=score,
                        severity_class=severity_class(score, self.requirements.severity),
                        requirements=required_services(event_type, features, self.requirements),
                        near_festival_route=near,
                        scenario=self.scenario.cfg.name,
                        replication=replication,
                    )
                )
        return incidents


def to_frame(incidents: list[GroundTruthIncident]) -> pd.DataFrame:
    rows = []
    for inc in incidents:
        f, r = inc.severity_features, inc.requirements
        rows.append(
            {
                "id": inc.id,
                "onset_ts": inc.onset_ts,
                "type": inc.type.value,
                "node": inc.node,
                "lat": inc.location.lat,
                "lon": inc.location.lon,
                "ward_id": inc.ward_id,
                "vehicle_classes": ",".join(f.vehicle_classes),
                "num_vehicles": f.num_vehicles,
                "overturned": f.overturned,
                "person_down": f.person_down,
                "fire_visible": f.fire_visible,
                "severity_score": inc.severity_score,
                "severity_class": inc.severity_class,
                "req_ambulance": r.ambulance,
                "req_police": r.police,
                "req_fire": r.fire,
                "near_festival_route": inc.near_festival_route,
                "scenario": inc.scenario,
                "replication": inc.replication,
                "synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    from geo.config import load_geo_config
    from geo.scenario import SCENARIO_DIR, load_scenario

    parser = argparse.ArgumentParser(description="Generate synthetic ground-truth incidents")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument(
        "--seed-set",
        choices=["evaluation", "tuning", "forecast_training"],
        default="evaluation",
        help="which disjoint seed base to use (configs/base.yaml: random)",
    )
    args = parser.parse_args(argv)

    base, geo_cfg = load_base_config(), load_geo_config()
    seed_base = {
        "evaluation": base.random.seed_base,
        "tuning": base.random.tuning_seed_base,
        "forecast_training": base.random.forecast_training_seed_base,
    }[args.seed_set]
    processed = base.paths.resolve("data_processed")
    network = RoadNetwork.load(processed, geo_cfg)
    scenario_path = SCENARIO_DIR / f"{args.scenario}.yaml"
    state = ScenarioState(load_scenario(scenario_path), network, base.project.timezone)
    gen = IncidentGenerator(network, state, load_incidents_config(), load_requirements_config())
    incidents = gen.generate(seed_base, args.replication)
    name = f"incidents_synthetic_{args.scenario}_{args.seed_set}_r{args.replication:03d}.parquet"
    out = processed / "synthetic" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    to_frame(incidents).to_parquet(out, index=False)
    write_meta(
        out,
        configs=[
            CONFIG_DIR / "base.yaml",
            CONFIG_DIR / "geo.yaml",
            CONFIG_DIR / "incidents.yaml",
            CONFIG_DIR / "requirements.yaml",
            scenario_path,
        ],
        synthetic=True,
        seeds={"seed_base": seed_base, "replication": args.replication},
        extra={"seed_set": args.seed_set, "incidents": len(incidents)},
    )
    print(f"{len(incidents)} synthetic incidents -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
