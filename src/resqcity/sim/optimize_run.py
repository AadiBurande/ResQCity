"""Run the phase-6 OR models on the real Pune network and save their results.

    uv run python -m resqcity.sim.optimize_run

Three things this produces, all SYNTHETIC in their inputs and labelled as such:
- ``results/optimization/fleet_roster.json`` — where the 91 units stand (press-derived counts).
- ``results/optimization/camera_placement.json`` — the MCLP marginal-gain curve for K = 1..k_max
  over candidate junctions, against the 40 sites already chosen in phase 2.
- ``results/optimization/erlang.json`` — M/M/c utilisation and waits per agency at the measured
  incident rate, the analytical cross-check phase 7 validates the simulator against.

It lives in ``sim`` because it reads the incident generator for demand weights, and
``tests/test_architecture.py`` forbids ``optimization -> sim``.
"""

import argparse
import json
import logging

import numpy as np

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.schemas import Agency
from resqcity.geo.cameras import load_cameras
from resqcity.geo.config import load_geo_config
from resqcity.geo.facilities import load_facilities
from resqcity.geo.network import RoadNetwork
from resqcity.optimization.config import (
    FleetConfig,
    OptimizationConfig,
    load_fleet_config,
    load_optimization_config,
)
from resqcity.optimization.fleet import build_fleet, roster_summary
from resqcity.optimization.placement import (
    covered_weight,
    greedy_placement,
    queue_metrics,
    servers_for_target,
)
from resqcity.sim.generator import hotspot_weights, load_incidents_config

OUT = PROJECT_ROOT / "results" / "optimization"
CANDIDATE_SITES = 400  # busiest junctions considered as camera sites
DEMAND_POINTS = 3000  # highest-risk nodes used as the demand set for MCLP
log = logging.getLogger(__name__)


def ward_demand(network: RoadNetwork, node_share: np.ndarray) -> dict[str, float]:
    wards = network.nodes["ward_id"].astype(str).to_numpy()
    out: dict[str, float] = {}
    for ward, share in zip(wards, node_share, strict=True):
        if share > 0:
            out[ward] = out.get(ward, 0.0) + float(share)
    return out


def camera_placement(
    network: RoadNetwork, node_share: np.ndarray, radius_m: float, k_max: int
) -> dict[str, object]:
    """MCLP over the busiest junctions, with the phase-2 camera sites as the comparison."""
    order = np.argsort(node_share)[::-1]
    demand_idx = order[:DEMAND_POINTS]
    weights = node_share[demand_idx]
    degree = network.distinct_degree()
    junctions = [int(i) for i in order if degree[i] >= 3][:CANDIDATE_SITES]
    node_ids = [int(network.node_ids[i]) for i in junctions]
    dist = network.shortest_lengths(node_ids, limit_m=radius_m)
    covers = np.zeros((len(demand_idx), len(junctions)), dtype=bool)
    for s, row in enumerate(dist):
        covers[:, s] = np.isfinite(row[demand_idx])
    chosen, gains = greedy_placement(weights, covers, k_max)
    total = float(weights.sum())
    curve = []
    running = 0.0
    for k, (site, gain) in enumerate(zip(chosen, gains, strict=True), start=1):
        running += gain
        curve.append({
            "k": k, "node": int(network.node_ids[junctions[site]]),
            "marginal_gain": float(gain), "covered_share": running / total,
        })  # fmt: skip
    existing = load_cameras(CONFIG_DIR / "cameras.yaml")
    existing_idx = [
        j for j, node in enumerate(node_ids) if node in {c.junction_node for c in existing}
    ]
    return {
        "status": "SYNTHETIC demand (the incident generator's hotspot weights). Coverage is an "
        "UPPER BOUND on value: a covered incident is detected with probability 0.31 and every "
        "camera adds ~29 false alarms per hour (D-029).",
        "coverage_radius_m": radius_m,
        "candidate_sites": len(junctions),
        "demand_points": len(demand_idx),
        "existing_sites_in_candidate_set": len(existing_idx),
        "existing_40_covered_share": (
            covered_weight(existing_idx, weights, covers) / total if existing_idx else None
        ),
        "curve": curve,
    }


def erlang_table(
    incidents_per_day: float, cfg: OptimizationConfig, fleet: FleetConfig
) -> list[dict]:
    rows = []
    mix = {Agency.AMBULANCE: 0.45, Agency.POLICE: 1.0, Agency.FIRE: 0.30}
    for agency, share in sorted(mix.items(), key=lambda kv: kv[0].value):
        rate = incidents_per_day * share / 86400.0
        service = cfg.erlang.service_time_s.value[agency]
        have = fleet.counts[agency].value
        metrics = queue_metrics(rate, service, have)
        try:
            need = servers_for_target(rate, service, cfg.erlang.target_wait_s.value)
        except RuntimeError:
            need = None
        rows.append({
            "agency": agency.value, "share_of_incidents": share, "fleet": have,
            "service_time_s": service, **metrics,
            "units_for_target_wait": need, "target_wait_s": cfg.erlang.target_wait_s.value,
        })  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    incidents_cfg = load_incidents_config()
    opt, fleet_cfg = load_optimization_config(), load_fleet_config()
    facilities = load_facilities(base.paths.resolve("data_processed") / "facilities.json")
    node_share = hotspot_weights(network, incidents_cfg.hotspot)
    demand = ward_demand(network, node_share)

    units = build_fleet(facilities.stations, facilities.hospitals, demand, fleet_cfg)
    roster = {
        "status": "SYNTHETIC roster on real OSM stations; counts are press-derived and "
        "UNVERIFIED (D-031)",
        **roster_summary(units),
        "counts_configured": {a.value: c.value for a, c in fleet_cfg.counts.items()},
        "units": [
            {"id": u.id, "agency": u.agency.value, "station": u.home_station, "node": u.node}
            for u in units
        ],
    }
    log.info("roster: %s", roster_summary(units))

    placement = camera_placement(
        network, node_share, opt.placement.coverage_radius_m.value, opt.placement.k_max
    )
    log.info(
        "camera placement: the 40 existing sites cover %.2f%% of risk; the best %d new "
        "sites cover %.2f%%",
        100 * (placement["existing_40_covered_share"] or 0.0),
        opt.placement.k_max,
        100 * placement["curve"][-1]["covered_share"],
    )

    erlang = {
        "status": "M/M/c with assumed service times (UNVERIFIED); the analytical cross-check for "
        "phase 7, not a result",
        "incidents_per_day": incidents_cfg.base_rate_per_day.value,
        "rows": erlang_table(incidents_cfg.base_rate_per_day.value, opt, fleet_cfg),
    }
    for row in erlang["rows"]:
        log.info(
            "erlang %-9s fleet %2d, utilisation %.4f, P(wait) %.4f, wait %.1f s",
            row["agency"], row["fleet"], row["utilisation"], row["p_wait"], row["expected_wait_s"],
        )  # fmt: skip

    OUT.mkdir(parents=True, exist_ok=True)
    configs = [CONFIG_DIR / "optimization.yaml", CONFIG_DIR / "fleet.yaml",
               CONFIG_DIR / "incidents.yaml"]  # fmt: skip
    for name, payload in (
        ("fleet_roster.json", roster),
        ("camera_placement.json", placement),
        ("erlang.json", erlang),
    ):
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        write_meta(path, configs=configs, synthetic=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
