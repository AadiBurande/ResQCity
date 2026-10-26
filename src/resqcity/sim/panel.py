"""Ward-hour-type panel of synthetic incident demand, for the phase-5 forecaster (D-030).

    uv run python -m resqcity.sim.panel

Generates the history scenario, counts incidents into (ward, hour, target type) cells, and adds
the columns each model is allowed to see. It lives in ``sim`` because it reads ground truth;
``forecasting`` may not import ``sim`` (``tests/test_architecture.py``).

Columns
-------
``count``       observed incidents in the cell -- the target.
``intensity``   the generator's OWN expected count for the cell, computed analytically rather
                than sampled. This is the oracle: the best any forecaster could do. It is never a
                model feature, only a reference and the ``OracleForecast`` source.
``lag_*``       counts in the SAME ward and type over earlier hours, and citywide over 24 h; every
                lag ends at the hour before the one being predicted, so nothing leaks.
``rain_mm_h``   the true rain -- an "informed" feature only.
``rain_fc``     a short-horizon rain forecast with error -- the "realistic" feature.
``hotspot_mass`` the ward's share of the generator's incident location mass -- "informed" only;
                the realistic model must infer a ward's base rate from its own past counts.
``warmup``      True while the longest lag is not yet fully observed; excluded from fit and score.

Everything here is SYNTHETIC and seeded from ``random.forecast_training_seed_base``, which is
disjoint from the evaluation and tuning seeds (``configs/base.yaml``).
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import EventType
from resqcity.forecasting.config import ForecastingConfig, load_forecasting_config
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.geo.scenario import ScenarioState, load_scenario
from resqcity.sim.generator import (
    IncidentGenerator,
    IncidentsConfig,
    _normalised_mix,
    hotspot_weights,
    load_incidents_config,
)

HOUR = timedelta(hours=1)
log = logging.getLogger(__name__)


def ward_masses(
    network: RoadNetwork, state: ScenarioState, node_share: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(wards, total incident mass per ward, mass on festival-route nodes per ward)."""
    ward_of = network.nodes["ward_id"].astype(str).to_numpy()
    wards = sorted({w for w, s in zip(ward_of, node_share, strict=True) if s > 0})
    index = {w: i for i, w in enumerate(wards)}
    code = np.array([index.get(w, -1) for w in ward_of])
    keep = code >= 0
    base = np.bincount(code[keep], weights=node_share[keep], minlength=len(wards))
    near = state.near_route_nodes & keep
    on_route = np.bincount(code[near], weights=node_share[near], minlength=len(wards))
    return wards, base, on_route


def true_intensity(
    hours: list[datetime],
    state: ScenarioState,
    cfg: IncidentsConfig,
    fc: ForecastingConfig,
    base_mass: np.ndarray,
    route_mass: np.ndarray,
) -> np.ndarray:
    """Expected incidents per (ward, hour, target type), from the generator's own parameters.

    Exact rather than sampled: the generator draws a Poisson count with mean ``node_rates`` and
    then a type per incident, so the expected count of a type in a ward is the ward's share of
    the rate times that type's share of the mix -- with the festival multiplier and the festival
    type-mix override applied to the route nodes only.
    """
    targets = fc.targets
    names = targets.types
    out = np.zeros((len(base_mass), len(hours), len(names)), dtype=np.float64)
    off_route = base_mass - route_mass
    for h, t in enumerate(hours):
        mult = cfg.base_rate_per_day.value / 24.0
        local = t.astimezone(state.tz)
        weights = cfg.type_hour_weights(local.hour)
        mult *= sum(weights.values())
        mult *= cfg.weekday_factors[local.weekday()]
        mult *= cfg.rain_effect.multiplier(state.rain_mm_h(t))
        mult *= state.cfg.incident_rate_multiplier
        base_mix = _normalised_mix(weights)
        near_mix = state.type_mix_near_route(t)
        route_mix = _normalised_mix(near_mix) if near_mix else base_mix
        festival = float(
            state.incident_node_multipliers(t)[state.near_route_nodes].max(initial=1.0)
        )
        for j, name in enumerate(names):
            share_base = sum(v for k, v in base_mix.items() if targets.of(EventType(k)) == name)
            share_route = sum(v for k, v in route_mix.items() if targets.of(EventType(k)) == name)
            out[:, h, j] = mult * (off_route * share_base + route_mass * festival * share_route)
    return out


def observed_counts(
    incidents: list, hours: list[datetime], wards: list[str], fc: ForecastingConfig
) -> np.ndarray:
    ward_index = {w: i for i, w in enumerate(wards)}
    type_index = {name: j for j, name in enumerate(fc.targets.types)}
    start = hours[0]
    counts = np.zeros((len(wards), len(hours), len(fc.targets.types)), dtype=np.int32)
    for inc in incidents:
        w = ward_index.get(inc.ward_id)
        h = int((inc.onset_ts - start) // HOUR)
        if w is None or not 0 <= h < len(hours):
            continue
        counts[w, h, type_index[fc.targets.of(inc.type)]] += 1
    return counts


def lagged(counts: np.ndarray, lags: list[int], citywide_h: int) -> dict[str, np.ndarray]:
    """Sums of earlier hours, per (ward, type) series and citywide. Shape (wards, hours, types).

    ``lag_1`` is the previous hour; ``lag_k`` for k > 1 is the sum of the previous k hours. Every
    window ends at hour h-1, so the current hour never enters its own feature.
    """
    n_hours = counts.shape[1]
    # prefix[h] = total over hours strictly before h, so a window [h-k, h) is prefix[h]-prefix[h-k]
    prefix = np.concatenate(
        [np.zeros_like(counts[:, :1, :], dtype=np.float64), np.cumsum(counts, axis=1)], axis=1
    )
    out: dict[str, np.ndarray] = {}
    for k in lags:
        values = np.full(counts.shape, np.nan)
        values[:, k:, :] = prefix[:, k:n_hours, :] - prefix[:, : n_hours - k, :]
        out[f"lag_{k}h"] = values
    city_prefix = np.concatenate([[0.0], np.cumsum(counts.sum(axis=(0, 2)))])
    city = np.full(n_hours, np.nan)
    city[citywide_h:] = city_prefix[citywide_h:n_hours] - city_prefix[: n_hours - citywide_h]
    out[f"lag_city_{citywide_h}h"] = np.broadcast_to(city[None, :, None], counts.shape).copy()
    return out


def rain_columns(
    hours: list[datetime], state: ScenarioState, fc: ForecastingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """True hourly rain, and a short-horizon forecast of it.

    The forecast keeps the intra-day shape but gets the DAY wrong with probability ``p_mixup``
    (swapped for another day of the same month) and the amount wrong by lognormal error, so it
    can be wrong about whether it rains at all. A purely multiplicative error could not: it would
    leave every dry hour exactly zero and so reveal the truth.
    """
    rng = np.random.default_rng(fc.rain_forecast.seed)
    true = np.array([state.rain_mm_h(t) for t in hours])
    day_of = np.array([(t - hours[0]).days for t in hours])
    months = np.array([t.astimezone(state.tz).month for t in hours])
    n_days = int(day_of.max()) + 1
    day_month = np.array([months[day_of == d][0] for d in range(n_days)])
    day_true = np.array([true[day_of == d].sum() for d in range(n_days)])

    # which day the forecast actually describes (sometimes the wrong one, same month)
    described = np.arange(n_days)
    for d in np.flatnonzero(rng.random(n_days) < fc.rain_forecast.p_mixup.value):
        described[d] = int(rng.choice(np.flatnonzero(day_month == day_month[d])))
    error = np.exp(fc.rain_forecast.sigma.value * rng.standard_normal(n_days))
    forecast_total = day_true[described] * error

    # intra-day shape: the true shape on a wet day, a flat one on a dry day (which has no shape)
    wet = day_true[day_of] > 0
    shape = np.where(wet, true / np.where(wet, day_true[day_of], 1.0), 1.0 / 24.0)
    return true, np.maximum(shape * forecast_total[day_of], 0.0)


def build(fc: ForecastingConfig, seed_base: int) -> pd.DataFrame:
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    incidents_cfg, req = load_incidents_config(), load_requirements_config()
    scenario = load_scenario(fc.history_scenario)
    state = ScenarioState(scenario, network)
    gen = IncidentGenerator(network, state, incidents_cfg, req)
    node_share = hotspot_weights(network, incidents_cfg.hotspot)
    wards, base_mass, route_mass = ward_masses(network, state, node_share)
    hours = gen.hours()
    log.info(
        "generating %d days of history (%d wards, %d hours)", scenario.days, len(wards), len(hours)
    )
    incidents = gen.generate(seed_base, 0)
    counts = observed_counts(incidents, hours, wards, fc)
    intensity = true_intensity(hours, state, incidents_cfg, fc, base_mass, route_mass)
    lags = lagged(counts, fc.features.ward_lags_h, fc.features.citywide_lag_h)
    rain_true, rain_fc = rain_columns(hours, state, fc)
    festival = np.array([state.festival_effect(t) is not None for t in hours])
    local = [t.astimezone(state.tz) for t in hours]
    n_w, n_h, n_t = counts.shape
    frame = pd.DataFrame({
        "ward": pd.Categorical(np.repeat(wards, n_h * n_t)),
        "ts": pd.DatetimeIndex(hours).tz_convert("UTC").repeat(n_t).tolist() * n_w,
        "type": pd.Categorical(np.tile(fc.targets.types, n_w * n_h)),
        "count": counts.reshape(-1).astype(np.int16),
        "intensity": intensity.reshape(-1).astype(np.float32),
        "hour": np.tile(np.repeat([t.hour for t in local], n_t), n_w).astype(np.int8),
        "weekday": np.tile(np.repeat([t.weekday() for t in local], n_t), n_w).astype(np.int8),
        "month": np.tile(np.repeat([t.month for t in local], n_t), n_w).astype(np.int8),
        "festival": np.tile(np.repeat(festival, n_t), n_w),
        "rain_mm_h": np.tile(np.repeat(rain_true, n_t), n_w).astype(np.float32),
        "rain_fc": np.tile(np.repeat(rain_fc, n_t), n_w).astype(np.float32),
        "hotspot_mass": np.repeat(base_mass, n_h * n_t).astype(np.float32),
    })  # fmt: skip
    for name, values in lags.items():
        frame[name] = values.reshape(-1).astype(np.float32)
    frame["warmup"] = np.tile(np.repeat(np.arange(n_h) < fc.features.max_lag_h, n_t), n_w)
    return frame


def panel_path(fc: ForecastingConfig) -> Path:
    base = load_base_config()
    out = base.paths.resolve("data_processed") / "synthetic"
    return out / f"panel_{fc.history_scenario}_synthetic.parquet"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fc = load_forecasting_config()
    seed_base = load_base_config().random.forecast_training_seed_base
    frame = build(fc, seed_base)
    out = panel_path(fc)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    write_meta(
        out,
        configs=[
            CONFIG_DIR / "forecasting.yaml",
            CONFIG_DIR / "incidents.yaml",
            CONFIG_DIR / "scenarios" / f"{fc.history_scenario}.yaml",
        ],
        synthetic=True,
        seeds={"forecast_training_seed_base": seed_base},
    )
    live = frame[~frame["warmup"]]
    log.info(
        "%s: %d rows, %d incidents, mean count %.5f, intensity sum %.1f (observed %d)",
        out.name,
        len(frame),
        int(frame["count"].sum()),
        float(live["count"].mean()),
        float(live["intensity"].sum()),
        int(live["count"].sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
