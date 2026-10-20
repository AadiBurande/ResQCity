"""Calibrate time-of-day speed multipliers to Google Maps "typical" travel times (D-026).

    uv run python -m resqcity.geo.calibrate_speeds

Data: ``results/geo/maps_check_google_2026-10-20.csv``: model routes with Google Maps "typically
X-Y min" ranges for a Tuesday departure at 10:00, 14:00 and 19:00, entered by the user.
Routes are split in advance (``configs/maps_check_routes.yaml``): ``pairs`` (4 routes) are used
to FIT, ``holdout_pairs`` (6 routes) only to EVALUATE.

Method: for each departure hour h, f_h = exp(mean log(google_mid / model_minutes)), with the
model under the BASELINE speeds (``configs/archive/geo_speeds_d014.yaml``). Step 1 (pre-specified)
fits on the 4 fit routes and evaluates on the 6 hold-out routes; it over-corrected. Step 2 (added
afterwards, disclosed in D-026) estimates the error of an all-route fit by leave-one-route-out
(LORO). Step 3 (added after an independent audit found the pooled step-2 gain came from 14:00
only, disclosed in D-026): a period is changed only if LORO improves BOTH the median error and
the share inside Google's range AT ITS HOUR; a route-level bootstrap gives intervals for the gains.
Scaling every edge speed in a period by 1/f_h scales every route time by f_h (routes do not change),
so: am_peak (09-11) and pm_peak (17-21) multipliers are divided by f_10 and f_19; a new midday
period (11-17) gets multiplier 1/f_14. Night and shoulder hours are not observed and stay as they
were. Output: ``results/geo/speed_calibration.json`` (factors, per-hour LORO errors, adopted
periods, calibrated multipliers) and ``speed_calibration_routes.csv``. ``configs/geo.yaml`` must
carry these multipliers (tested).
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from resqcity.common.config import CONFIG_DIR, load_base_config, load_model
from resqcity.common.provenance import write_meta
from resqcity.geo.config import GeoConfig
from resqcity.geo.network import RoadNetwork
from resqcity.geo.sanity import SanityConfig, sanity_table

GOOGLE_FILE = Path("geo") / "maps_check_google_2026-10-20.csv"  # under results/
ROUTES = CONFIG_DIR / "maps_check_routes.yaml"
BASELINE = CONFIG_DIR / "archive" / "geo_speeds_d014.yaml"
PERIOD_OF_HOUR = {10: "am_peak", 14: "midday", 19: "pm_peak"}
MIDDAY = (11, 17)
BOOTSTRAP_REPS = 2000  # route-level bootstrap of the LORO error reduction
LORO_NOTE = (
    "mean_log_ratio of LORO predictions is zero by construction (per-hour geometric-mean fit); "
    "it is not evidence of reduced bias"
)
RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")


def parse_range(text: object) -> tuple[float, float]:
    m = RANGE.search(str(text))
    if not m:
        raise ValueError(f"cannot parse a 'X-Y min' range from {text!r}")
    return float(m.group(1)), float(m.group(2))


def load_google(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"google_typical_min_low": str, "google_typical_min_high": str})
    ranges = [
        parse_range(lo if isinstance(lo, str) and RANGE.search(lo) else f"{lo}-{hi}")
        for lo, hi in zip(df["google_typical_min_low"], df["google_typical_min_high"], strict=True)
    ]
    df["g_low"], df["g_high"] = zip(*ranges, strict=True)
    df["g_mid"] = (df["g_low"] + df["g_high"]) / 2
    df["hour"] = df["depart_local"].str[-5:-3].astype(int)
    df["holdout"] = df["holdout"].astype(str).str.upper().eq("TRUE")
    return df[["from", "to", "holdout", "hour", "g_low", "g_high", "g_mid"]]


def model_minutes(cfg: GeoConfig, sanity: SanityConfig, processed: Path, tz: str) -> pd.DataFrame:
    table = sanity_table(RoadNetwork.load(processed, cfg), sanity, tz)
    table["hour"] = table["depart_local"].str[-5:-3].astype(int)
    return table[["from", "to", "hour", "minutes"]]


def calibrated_config(
    baseline: GeoConfig, factors: dict[int, float], hours: set[int] | None = None
) -> GeoConfig:
    """Baseline with the periods of ``hours`` (default: all observed hours) rescaled."""
    hours = set(PERIOD_OF_HOUR) if hours is None else hours
    data = baseline.model_dump(mode="json")
    source = "calibrated to Google Maps typical times (results/geo/speed_calibration.json, D-026)"
    periods = []
    for p in data["traffic_periods"]:
        hour = next((h for h, name in PERIOD_OF_HOUR.items() if name == p["name"]), None)
        if hour in hours:
            for v in p["multipliers"].values():
                v["value"] = round(v["value"] / factors[hour], 4)
                v["source"] = source
        periods.append(p)
    if 14 in hours:
        classes = list(data["base_speeds_kph"])
        periods.append(
            {
                "name": "midday",
                "start_hour": MIDDAY[0],
                "end_hour": MIDDAY[1],
                "multipliers": {
                    c: {"value": round(1 / factors[14], 4), "source": source} for c in classes
                },
            }
        )
    data["traffic_periods"] = periods
    return GeoConfig.model_validate(data)


def hour_factors(df: pd.DataFrame) -> dict[int, float]:
    """Time multiplier per departure hour: geometric mean of google_mid / model_baseline."""
    return {
        int(h): float(math.exp(np.log(g["g_mid"] / g["model_baseline"]).mean()))
        for h, g in df.groupby("hour")
    }


def loro_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``model_loro``: each route predicted with per-hour factors fitted on the OTHER routes.
    Exact because a uniform per-period speed scale rescales every route time (routes unchanged)."""
    routes = df[["from", "to"]].drop_duplicates().itertuples(index=False)
    parts = []
    for r in routes:
        held = (df["from"] == r[0]) & (df["to"] == r[1])
        f = hour_factors(df[~held])
        part = df[held].copy()
        part["model_loro"] = part["model_baseline"] * part["hour"].map(f)
        parts.append(part)
    return pd.concat(parts)


def leave_one_route_out(pooled: pd.DataFrame) -> dict[str, object]:
    """Baseline vs LORO error, pooled, per departure hour and per route group (fit / hold-out)."""

    def both(d: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {"baseline": errors(d, "model_baseline"), "calibrated": errors(d, "model_loro")}

    return {
        **both(pooled),
        "by_hour": {str(h): both(g) for h, g in pooled.groupby("hour")},
        "by_route_group": {
            ("holdout6" if h else "fit4"): both(g) for h, g in pooled.groupby("holdout")
        },
        "note": LORO_NOTE,
    }


def adopted_hours(loro: dict[str, object]) -> set[int]:
    """Step 3: an hour's period changes only if LORO improves both error measures at that hour."""
    out = set()
    for h, e in loro["by_hour"].items():  # type: ignore[union-attr]
        base, cal = e["baseline"], e["calibrated"]
        if (
            cal["median_abs_pct_error_vs_mid"] < base["median_abs_pct_error_vs_mid"]
            and cal["share_within_google_range"] > base["share_within_google_range"]
        ):
            out.add(int(h))
    return out


def bootstrap_gain(pooled: pd.DataFrame, seed: int) -> dict[str, dict[str, float]]:
    """Reduction in median absolute % error (baseline minus LORO), pooled and per hour, with a
    95 % route-level bootstrap interval (routes resampled; factors not refitted)."""
    pooled = pooled.assign(
        route=pooled["from"] + " -> " + pooled["to"],
        e_base=(pooled["model_baseline"] / pooled["g_mid"] - 1).abs() * 100,
        e_loro=(pooled["model_loro"] / pooled["g_mid"] - 1).abs() * 100,
    )
    routes = pooled["route"].unique()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(routes), size=(BOOTSTRAP_REPS, len(routes)))
    out = {}
    for key, sub in [("all", pooled), *((str(h), g) for h, g in pooled.groupby("hour"))]:
        by_route = {r: g for r, g in sub.groupby("route")}

        def gain(rs: np.ndarray, by_route: dict = by_route) -> float:
            d = pd.concat([by_route[routes[i]] for i in rs])
            return float(d["e_base"].median() - d["e_loro"].median())

        boots = np.array([gain(rs) for rs in draws])
        out[key] = {
            "gain_pct_points": gain(np.arange(len(routes))),
            "ci95_lo": float(np.percentile(boots, 2.5)),
            "ci95_hi": float(np.percentile(boots, 97.5)),
        }
    return out


def baseline_direction(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per hour: routes on which the BASELINE is faster than Google's midpoint and below its range,
    with a two-sided sign test. Nothing is fitted, so this needs no hold-out."""
    out = {}
    for h, g in df.groupby("hour"):
        faster = int((g["model_baseline"] < g["g_mid"]).sum())
        out[str(h)] = {
            "routes": len(g),
            "baseline_faster_than_google_mid": faster,
            "baseline_below_google_low": int((g["model_baseline"] < g["g_low"]).sum()),
            "sign_test_p": float(binomtest(faster, len(g)).pvalue),
        }
    return out


def errors(df: pd.DataFrame, col: str) -> dict[str, float]:
    ratio = df[col] / df["g_mid"]
    inside = (df[col] >= df["g_low"]) & (df[col] <= df["g_high"])
    return {
        "n": len(df),
        "share_within_google_range": float(inside.mean()),
        "median_abs_pct_error_vs_mid": float((ratio - 1).abs().median() * 100),
        "mean_log_ratio": float(np.log(ratio).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base = load_base_config()
    processed = base.paths.resolve("data_processed")
    sanity = load_model(SanityConfig, ROUTES)
    google = load_google(base.paths.resolve("results") / GOOGLE_FILE)
    baseline = load_model(GeoConfig, CONFIG_DIR / "geo.yaml", BASELINE)
    tz = base.project.timezone

    df = google.merge(model_minutes(baseline, sanity, processed, tz), on=["from", "to", "hour"])
    df = df.rename(columns={"minutes": "model_baseline"})
    # Step 1 (pre-specified): fit on the 4 fit routes, evaluate on the 6 hold-out routes.
    fit = df[~df["holdout"]]
    factors_fit4 = hour_factors(fit)
    cal4 = model_minutes(calibrated_config(baseline, factors_fit4), sanity, processed, tz)
    df = df.merge(cal4.rename(columns={"minutes": "model_fit4"}), on=["from", "to", "hour"])
    hold = df[df["holdout"]]
    # Step 2 (added after step 1 over-corrected, D-026): leave-one-route-out over all 10 routes.
    pooled = loro_predictions(df)
    loro = leave_one_route_out(pooled)
    # Step 3 (added after an independent audit, D-026): adopt per period, judged at its own hour.
    factors_all = hour_factors(df)
    adopt = adopted_hours(loro)
    final = calibrated_config(baseline, factors_all, adopt)
    cal = model_minutes(final, sanity, processed, tz)
    df = df.merge(cal.rename(columns={"minutes": "model_calibrated"}), on=["from", "to", "hour"])
    report = {
        "data": GOOGLE_FILE.as_posix(),
        "collected": "2026-10-20, Google Maps 'typically' ranges, Tuesday departures (user)",
        "step1_prespecified_fit4_holdout6": {
            "factors_by_hour": {str(h): f for h, f in factors_fit4.items()},
            "holdout_before": errors(hold, "model_baseline"),
            "holdout_after": errors(hold, "model_fit4"),
            "verdict": "over-corrects: the 4 fit routes are the most congested (central city)",
        },
        "baseline_direction_by_hour": baseline_direction(df),
        "step2_leave_one_route_out_all10": loro,
        "step2_bootstrap_gain": bootstrap_gain(pooled, base.random.seed_base),
        "step3_rule": "change a period only if LORO improves both the median error and the share "
        "inside Google's range at its hour (added after an independent audit, D-026)",
        "adopted_periods": [PERIOD_OF_HOUR[h] for h in sorted(adopt)],
        "kept_at_baseline": [PERIOD_OF_HOUR[h] for h in sorted(set(PERIOD_OF_HOUR) - adopt)],
        "all_route_factors_by_hour": {str(h): f for h, f in factors_all.items()},
        "final_all_routes": errors(df, "model_calibrated"),
        "calibrated_multipliers": {
            p.name: {c: v.value for c, v in p.multipliers.items()} for p in final.traffic_periods
        },
        "uncalibrated_hours": "06-09, 21-23 (base speed; probably too fast, L2), night 23-06 "
        "and every period kept at baseline (D-014 multipliers)",
    }
    out = base.paths.resolve("results") / "geo" / "speed_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    routes = out.with_name("speed_calibration_routes.csv")
    df.to_csv(routes, index=False, lineterminator="\n")
    for f in (out, routes):
        write_meta(f, configs=[CONFIG_DIR / "geo.yaml", BASELINE, ROUTES], synthetic=False)
    print(json.dumps({k: v for k, v in report.items() if k != "calibrated_multipliers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
