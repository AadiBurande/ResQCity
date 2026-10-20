"""Travel times between known Pune places, for a human sanity check (not a validation dataset).

uv run python -m resqcity.geo.sanity      ->  results/geo/sanity_travel_times.csv
"""

import argparse
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field

from resqcity.common.config import CONFIG_DIR, ConfigModel, load_base_config, load_model
from resqcity.common.provenance import write_meta
from resqcity.geo.config import GeoConfig
from resqcity.geo.network import RoadNetwork


class SanityConfig(ConfigModel):
    departure_date: str
    departure_hours: list[int] = Field(min_length=1)
    profiles: list[str] = Field(min_length=1)
    places: dict[str, tuple[float, float]]
    pairs: list[tuple[str, str]] = Field(min_length=1)
    holdout_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="routes NOT used for speed calibration (independent check)",
    )


# Blank columns a person fills in from Google Maps ("typically X-Y min" for the departure time).
TEMPLATE_COLUMNS = ["google_typical_min_low", "google_typical_min_high", "checked_on", "comment"]


def has_filled_values(path: Path) -> bool:
    """True if any manual-entry column of an existing template has content."""
    existing = pd.read_csv(path, dtype=str)
    cols = [c for c in TEMPLATE_COLUMNS if c in existing]
    filled = existing[cols].fillna("").apply(lambda c: c.str.strip()).ne("")
    return bool(cols) and bool(filled.any().any())


def sanity_table(network: RoadNetwork, cfg: SanityConfig, timezone: str) -> pd.DataFrame:
    names = list(cfg.places)
    lat = [cfg.places[n][0] for n in names]
    lon = [cfg.places[n][1] for n in names]
    node_ids, snap = network.nearest_nodes(lat, lon, in_city_only=True)
    node_of = dict(zip(names, node_ids.tolist(), strict=True))
    snap_of = dict(zip(names, snap.tolist(), strict=True))
    tz = ZoneInfo(timezone)
    day = datetime.fromisoformat(cfg.departure_date).date()
    rows = []
    holdout = set(cfg.holdout_pairs)
    for a, b in [*cfg.pairs, *cfg.holdout_pairs]:
        for profile in cfg.profiles:
            for hour in cfg.departure_hours:
                t = datetime.combine(day, time(hour), tzinfo=tz)
                path, seconds = network.shortest_path(node_of[a], node_of[b], t, profile=profile)
                length_m = network.path_length_m(path)
                rows.append(
                    {
                        "from": a,
                        "to": b,
                        "holdout": (a, b) in holdout,
                        "profile": profile,
                        "depart_local": t.strftime("%a %H:%M"),
                        "route_km": round(length_m / 1000, 2),
                        "minutes": round(seconds / 60, 1),
                        "avg_kph": round(length_m / seconds * 3.6, 1) if seconds > 0 else None,
                        "snap_from_m": round(snap_of[a]),
                        "snap_to_m": round(snap_of[b]),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "sanity_pairs.yaml")
    parser.add_argument(
        "--geo-overlay",
        type=Path,
        default=None,
        help="YAML merged over configs/geo.yaml (e.g. archived speed settings)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output CSV path")
    parser.add_argument(
        "--template", action="store_true", help="add blank columns for manual Google Maps times"
    )
    args = parser.parse_args(argv)
    base = load_base_config()
    geo_paths = [CONFIG_DIR / "geo.yaml", *([args.geo_overlay] if args.geo_overlay else [])]
    geo_cfg = load_model(GeoConfig, *geo_paths)
    network = RoadNetwork.load(base.paths.resolve("data_processed"), geo_cfg)
    table = sanity_table(network, load_model(SanityConfig, args.config), base.project.timezone)
    out = args.out or base.paths.resolve("results") / "geo" / "sanity_travel_times.csv"
    if args.template:
        if out.exists() and has_filled_values(out):
            raise SystemExit(f"{out} contains filled-in Google Maps times; refusing to overwrite")
        table = table.assign(**{c: "" for c in TEMPLATE_COLUMNS})
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    write_meta(
        out,
        configs=[*geo_paths, args.config],
        synthetic=False,
        extra={"note": "model travel times for a human sanity check; not observations"},
    )
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
