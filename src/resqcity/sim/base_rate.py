"""Derive the base incident rate, type mix and hour profile from cited inputs (D-018, D-025, D-026).

    uv run python -m resqcity.sim.base_rate

Inputs: ``configs/base_rate.yaml``, the PMC fire-call CSV (downloaded to ``data/raw`` if missing)
and the MoRTH extract (``resqcity.sim.morth_extract``).
Output: ``results/calibration/base_rate.json``.
- fires/yr = mean of the selected call types over the selected years;
- road accidents/yr = MoRTH Table 6.2, Pune, selected year;
- incidents/day = (fires + accidents) / 365 / (1 - extra share): crowd/other added on top;
- hour profile = MoRTH Annexure 42 Maharashtra 3-hour shares, spread evenly over each hour of
  the interval, normalised to mean 1 (unknown-time accidents excluded).
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

from pydantic import Field

from resqcity.common.config import (
    CONFIG_DIR,
    ConfigModel,
    SourcedValue,
    load_base_config,
    load_model,
)
from resqcity.common.provenance import write_meta

DAYS_PER_YEAR = 365.0
HOURS_PER_BUCKET = 3


class FireCalls(ConfigModel):
    url: str
    raw_file: Path
    source: str
    call_types: list[str] = Field(min_length=1)
    years: list[int] = Field(min_length=1)


class RoadAccidents(ConfigModel):
    extract_file: Path
    year: int
    source: str


class HourProfile(ConfigModel):
    source: str


class BaseRateConfig(ConfigModel):
    fire_calls: FireCalls
    road_accidents: RoadAccidents
    hour_profile: HourProfile
    assumed_extra_share: SourcedValue[float]
    caveats: list[str]


def fire_calls_per_year(csv_path: Path, call_types: list[str], years: list[int]) -> float:
    with csv_path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    wanted = {t.strip().lower() for t in call_types}
    total = 0.0
    for row in rows:
        if row["Type of Call"].strip().lower() in wanted:
            total += sum(float(row[str(y)]) for y in years)
    return total / len(years)


def hourly_profile(buckets: dict[str, int]) -> list[float]:
    """3-hour bucket counts ("06-09": n, ...) -> 24 hourly factors with mean 1."""
    hourly = [0.0] * 24
    for label, count in buckets.items():
        start = int(label.split("-")[0])
        for h in range(start, start + HOURS_PER_BUCKET):
            hourly[h % 24] = count / HOURS_PER_BUCKET
    mean = sum(hourly) / 24
    return [round(v / mean, 4) for v in hourly]


def derive(cfg: BaseRateConfig, csv_path: Path, extract: dict) -> dict[str, object]:
    fires = fire_calls_per_year(csv_path, cfg.fire_calls.call_types, cfg.fire_calls.years)
    accidents = float(extract["pune_million_plus_city"][f"accidents_{cfg.road_accidents.year}"])
    core = fires + accidents
    extra = cfg.assumed_extra_share.value
    killed = extract["pune_million_plus_city"][f"killed_{cfg.road_accidents.year}"]
    return {
        "fires_per_year": fires,
        "accidents_per_year": accidents,
        "accident_plus_fire_per_day": core / DAYS_PER_YEAR,
        "incidents_per_day": core / DAYS_PER_YEAR / (1 - extra),
        "type_share_accident": accidents / core * (1 - extra),
        "type_share_fire": fires / core * (1 - extra),
        "type_share_crowd_other": extra,
        "hour_profile": hourly_profile(
            extract["maharashtra_time_of_day_2022"]["accidents_by_interval"]
        ),
        "pune_killed_per_100_accidents": 100.0 * killed / accidents,
        "million_plus_cities_killed_per_100_accidents": 100.0
        * extract["all_million_plus_cities"][f"killed_{cfg.road_accidents.year}"]
        / extract["all_million_plus_cities"][f"accidents_{cfg.road_accidents.year}"],
        "pune_rainy_weather_share": extract["pune_weather_2022"]["rainy"] / accidents,
        "sources": {
            "fires": cfg.fire_calls.source,
            "road_accidents": cfg.road_accidents.source,
            "hour_profile": cfg.hour_profile.source,
            "extra_share": cfg.assumed_extra_share.source,
        },
        "caveats": cfg.caveats,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base = load_base_config()
    cfg_path = CONFIG_DIR / "base_rate.yaml"
    cfg = load_model(BaseRateConfig, cfg_path)
    csv_path = base.paths.resolve("data_raw") / cfg.fire_calls.raw_file
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg.fire_calls.url, csv_path)
    extract_path = base.paths.resolve("results") / cfg.road_accidents.extract_file
    extract = json.loads(extract_path.read_text(encoding="utf-8"))
    result = derive(cfg, csv_path, extract)
    out = base.paths.resolve("results") / "calibration" / "base_rate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[cfg_path], synthetic=False, extra={"extract": str(extract_path.name)})
    shown = {k: v for k, v in result.items() if k not in ("sources", "caveats")}
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
