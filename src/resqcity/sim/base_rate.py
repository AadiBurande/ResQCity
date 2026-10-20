"""Derive the base incident rate and type mix from cited inputs (D-018, corrected in D-025).

    uv run python -m resqcity.sim.base_rate

Reads ``configs/base_rate.yaml`` and the PMC fire-call CSV (downloaded to ``data/raw`` if
missing) and writes ``results/calibration/base_rate.json``:
fires/yr = mean of the selected call types over the selected years; accidents/yr = road deaths /
(severity index / 100); incidents/day = (fires + accidents) / 365 / (1 - extra share), so the
accident + fire rate equals the cited figure and crowd/other events are added on top.
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


class FireCalls(ConfigModel):
    url: str
    raw_file: Path
    source: str
    call_types: list[str] = Field(min_length=1)
    years: list[int] = Field(min_length=1)


class BaseRateConfig(ConfigModel):
    fire_calls: FireCalls
    road_deaths_per_year: SourcedValue[float]
    severity_index_deaths_per_100_accidents: SourcedValue[float]
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


def derive(cfg: BaseRateConfig, csv_path: Path) -> dict[str, object]:
    fires = fire_calls_per_year(csv_path, cfg.fire_calls.call_types, cfg.fire_calls.years)
    accidents = cfg.road_deaths_per_year.value / (
        cfg.severity_index_deaths_per_100_accidents.value / 100
    )
    core = fires + accidents
    extra = cfg.assumed_extra_share.value
    per_day = core / DAYS_PER_YEAR / (1 - extra)
    return {
        "fires_per_year": fires,
        "accidents_per_year": accidents,
        "accident_plus_fire_per_day": core / DAYS_PER_YEAR,
        "incidents_per_day": per_day,
        "type_share_accident": accidents / core * (1 - extra),
        "type_share_fire": fires / core * (1 - extra),
        "type_share_crowd_other": extra,
        "sources": {
            "fires": cfg.fire_calls.source,
            "road_deaths": cfg.road_deaths_per_year.source,
            "severity_index": cfg.severity_index_deaths_per_100_accidents.source,
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
    result = derive(cfg, csv_path)
    out = base.paths.resolve("results") / "calibration" / "base_rate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[cfg_path], synthetic=False)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("sources", "caveats")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
