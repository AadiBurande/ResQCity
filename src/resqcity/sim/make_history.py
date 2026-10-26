"""Write ``configs/scenarios/history_3y.yaml``: three years of synthetic history for forecasting.

    uv run python -m resqcity.sim.make_history

The evaluation scenarios are single weeks; the forecaster needs years, with a seasonal rain cycle
and a recurring festival to learn from. This script writes that scenario as DATA (like
``configs/cameras.yaml``), so the long rain array stays out of the code.

Rain: a daily total in mm per day, drawn once here with a fixed seed from a monthly profile
(wet-day probability plus gamma intensity, rescaled so each month hits its target). A very dry
month can draw no wet day at all and then stays completely dry, which is what the profile's low
wet-day probability means; the target is met whenever any rain falls. The array is
as long as the scenario, NOT 365 days, so rain is **not** a deterministic function of the calendar
-- otherwise a model given only the date would know the rain exactly and the informed/realistic
comparison of phase 5 would collapse. The hourly shape sums to 1, so a day factor IS that day's
rainfall in mm.

Festival: the Ganeshotsav block of ``configs/scenarios/ganeshotsav.yaml`` is reused verbatim
(routes, multipliers) with windows placed in each year. Placement is ILLUSTRATIVE: the real
festival moves with the lunar calendar, which this fixed-date placement does not model.
"""

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from resqcity.common.config import CONFIG_DIR

IST = ZoneInfo("Asia/Kolkata")
OUT = CONFIG_DIR / "scenarios" / "history_3y.yaml"
SOURCE_SCENARIO = CONFIG_DIR / "scenarios" / "ganeshotsav.yaml"
START = datetime(2023, 1, 2, tzinfo=IST)  # a Monday; history ends before the 2026 eval scenarios
WEEKS = 156  # 1092 days = whole weeks, so weekday structure is balanced
RAIN_SEED = 20260126

# Illustrative monthly rainfall for Pune (mm). IMD station normals could not be confirmed at the
# source on 2026-10-26 (only third-party climate pages, no station stated), so this is UNVERIFIED:
# it carries the right seasonal shape (dry Nov-May, monsoon Jun-Sep, ~760 mm/year), nothing more.
MONTHLY_MM = {1: 1, 2: 1, 3: 3, 4: 10, 5: 30, 6: 150, 7: 180, 8: 140, 9: 140, 10: 80, 11: 20, 12: 3}
WET_P_PER_MM = 1 / 200.0
WET_P_RANGE = (0.03, 0.75)
GAMMA_SHAPE = 1.2
# Rain falls mostly in the afternoon and evening (same shape as configs/scenarios/monsoon.yaml).
HOUR_SHAPE = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.4, 0.4, 0.8, 1.2, 1.5, 1.2, 0.9, 0.6, 0.4, 0.3, 0.2, 0.2]  # fmt: skip
FESTIVAL_START = (9, 1)  # (month, day) of the first evening window each year
FESTIVAL_EVENINGS = 10  # evening crowds, then a two-day immersion procession

log = logging.getLogger(__name__)


def daily_rain_mm(start: datetime, days: int, seed: int) -> list[float]:
    """One rainfall total (mm) per day: wet/dry draw then gamma intensity, rescaled per month."""
    rng = np.random.default_rng(seed)
    dates = [start.date() + timedelta(days=d) for d in range(days)]
    out = np.zeros(days)
    for key in sorted({(d.year, d.month) for d in dates}):
        idx = np.array([i for i, d in enumerate(dates) if (d.year, d.month) == key])
        target = MONTHLY_MM[key[1]] * len(idx) / _days_in_month(*key)  # partial months pro-rata
        p_wet = float(np.clip(MONTHLY_MM[key[1]] * WET_P_PER_MM, *WET_P_RANGE))
        wet = rng.random(len(idx)) < p_wet
        amount = np.where(wet, rng.gamma(GAMMA_SHAPE, 1.0, len(idx)), 0.0)
        if amount.sum() > 0:
            amount *= target / amount.sum()
        out[idx] = amount
    return [round(float(x), 3) for x in out]


def _days_in_month(year: int, month: int) -> int:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return (nxt - date(year, month, 1)).days


def festival_windows(start: datetime, days: int) -> list[dict]:
    """Evening windows then a two-day immersion, once per year the scenario spans."""
    evenings, closed_full, closed_half = [], [], []
    for year in sorted({(start.date() + timedelta(days=d)).year for d in range(days)}):
        first = (date(year, *FESTIVAL_START) - start.date()).days
        if first < 0 or first + FESTIVAL_EVENINGS + 1 >= days:
            continue
        evenings += list(range(first, first + FESTIVAL_EVENINGS))
        closed_full.append(first + FESTIVAL_EVENINGS)
        closed_half.append(first + FESTIVAL_EVENINGS + 1)
    return [
        {"days": evenings, "start_hour": 18, "end_hour": 24, "effect": "slowed"},
        {"days": closed_full, "start_hour": 9, "end_hour": 24, "effect": "closed"},
        {"days": closed_half, "start_hour": 0, "end_hour": 14, "effect": "closed"},
    ]


def build(start: datetime, days: int, seed: int) -> dict:
    festival = yaml.safe_load(SOURCE_SCENARIO.read_text(encoding="utf-8"))["festival"]
    festival["windows"] = festival_windows(start, days)
    total = sum(HOUR_SHAPE)
    return {
        "name": "history_3y",
        "description": (
            f"{days // 7} weeks ({days} days) of synthetic history for the phase-5 forecaster: "
            "seasonal rain and a yearly festival. Not an evaluation scenario."
        ),
        "synthetic": True,
        "start": start.isoformat(),
        "days": days,
        "incident_rate_multiplier": 1.0,
        "rain": {
            "hourly_mm_h": [round(h / total, 5) for h in HOUR_SHAPE],
            "day_factors": daily_rain_mm(start, days, seed),
        },
        "flooding": None,
        "festival": festival,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weeks", type=int, default=WEEKS)
    parser.add_argument("--seed", type=int, default=RAIN_SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    days = args.weeks * 7
    doc = build(START, days, args.seed)
    header = (
        f"# GENERATED by `python -m resqcity.sim.make_history` (seed {args.seed}). Do not edit by\n"
        "# hand; change src/resqcity/sim/make_history.py instead. SYNTHETIC history for the\n"
        "# phase-5 forecaster only -- never an evaluation scenario.\n"
        "# Rain: illustrative monthly profile for Pune, UNVERIFIED (IMD station normals could not\n"
        "# be confirmed at the source on 2026-10-26). Daily totals are drawn once, per day, so\n"
        "# rain is not a deterministic function of the calendar.\n"
        "# Festival: Ganeshotsav block reused from scenarios/ganeshotsav.yaml; the fixed\n"
        "# 1 September placement is ILLUSTRATIVE (the real festival moves with the lunar year).\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=100)
    Path(OUT).write_text(header + body, encoding="utf-8", newline="\n")
    rain = doc["rain"]["day_factors"]
    log.info(
        "%s: %d days, rain %.0f mm/year over %d wet days",
        OUT.relative_to(CONFIG_DIR.parent),
        days,
        sum(rain) / (days / 365.25),
        sum(1 for x in rain if x > 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
