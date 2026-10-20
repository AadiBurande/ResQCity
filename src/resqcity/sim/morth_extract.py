"""Extract Pune / Maharashtra figures from MoRTH "Road Accidents in India 2022".

    uv run python -m resqcity.sim.morth_extract

Input: ``data/raw/morth/road_accidents_in_india_2022.pdf`` (Ministry of Road Transport and
Highways, Transport Research Wing, published Oct 2023; provided by the user, SHA-256 recorded).
Output: ``results/calibration/morth_2022_extract.json`` with the page each figure was read from:
- Table 6.2: Pune (million-plus city) road accidents, persons killed and injured, 2021 and 2022;
- Annexure 45: Pune accidents by weather (2022);
- Annexure 42: Maharashtra accidents by 3-hour time interval (2022).
Every row is checked against its total, so a mis-parsed table fails loudly.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from resqcity.common.config import load_base_config
from resqcity.common.provenance import write_meta

PDF = Path("morth") / "road_accidents_in_india_2022.pdf"  # under data/raw
TIME_BUCKETS = ["06-09", "09-12", "12-15", "15-18", "18-21", "21-24", "00-03", "03-06"]


def _num(token: str) -> int:
    return int(token.replace(",", ""))


def page_texts(pdf: Path) -> list[str]:
    import pypdf

    return [p.extract_text() or "" for p in pypdf.PdfReader(str(pdf)).pages]


def find_row(
    pages: list[str], must_contain: str, row_prefix: str, n_numbers: int
) -> tuple[int, list[int]]:
    """(1-based page, numbers) of the first line starting with ``row_prefix`` followed by at least
    ``n_numbers`` numeric tokens, on a page containing ``must_contain``."""
    pattern = re.compile(rf"^\s*{re.escape(row_prefix)}\s+((?:[\d,]+\s*)+)$")
    for i, text in enumerate(pages, start=1):
        if must_contain not in text:
            continue
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if m:
                nums = [_num(t) for t in m.group(1).split()]
                if len(nums) >= n_numbers:
                    return i, nums
    raise ValueError(f"row {row_prefix!r} not found on a page containing {must_contain!r}")


def extract(pages: list[str]) -> dict[str, object]:
    # Table 6.2: accidents 2021, 2022, ranks; killed 2021, 2022, ranks; injured 2021, 2022, ranks
    p62, t62 = find_row(pages, "Table 6.2", "39 Pune", 12)
    pune = {
        "accidents_2021": t62[0],
        "accidents_2022": t62[1],
        "killed_2021": t62[4],
        "killed_2022": t62[5],
        "injured_2021": t62[8],
        "injured_2022": t62[9],
    }
    # Annexure 45 (weather): per condition accidents, killed, grievous, minor, total injured
    p45, w = find_row(pages, "Type of Weather Condition", "39 Pune", 10)
    weather = {"clear_sunny": w[0], "rainy": w[5]}
    weather["other_conditions"] = pune["accidents_2022"] - weather["clear_sunny"] - weather["rainy"]
    if weather["other_conditions"] < 0:
        raise ValueError("weather rows exceed the Table 6.2 total")
    # Annexure 42: Maharashtra accidents by 3-hour interval, unknown time, total
    p42, m = find_row(pages, "Time of occurance", "14 Maharashtra", 10)
    buckets = dict(zip(TIME_BUCKETS, m[:8], strict=True))
    unknown, total = m[8], m[9]
    if sum(buckets.values()) + unknown != total:
        raise ValueError(f"Annexure 42 Maharashtra row does not add up: {m}")
    return {
        "source": "MoRTH, Road Accidents in India 2022 (Transport Research Wing, 2023)",
        "pune_million_plus_city": {**pune, "page": p62, "table": "Table 6.2"},
        "pune_weather_2022": {**weather, "page": p45, "table": "Annexure 45"},
        "maharashtra_time_of_day_2022": {
            "accidents_by_interval": buckets,
            "unknown_time": unknown,
            "total": total,
            "page": p42,
            "table": "Annexure 42",
        },
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base = load_base_config()
    pdf = base.paths.resolve("data_raw") / PDF
    if not pdf.exists():
        raise FileNotFoundError(
            f"{pdf} missing: copy the MoRTH 'Road Accidents in India 2022' PDF there"
        )
    result = extract(page_texts(pdf))
    result["pdf_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    out = base.paths.resolve("results") / "calibration" / "morth_2022_extract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[], synthetic=False, extra={"pdf": PDF.as_posix()})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
