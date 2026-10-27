"""How much of the city the cameras actually see (phase 7, D-032 item 1).

    uv run python -m resqcity.sim.evidence_funnel --reps 8

This answers the question the whole CCTV premise rests on: of the incidents that happen, how many
does a camera detect? It needs no simulation -- only the pre-generated evidence of a week -- so it
is separated from the runner and is cheap enough to average over many replications.

The funnel has four stages and each one is bounded by a quantity an earlier phase measured, so the
output doubles as a consistency check on phases 2, 3b and 6:

1. incidents generated (base rate, D-018);
2. of those, how many fall within a camera's coverage radius (phase-2 camera siting; D-031
   measured the 40 sites covering 3.95 % of risk-weighted demand);
3. of those, how many the detector actually alarms on (D-029 measured effective detection at
   0.31, self-masking included);
4. how many alarms were raised in total, i.e. what the operators would have to look at.

Coverage is compared in OSM node IDs on both sides. ``camera_coverage`` returns internal node
INDICES, and intersecting the two numbering schemes silently reports that no incident was ever in
camera range while the detector is detecting some -- a mistake this script made before it was a
script.

Output: ``results/sim/evidence_funnel.json``.
"""

import argparse
import json
import logging

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.sim.runner import Week

OUT = PROJECT_ROOT / "results" / "sim" / "evidence_funnel.json"
log = logging.getLogger(__name__)


def measure(network: RoadNetwork, scenario: str, seed_base: int, reps: int) -> dict[str, object]:
    rows = []
    for rep in range(reps):
        week = Week(network, scenario, seed_base, rep)
        in_range = [i for i in week.incidents if int(i.node) in week.covered_nodes]
        true_events = [e for e in week.events if e.incident_id]
        detected = {e.incident_id for e in true_events}
        rows.append(
            {
                "replication": rep,
                "incidents": len(week.incidents),
                "within_camera_range": len(in_range),
                "detected_by_a_camera": len(detected),
                "alarms": len(week.events),
                "true_alarms": len(true_events),
                "calls": len(week.calls),
                "incidents_with_a_call": len({c.incident_id for c in week.calls if c.incident_id}),
            }
        )
        log.info("rep %d: %s", rep, rows[-1])

    def total(key: str) -> int:
        return sum(int(r[key]) for r in rows)

    incidents, alarms = total("incidents"), total("alarms")
    in_range_n, detected_n = total("within_camera_range"), total("detected_by_a_camera")
    return {
        "status": "SYNTHETIC incidents and the MEASURED detector model. Each stage is bounded by "
        "a quantity an earlier phase measured, so this is also a consistency check on D-018, "
        "D-029 and D-031.",
        "scenario": scenario,
        "replications": reps,
        "weeks": reps,
        "totals": {
            "incidents": incidents,
            "within_camera_range": in_range_n,
            "detected_by_a_camera": detected_n,
            "alarms": alarms,
            "true_alarms": total("true_alarms"),
            "incidents_with_a_call": total("incidents_with_a_call"),
        },
        "shares": {
            "within_camera_range": in_range_n / incidents if incidents else 0.0,
            "detected_of_in_range": detected_n / in_range_n if in_range_n else 0.0,
            "detected_of_all_incidents": detected_n / incidents if incidents else 0.0,
            "alarm_precision": total("true_alarms") / alarms if alarms else 0.0,
            "incidents_with_a_call": (
                total("incidents_with_a_call") / incidents if incidents else 0.0
            ),
        },
        "expected_from_earlier_phases": {
            "within_camera_range": 0.0395,
            "within_camera_range_source": "D-031: the 40 phase-2 sites cover 3.95 % of "
            "risk-weighted demand, and incidents are drawn proportionally to that demand",
            "detected_of_in_range": 0.306,
            "detected_of_in_range_source": "D-029: effective detection p_detect x armed_fraction "
            "= 0.306, i.e. the detector's own false alarms silence it for much of the time",
        },
        "note": "A camera-detected incident is rare enough that a single week is mostly noise; "
        "read the pooled shares, not a replication.",
        "by_replication": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--reps", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    report = measure(network, args.scenario, base.random.seed_base, args.reps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(
        OUT,
        configs=[
            CONFIG_DIR / "cameras.yaml",
            CONFIG_DIR / "fusion.yaml",
            CONFIG_DIR / "incidents.yaml",
        ],
        synthetic=True,
        seeds={"seed_base": base.random.seed_base},
    )
    shares = report["shares"]
    log.info(
        "%d weeks: %d incidents, %d in camera range (%.2f %%), %d detected (%.2f %% of all), "
        "%d alarms at %.6f %% precision",
        args.reps,
        report["totals"]["incidents"],
        report["totals"]["within_camera_range"],
        100 * shares["within_camera_range"],
        report["totals"]["detected_by_a_camera"],
        100 * shares["detected_of_all_incidents"],
        report["totals"]["alarms"],
        100 * shares["alarm_precision"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
