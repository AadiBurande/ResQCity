"""Replay scripted evidence through the fusion engine (phase-4 acceptance; SYNTHETIC).

    uv run python -m resqcity.sim.fusion_replay --all
    uv run python -m resqcity.sim.fusion_replay two_views_one_crash

Each ``configs/fusion_scenarios/<name>.yaml`` scripts camera alarms (at real camera sites of
``configs/cameras.yaml``, plus optional extra views of the same junction), calls (placed by an
offset in metres from a camera) and operator reviews on the real Pune road network. The replay
feeds them in time order, ticks the engine in between, records each candidate's posterior trace
and checks the expected grouping (each listed group must end in one incident, different groups
in different incidents). Output: ``results/fusion/replay_<name>_synthetic.json``; exit code 1
if a grouping check fails.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Call, Camera, Event, EventType, GeoPoint, Language, Review
from resqcity.common.schemas import SeverityFeatures as Features
from resqcity.fusion.config import (
    load_calls_config,
    load_detector_model,
    load_fusion_config,
    load_triage_config,
)
from resqcity.fusion.engine import FusionEngine
from resqcity.fusion.triage import RuleTriager
from resqcity.geo.cameras import camera_coverage, load_cameras
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.sim.background import build_background
from resqcity.sim.generator import load_incidents_config

SCENARIOS = CONFIG_DIR / "fusion_scenarios"
OUT = PROJECT_ROOT / "results" / "fusion"
log = logging.getLogger(__name__)


def _cameras(script: dict) -> list[Camera]:
    cams = load_cameras(CONFIG_DIR / "cameras.yaml")
    by_id = {c.id: c for c in cams}
    for extra in script.get("extra_cameras", []):
        base = by_id[extra["same_as"]]
        cams.append(base.model_copy(update={"id": extra["id"], "fov_note": "scripted extra view"}))
    return cams


def _offset(network: RoadNetwork, cam: Camera, dx: float, dy: float) -> GeoPoint:
    x, y = network.xy[network.index_of(cam.junction_node)]
    lat, lon = network.unproject(x + dx, y + dy)
    return GeoPoint(lat=lat, lon=lon)


def replay(name: str, network: RoadNetwork) -> dict:
    script = yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))
    cams = _cameras(script)
    by_id = {c.id: c for c in cams}
    fusion, calls_cfg, req = load_fusion_config(), load_calls_config(), load_requirements_config()
    detector = load_detector_model(PROJECT_ROOT / fusion.detector_inputs)
    background = build_background(network, load_incidents_config(), calls_cfg)
    cover = camera_coverage(network, cams, fusion.camera.coverage_radius_m.value)
    engine = FusionEngine(network, cams, detector, calls_cfg, fusion, req, background, cover)
    triager = RuleTriager(load_triage_config(), req)
    start = datetime.fromisoformat(script["start"])
    owner: dict[str, str] = {}
    trace: list[dict] = []
    triage_log: dict[str, dict] = {}

    def record(t_s: float) -> None:
        for cand in engine.candidates.values():
            trace.append({"t_s": t_s, "candidate": cand.id, "posterior": cand.posterior})

    items = sorted(script["evidence"], key=lambda e: (e["t"], e["id"]))
    tick, t_next, k = float(script["tick_s"]), 0.0, 0
    while t_next <= script["end_s"] or k < len(items):
        if k < len(items) and items[k]["t"] <= t_next:
            it = items[k]
            ts = start + timedelta(seconds=it["t"])
            if it["kind"] == "camera":
                event = Event(
                    id=it["id"], camera_id=it["camera"], ts=ts, type=EventType.ACCIDENT,
                    confidence=it["confidence"],
                    severity_features=Features(person_down=True, num_vehicles=2),
                    severity_score=0.5, required_services={"ambulance": 1, "police": 1},
                )  # fmt: skip
                out = engine.on_event(event)
            elif it["kind"] == "call":
                call = Call(
                    id=it["id"], ts=ts,
                    reported_location=_offset(network, by_id[it["near"]], *it["offset_m"]),
                    location_error_m=it["error_m"], caller_text=it["text"],
                    language=Language(it["language"]),
                )  # fmt: skip
                result = triager.triage(call)
                triage_log[it["id"]] = {"type": result.type.value, "severity": result.severity}
                out = engine.on_call(call, result)
            else:
                review = Review(id=it["id"], ts=ts, event_id=it["event"], confirmed=it["confirmed"])
                out = engine.on_review(review)
            for inc in out:
                owner[it["id"]] = inc.id
            record(float(it["t"]))
            k += 1
            continue
        engine.tick(start + timedelta(seconds=t_next))
        record(t_next)
        t_next += tick
    groups = script["expect"]["groups"]
    ids = [{owner.get(e) for e in g} for g in groups]
    ok = all(len(s) == 1 and None not in s for s in ids) and len(set().union(*ids)) == len(groups)
    final = {
        c.id: engine.incident(c, start + timedelta(seconds=script["end_s"])).model_dump(mode="json")
        for c in engine.candidates.values()
    }
    return {
        "scenario": name,
        "description": script["description"],
        "synthetic": True,
        "grouping_ok": ok,
        "expected_groups": groups,
        "evidence_to_incident": owner,
        "triage": triage_log,
        "incidents": final,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="scenario names (default with --all: every one)")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    names = args.names or (sorted(p.stem for p in SCENARIOS.glob("*.yaml")) if args.all else [])
    if not names:
        parser.error("name a scenario or use --all")
    base = load_base_config()
    network = RoadNetwork.load(base.paths.resolve("data_processed"), load_geo_config())
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        report = replay(name, network)
        path = OUT / f"replay_{name}_synthetic.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
                        newline="\n")  # fmt: skip
        configs = [
            SCENARIOS / f"{name}.yaml", CONFIG_DIR / "fusion.yaml", CONFIG_DIR / "calls.yaml",
            CONFIG_DIR / "triage.yaml", CONFIG_DIR / "cameras.yaml", CONFIG_DIR / "incidents.yaml",
        ]  # fmt: skip
        write_meta(path, configs=configs, synthetic=True)
        finals = {i: round(v["posterior"], 6) for i, v in report["incidents"].items()}
        log.info("%s: grouping %s, final posteriors %s", name,
                 "OK" if report["grouping_ok"] else "FAILED", finals)  # fmt: skip
        if not report["grouping_ok"]:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
