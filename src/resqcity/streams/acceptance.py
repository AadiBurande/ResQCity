"""Live end-to-end acceptance run: simulated RTSP cameras -> ``vision.run`` -> Events on the bus.

    uv run python -m resqcity.streams.acceptance [--duration 180] [--cameras C01 C02]

Starts the stream player with ``configs/streams_acceptance.yaml`` (one accident clip per camera,
on cue; normal clips are road footage, D-025) and one ``vision.run`` process per watched camera,
then compares the Events published on the JSONL bus with the ground truth from the scenario log.
Scoring uses the SAME rule as the offline evaluation: an Event in [onset, offset + grace] is a
detection (grace = ``resqcity.vision.eval.metrics.DETECTION_GRACE_S``); any other Event is a
false alarm. The JSON records the model hashes and operating point used. Requires MediaMTX
(``make streams`` starts it).
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.streams.config import load_streams_config
from resqcity.streams.loop_cameras import SCENARIO_LOG, start_server
from resqcity.vision.eval.metrics import DETECTION_GRACE_S

RUNTIME = PROJECT_ROOT / "data" / "runtime"
EVENTS = RUNTIME / "acceptance_events.jsonl"
CONFIG = CONFIG_DIR / "streams_acceptance.yaml"
STARTUP_S = 20.0  # vision processes load models before the first cue


def detector_provenance() -> dict[str, object]:
    """Selected clip run, its val operating point and model hashes (as used by vision.run)."""
    import pandas as pd

    results = PROJECT_ROOT / "results" / "vision"
    selection = results / "clip_selection.csv"
    if not selection.exists():
        return {"mode": "no trained model"}
    table = pd.read_csv(selection)
    run = str(table.loc[table["selected"], "run_id"].iloc[0])
    report_path = results / run / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "clip_run": run,
        "evaluation_version": report.get("evaluation_version"),
        "operating_point": report.get("operating_points_chosen_on_val", {}).get("combined"),
        "models": report.get("models"),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def score(run_start: datetime, run_end: datetime, cameras: list[str]) -> dict[str, object]:
    switches = [
        s
        for s in _read_jsonl(SCENARIO_LOG)
        if datetime.fromisoformat(s["started_at_utc"]) >= run_start
    ]
    events = [r["payload"] for r in _read_jsonl(EVENTS)]
    per_camera = []
    for cam in cameras:
        sw = next((s for s in switches if s["camera"] == cam), None)
        cam_events = sorted(
            datetime.fromisoformat(e["ts"]) for e in events if e["camera_id"] == cam
        )
        row: dict[str, object] = {"camera": cam, "events": len(cam_events), "accident_clip": None}
        if sw is None:
            row.update(detected=False, false_alarms=len(cam_events), note="no accident clip played")
            per_camera.append(row)
            continue
        start = datetime.fromisoformat(sw["started_at_utc"])
        onset_s = sw.get("accident_onset_s_in_clip") or 0.0
        offset_s = sw.get("accident_offset_s_in_clip") or onset_s
        onset = start + timedelta(seconds=onset_s)
        end = start + timedelta(seconds=offset_s + DETECTION_GRACE_S)
        hits = [t for t in cam_events if onset <= t <= end]
        row.update(
            accident_clip=sw["clip"],
            true_onset_utc=onset.isoformat(),
            detection_window_end_utc=end.isoformat(),
            detected=bool(hits),
            delay_s=(hits[0] - onset).total_seconds() if hits else None,
            false_alarms=len(cam_events) - len(hits),
        )
        per_camera.append(row)
    hours = len(cameras) * (run_end - run_start).total_seconds() / 3600.0
    return {
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "detection_rule": f"event in [onset, offset + {DETECTION_GRACE_S:g} s]",
        "detector": detector_provenance(),
        "cameras": per_camera,
        "detected": sum(bool(r["detected"]) for r in per_camera),
        "accident_clips": sum(r["accident_clip"] is not None for r in per_camera),
        "false_alarms": sum(int(r["false_alarms"]) for r in per_camera),
        "camera_hours": hours,
        "events_published": len(events),
        "note": "Live check of the end-to-end path (RTSP -> vision -> bus) on road footage. "
        "A few clips cannot estimate detector accuracy; see test_video_metrics.csv for that.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="cameras to watch (default: all; fewer = less RAM)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.start_server:
        start_server()
    cfg = load_streams_config(CONFIG)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    watched = args.cameras or cfg.cameras
    EVENTS.unlink(missing_ok=True)
    run_start = datetime.now(UTC)
    py = [sys.executable, "-m"]
    player = subprocess.Popen(
        [
            *py,
            "resqcity.streams.loop_cameras",
            "--config",
            str(CONFIG),
            "--duration",
            str(args.duration),
        ]
    )
    time.sleep(3.0)  # publishers up before readers connect
    vision = [
        subprocess.Popen(
            [
                *py,
                "resqcity.vision.run",
                "--camera",
                cam,
                "--events",
                str(EVENTS),
                "--max-seconds",
                str(args.duration - 10),
            ],
            stdout=(RUNTIME / f"acceptance_{cam}.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        for cam in watched
    ]
    for proc in vision:
        proc.wait(timeout=args.duration + STARTUP_S + 120)
    player.wait(timeout=60)
    run_end = datetime.now(UTC)
    report = score(run_start, run_end, watched)
    out = args.out or PROJECT_ROOT / "results" / "vision" / "live_acceptance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[CONFIG, CONFIG_DIR / "vision.yaml"], synthetic=False)
    print(json.dumps(report, indent=2))
    return 0 if report["events_published"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
