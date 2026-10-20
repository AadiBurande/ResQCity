"""Live end-to-end acceptance run: simulated RTSP cameras -> ``vision.run`` -> Events on the bus.

    uv run python -m resqcity.streams.acceptance [--duration 180]

Starts the stream player with ``configs/streams_acceptance.yaml`` (one accident clip per camera,
on cue) and one ``vision.run`` process per camera, then compares the Events published on the
JSONL bus with the ground truth from the scenario log (clip start + annotated onset in the clip).
Scoring follows the offline evaluation: an Event at/after the true onset (within the clip, plus
``GRACE_S``) is a detection; any other Event is a false alarm. Results:
``results/vision/live_acceptance.json``. Requires MediaMTX (``make streams`` starts it).
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
from resqcity.streams.loop_cameras import SCENARIO_LOG, clips, start_server

RUNTIME = PROJECT_ROOT / "data" / "runtime"
EVENTS = RUNTIME / "acceptance_events.jsonl"
CONFIG = CONFIG_DIR / "streams_acceptance.yaml"
GRACE_S = 15.0  # an alarm this long after the accident clip ends still counts (stream latency)
STARTUP_S = 20.0  # vision processes load models before the first cue


def clip_duration_s(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()


def score(run_start: datetime, run_end: datetime, cameras: list[str]) -> dict[str, object]:
    cfg = load_streams_config(CONFIG)
    switches = [
        json.loads(line)
        for line in SCENARIO_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip() and datetime.fromisoformat(json.loads(line)["started_at_utc"]) >= run_start
    ]
    events = (
        [
            json.loads(line)["payload"]
            for line in EVENTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if EVENTS.exists()
        else []
    )
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
        duration = clip_duration_s(
            next(p for p in clips(cfg, cam, "accident") if p.name == sw["clip"])
        )
        onset = start + timedelta(seconds=sw["accident_onset_s_in_clip"] or 0.0)
        end = start + timedelta(seconds=duration + GRACE_S)
        hits = [t for t in cam_events if onset <= t <= end]
        row.update(
            accident_clip=sw["clip"],
            true_onset_utc=onset.isoformat(),
            detected=bool(hits),
            delay_s=(hits[0] - onset).total_seconds() if hits else None,
            false_alarms=len(cam_events) - len(hits),
        )
        per_camera.append(row)
    hours = len(cameras) * (run_end - run_start).total_seconds() / 3600.0
    return {
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "cameras": per_camera,
        "detected": sum(bool(r["detected"]) for r in per_camera),
        "accident_clips": sum(r["accident_clip"] is not None for r in per_camera),
        "false_alarms": sum(int(r["false_alarms"]) for r in per_camera),
        "camera_hours": hours,
        "events_published": len(events),
        "note": "Live check of the end-to-end path (RTSP -> vision -> bus); 4 clips is far too few "
        "to estimate detector accuracy: see results/vision/<run>/test_video_metrics.csv for that.",
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
