"""Live end-to-end acceptance run: simulated RTSP cameras -> ``vision.run`` -> Events on the bus.

    uv run python -m resqcity.streams.acceptance [--duration 180] [--cameras C01 C02]

Starts the stream player with ``configs/streams_acceptance.yaml`` (one accident clip per camera,
on cue; normal clips are road footage, D-025) and one ``vision.run`` process per watched camera,
then compares the Events published on the JSONL bus with the ground truth from the scenario log.
Only ``accident`` Events are scored (other types are counted separately). The rule follows the
offline evaluation, an accident Event in [onset, offset + grace] is a detection (grace =
``resqcity.vision.eval.metrics.DETECTION_GRACE_S``), except that the window also ends when the
accident clip ends: afterwards the camera shows other footage, so an alarm there cannot be
attributed to the accident. Every other accident Event is a false alarm; those within
``cut_effect_s`` after a scene cut (clip switch or loop, which also forces an RTSP reconnect) are
flagged, because real cameras have no cuts. The JSON records the model hashes and operating point
used. Requires MediaMTX (``make streams`` starts it).
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.streams.config import load_streams_config
from resqcity.streams.loop_cameras import SCENARIO_LOG, start_server
from resqcity.vision.config import load_vision_config
from resqcity.vision.eval.metrics import DETECTION_GRACE_S

RUNTIME = PROJECT_ROOT / "data" / "runtime"
EVENTS = RUNTIME / "acceptance_events.jsonl"
CONFIG = CONFIG_DIR / "streams_acceptance.yaml"
STARTUP_S = 20.0  # vision processes load models before the first cue
ACCIDENT = "accident"


def detector_provenance() -> dict[str, object]:
    """Clip run, stacker, operating point and model hashes, resolved exactly as ``vision.run``
    resolves the deployed detector (``results/vision/deployed_detector.json`` first)."""
    from resqcity.vision.eval.evaluate import sha16
    from resqcity.vision.run import deployed_detector

    results = PROJECT_ROOT / "results" / "vision"
    chosen = deployed_detector(results, PROJECT_ROOT / "data" / "models" / "clip")
    if chosen is None:
        return {"mode": "no trained model"}
    run, stacker, op = chosen
    deployed = results / "deployed_detector.json"
    return {
        "clip_run": run.name,
        "evaluation": (
            json.loads(deployed.read_text(encoding="utf-8")).get("evaluation")
            if deployed.exists()
            else None
        ),
        "stacker": stacker.relative_to(PROJECT_ROOT).as_posix() if stacker else None,
        "operating_point": op,
        "models": {
            "clip_weights_sha256_16": sha16(run / "clip_head.pt"),
            "stacker_sha256_16": sha16(stacker) if stacker else None,
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def cut_effect_s(consecutive_windows: int) -> float:
    """How long after a scene cut an alarm can still involve pre-cut frames: the RTSP reconnect
    delay plus the span of the ``consecutive_windows`` windows an alarm needs."""
    cfg = load_vision_config()
    clip = cfg.clip_classifier
    return cfg.ingest.reconnect_delay_s + clip.window_s + (consecutive_windows - 1) * clip.stride_s


def scene_cuts(switches: list[dict], camera: str, run_end: datetime) -> list[datetime]:
    """Times the camera's picture jumps: every clip start, and every restart of a looped clip."""
    own = sorted((s for s in switches if s["camera"] == camera), key=lambda s: s["started_at_utc"])
    cuts = []
    for i, s in enumerate(own):
        start = datetime.fromisoformat(s["started_at_utc"])
        stop = datetime.fromisoformat(own[i + 1]["started_at_utc"]) if i + 1 < len(own) else run_end
        cuts.append(start)
        if s.get("looped") and s.get("clip_duration_s"):
            t = start + timedelta(seconds=s["clip_duration_s"])
            while t < stop:
                cuts.append(t)
                t += timedelta(seconds=s["clip_duration_s"])
    return cuts


def score(
    run_start: datetime, run_end: datetime, cameras: list[str], cut_s: float
) -> dict[str, object]:
    switches = [
        s
        for s in _read_jsonl(SCENARIO_LOG)
        if datetime.fromisoformat(s["started_at_utc"]) >= run_start
    ]
    events = [r["payload"] for r in _read_jsonl(EVENTS)]
    per_camera = []
    for cam in cameras:
        cam_events = [e for e in events if e["camera_id"] == cam]
        acc = sorted(datetime.fromisoformat(e["ts"]) for e in cam_events if e["type"] == ACCIDENT)
        cuts = scene_cuts(switches, cam, run_end)

        def near_cut(t: datetime, cuts: list[datetime] = cuts) -> bool:
            return any(timedelta(0) <= t - c <= timedelta(seconds=cut_s) for c in cuts)

        row: dict[str, object] = {
            "camera": cam,
            "accident_events": len(acc),
            "other_events": dict(Counter(e["type"] for e in cam_events if e["type"] != ACCIDENT)),
            "scene_cuts": len(cuts),
            "accident_clip": None,
        }
        own = [s for s in switches if s["camera"] == cam]
        sw = next((s for s in own if s.get("kind", ACCIDENT) == ACCIDENT), None)
        if sw is None:
            row.update(
                detected=False,
                false_alarms=len(acc),
                false_alarms_near_scene_cut=sum(map(near_cut, acc)),
                note="no accident clip played",
            )
            per_camera.append(row)
            continue
        start = datetime.fromisoformat(sw["started_at_utc"])
        following = [s for s in own if s["started_at_utc"] > sw["started_at_utc"]]
        if following:
            clip_end = datetime.fromisoformat(min(s["started_at_utc"] for s in following))
        elif sw.get("clip_duration_s"):
            clip_end = start + timedelta(seconds=sw["clip_duration_s"])
        else:
            clip_end = None
        onset_s = sw.get("accident_onset_s_in_clip") or 0.0
        offset_s = sw.get("accident_offset_s_in_clip") or onset_s
        onset = start + timedelta(seconds=onset_s)
        rule_end = start + timedelta(seconds=offset_s + DETECTION_GRACE_S)
        end = min(rule_end, clip_end) if clip_end else rule_end
        hits = [t for t in acc if onset <= t <= end]
        false_alarms = [t for t in acc if t not in hits]
        row.update(
            accident_clip=sw["clip"],
            true_onset_utc=onset.isoformat(),
            clip_end_utc=clip_end.isoformat() if clip_end else None,
            detection_window_end_utc=end.isoformat(),
            detected=bool(hits),
            delay_s=(hits[0] - onset).total_seconds() if hits else None,
            events_after_clip_end_within_offline_window=sum(end < t <= rule_end for t in acc),
            false_alarms=len(false_alarms),
            false_alarms_near_scene_cut=sum(map(near_cut, false_alarms)),
        )
        per_camera.append(row)
    hours = len(cameras) * (run_end - run_start).total_seconds() / 3600.0
    return {
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "detection_rule": f"accident event in [onset, min(offset + {DETECTION_GRACE_S:g} s, "
        "clip end)]",
        "cut_effect_s": cut_s,
        "detector": detector_provenance(),
        "cameras": per_camera,
        "detected": sum(bool(r["detected"]) for r in per_camera),
        "accident_clips": sum(r["accident_clip"] is not None for r in per_camera),
        "false_alarms": sum(int(r["false_alarms"]) for r in per_camera),
        "false_alarms_near_scene_cut": sum(
            int(r["false_alarms_near_scene_cut"]) for r in per_camera
        ),
        "camera_hours": hours,
        "events_published": len(events),
        "note": "Live check of the end-to-end path (RTSP -> vision -> bus) on road footage. "
        "A few clips cannot estimate detector accuracy; see results/vision/crossfit_v4/ for that. "
        "Scene cuts and reconnects are artefacts of the simulated cameras.",
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
    op = detector_provenance().get("operating_point") or {}
    k = int(op.get("consecutive_windows", load_vision_config().alarm.consecutive_windows))
    report = score(run_start, run_end, watched, cut_effect_s(k))
    out = args.out or PROJECT_ROOT / "results" / "vision" / "live_acceptance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(out, configs=[CONFIG, CONFIG_DIR / "vision.yaml"], synthetic=False)
    print(json.dumps(report, indent=2))
    return 0 if report["events_published"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
