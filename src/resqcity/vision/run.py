"""Run the vision pipeline on one camera and publish Events to the bus.

    uv run python -m resqcity.vision.run --camera C01 [--source URL_OR_FILE] [--preview-port 8091]

Source: ``--source`` (file, RTSP, or a phone IP-camera HTTP URL) > the camera's ``stream_url`` in
``configs/cameras.yaml`` > ``rtsp://127.0.0.1:8554/<camera_id>`` (simulated MediaMTX stream).
Models: the newest clip-classifier run under ``data/models/clip/`` (+ its stacker) and fire run
under ``data/models/fire/``; without them it runs in heuristics-only mode and says so.
Events: JSON lines in ``data/runtime/events.jsonl`` (topic ``events.vision``) and the log.
"""

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from resqcity.common.bus import JsonlBus
from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Camera
from resqcity.vision.config import load_vision_config
from resqcity.vision.ingest import frames, is_live
from resqcity.vision.pipeline import CameraPipeline
from resqcity.vision.privacy import Anonymizer, require_source_allowed
from resqcity.vision.scoring import Stacker

log = logging.getLogger(__name__)
DEFAULT_RTSP = "rtsp://127.0.0.1:8554/{camera_id}"
EVENTS_FILE = PROJECT_ROOT / "data" / "runtime" / "events.jsonl"


def load_cameras(path: Path) -> list[Camera]:
    """Cameras from the generated YAML (vision may only depend on common, not geo)."""
    import yaml
    from pydantic import TypeAdapter

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TypeAdapter(list[Camera]).validate_python(data.get("cameras", []))


def deployed_detector(results: Path, models: Path) -> tuple[Path, Path | None, dict | None] | None:
    """(clip run dir, stacker path, operating point) of the deployed detector.

    Preference: ``results/vision/deployed_detector.json`` (evaluation v4, cross-fitted, D-026), then
    the older single-split selection (``clip_selection.csv`` + that run's ``report.json``)."""
    import json

    deployed = results / "deployed_detector.json"
    if deployed.exists():
        d = json.loads(deployed.read_text(encoding="utf-8"))
        run = models / d["clip_run"]
        if run.exists():
            return run, PROJECT_ROOT / d["stacker"], d["operating_point"]
    selection = results / "clip_selection.csv"
    if selection.exists():
        import pandas as pd

        table = pd.read_csv(selection)
        chosen = table.loc[table["selected"], "run_id"]
        if len(chosen) and (models / chosen.iloc[0]).exists():
            run = models / chosen.iloc[0]
            report = results / run.name / "report.json"
            op = (
                json.loads(report.read_text(encoding="utf-8"))
                .get("operating_points_chosen_on_val", {})
                .get("combined")
                if report.exists()
                else None
            )
            return run, results / run.name / "stacker.json", op
    run = newest(models)
    return (run, None, None) if run is not None else None


def deployed_extras(results: Path) -> dict[str, object]:
    """Heuristics detector and verifier of the deployed detector (evaluation v5; D-027)."""
    import json

    deployed = results / "deployed_detector.json"
    d = json.loads(deployed.read_text(encoding="utf-8")) if deployed.exists() else {}
    return {
        "detector": d.get("detector", "coco"),
        "verifier": d.get("verifier", "none"),
        "verifier_tau": d.get("verifier_tau"),
    }


def newest(path: Path) -> Path | None:
    runs = sorted((p for p in path.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--clip-run", type=Path, default=None, help="data/models/clip/<run>")
    parser.add_argument("--fire-run", type=Path, default=None, help="data/models/fire/<run>")
    parser.add_argument("--no-models", action="store_true", help="heuristics only")
    parser.add_argument("--device", default=None, help="override detector.device (cpu, cuda:0)")
    parser.add_argument(
        "--privacy",
        choices=["faces_plates", "full", "off"],
        default=None,
        help="override privacy.mode; 'off' only for local files / simulated 127.0.0.1 streams",
    )
    parser.add_argument("--events", type=Path, default=EVENTS_FILE)
    parser.add_argument(
        "--no-verifier", action="store_true", help="skip the deployed VLM verifier (no GPU)"
    )
    parser.add_argument("--preview-port", type=int, default=None)
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="stop after this stream time"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from resqcity.vision.detect_track import (
        DualTracker,
        YoloTracker,
        resolve_device,
        uvh26_vehicle_classes,
    )

    base, cfg = load_base_config(), load_vision_config()
    cameras = {c.id: c for c in load_cameras(CONFIG_DIR / "cameras.yaml")}
    if args.camera not in cameras:
        parser.error(f"unknown camera {args.camera!r}; see configs/cameras.yaml")
    camera = cameras[args.camera]
    source = args.source or camera.stream_url or DEFAULT_RTSP.format(camera_id=camera.id)
    if args.device:
        cfg = cfg.model_copy(
            update={"detector": cfg.detector.model_copy(update={"device": args.device})}
        )
    device = resolve_device(cfg.detector.device)
    if args.privacy:
        cfg = cfg.model_copy(
            update={"privacy": cfg.privacy.model_copy(update={"mode": args.privacy})}
        )
    try:
        require_source_allowed(cfg.privacy, source)
    except PermissionError as exc:
        parser.error(str(exc))
    if cfg.privacy.mode == "off":
        log.warning("privacy OFF: unblurred preview of public-dataset footage (local demo only)")

    classifier = stacker = fire = verifier = None
    threshold = None
    results = base.paths.resolve("results") / "vision"
    extras = deployed_extras(results) if not (args.no_models or args.clip_run) else {}
    if extras.get("detector") == "uvh26":  # Indian vehicle classes; people from COCO
        vehicles = sorted(set(cfg.detector.vehicle_classes) | uvh26_vehicle_classes(cfg.detector))
        cfg = cfg.model_copy(
            update={"detector": cfg.detector.model_copy(update={"vehicle_classes": vehicles})}
        )
    if not args.no_models:
        models = PROJECT_ROOT / "data" / "models" / "clip"
        chosen = (
            (args.clip_run, None, None) if args.clip_run else deployed_detector(results, models)
        )
        if chosen is not None:
            from resqcity.vision.clip_classifier import ClipClassifier

            clip_run, stacker_path, op = chosen
            classifier = ClipClassifier(clip_run, cfg.clip_classifier, device)
            stacker = Stacker.load(stacker_path) if stacker_path and stacker_path.exists() else None
            if op is not None:
                threshold = float(op["threshold"])
                alarm = cfg.alarm.model_copy(
                    update={"consecutive_windows": int(op["consecutive_windows"])}
                )
                cfg = cfg.model_copy(update={"alarm": alarm})
                log.info("detector %s, operating point %s", clip_run.name, op)
        fire_run = args.fire_run or newest(PROJECT_ROOT / "data" / "models" / "fire")
        if fire_run is not None and (fire_run / "best.pt").exists():
            from resqcity.vision.fire_smoke import FireSmokeDetector

            fire = FireSmokeDetector(fire_run / "best.pt", cfg.fire_smoke, device)
        name = extras.get("verifier", "none")
        if name != "none" and cfg.verifier is not None and not args.no_verifier:
            if device.startswith("cuda"):
                from resqcity.vision.verifier import load_verifier

                verifier = load_verifier(str(name), cfg.verifier, device)
            else:
                log.warning("verifier %s needs a GPU; running without it", name)
    mode = "combined" if stacker else ("classifier+heuristics" if classifier else "heuristics only")
    log.info(
        "camera %s, source %s, device %s, mode %s, fire %s, detector %s, verifier %s",
        camera.id,
        source,
        device,
        mode,
        bool(fire),
        extras.get("detector", "coco"),
        extras.get("verifier", "none") if verifier is not None else "none",
    )

    on_preview = None
    if args.preview_port:
        from resqcity.vision.preview import FrameStore, serve

        store = FrameStore()
        serve(store, args.preview_port)
        on_preview = store.update
        log.info(
            "preview: http://127.0.0.1:%d/stream.mjpg (privacy mode %s)",
            args.preview_port,
            cfg.privacy.mode,
        )

    pipeline = CameraPipeline(
        camera,
        cfg,
        DualTracker(cfg.detector)
        if extras.get("detector") == "uvh26"
        else YoloTracker(cfg.detector),
        Anonymizer(cfg.privacy, cfg.detector.vehicle_classes, cfg.detector.person_class),
        load_requirements_config(),
        JsonlBus(args.events),
        classifier=classifier,
        stacker=stacker,
        fire=fire,
        threshold=threshold,
        start_time=datetime.now(UTC),
        on_preview=on_preview,
        source=source,
        verifier=verifier,
        verifier_tau=extras.get("verifier_tau") if verifier is not None else None,
    )
    t0 = time.perf_counter()
    n = 0
    for frame in frames(source, cfg.ingest):
        pipeline.process(frame)
        n += 1
        if args.max_seconds is not None and frame.t >= args.max_seconds:
            break
    elapsed = time.perf_counter() - t0
    log.info(
        "%d frames in %.1f s (%.1f fps), %d events -> %s",
        n,
        elapsed,
        n / max(elapsed, 1e-9),
        len(pipeline.events),
        args.events,
    )
    return 0 if not is_live(source) or n else 1


if __name__ == "__main__":
    raise SystemExit(main())
