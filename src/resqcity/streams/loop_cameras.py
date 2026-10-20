"""Publish simulated CCTV cameras to MediaMTX and play scenario cues.

    uv run python -m resqcity.streams.loop_cameras [--start-server] [--no-scenario] [--duration 600]

Each camera loops its normal clips (``ffmpeg -re -stream_loop -1``) to ``rtsp://.../<camera_id>``.
A scenario cue (``configs/streams.yaml``) switches a camera to an accident clip at ``at_s`` seconds,
played once, then back to normal. Every switch is logged to ``data/runtime/scenario_log.jsonl``
with the clip's annotated accident onset, so live alarms can be checked against ground truth.
"""

import argparse
import itertools
import json
import logging
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from resqcity.common.config import PROJECT_ROOT
from resqcity.streams.config import StreamsConfig, find_ffmpeg, load_streams_config

log = logging.getLogger(__name__)
SCENARIO_LOG = PROJECT_ROOT / "data" / "runtime" / "scenario_log.jsonl"
COMPOSE_FILE = PROJECT_ROOT / "streams" / "docker-compose.yml"
POLL_S = 0.5


def build_command(ffmpeg: str, clip: Path, url: str, cfg: StreamsConfig, loop: bool) -> list[str]:
    t = cfg.transcode
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-re",
        "-stream_loop",
        "-1" if loop else "0",
        "-i",
        str(clip),
        "-an",
        "-c:v",
        t.codec,
        "-preset",
        t.preset,
        "-tune",
        t.tune,
        "-g",
        str(t.gop),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]


def clips(cfg: StreamsConfig, camera: str, kind: str) -> list[Path]:
    return sorted((cfg.camera_dir(camera) / kind).glob("*.mp4"))


def clip_onsets(cfg: StreamsConfig, camera: str) -> dict[str, float | None]:
    path = cfg.camera_dir(camera) / "clips.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {c["file"]: c.get("accident_onset_s") for k in data.values() for c in k}


class CameraPublisher(threading.Thread):
    def __init__(self, camera: str, cfg: StreamsConfig, ffmpeg: str, stop: threading.Event) -> None:
        super().__init__(daemon=True, name=f"pub-{camera}")
        self.camera, self.cfg, self.ffmpeg, self.stop_event = camera, cfg, ffmpeg, stop
        self.normal = itertools.cycle(clips(cfg, camera, "normal"))
        self.accidents = itertools.cycle(clips(cfg, camera, "accident"))
        self.onsets = clip_onsets(cfg, camera)
        self.trigger = threading.Event()
        self.proc: subprocess.Popen[bytes] | None = None

    def _start(self, clip: Path, loop: bool) -> None:
        cmd = build_command(self.ffmpeg, clip, self.cfg.url(self.camera), self.cfg, loop)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)

    def _stop_proc(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def run(self) -> None:
        self._start(next(self.normal), loop=True)
        log.info("%s publishing normal footage at %s", self.camera, self.cfg.url(self.camera))
        while not self.stop_event.is_set():
            if self.trigger.is_set():
                self.trigger.clear()
                self._stop_proc()
                clip = next(self.accidents)
                started = datetime.now(UTC)
                self._start(clip, loop=False)
                log_switch(self.camera, clip.name, started, self.onsets.get(clip.name))
                log.info(
                    "%s ACCIDENT clip %s (onset %ss into the clip)",
                    self.camera,
                    clip.name,
                    self.onsets.get(clip.name),
                )
                while (
                    self.proc is not None
                    and self.proc.poll() is None
                    and not self.stop_event.is_set()
                ):
                    time.sleep(POLL_S)
                if not self.stop_event.is_set():
                    self._start(next(self.normal), loop=True)
            elif self.proc is not None and self.proc.poll() is not None:
                log.warning(
                    "%s publisher exited (%s); restarting", self.camera, self.proc.returncode
                )
                self._start(next(self.normal), loop=True)
            time.sleep(POLL_S)
        self._stop_proc()


def log_switch(camera: str, clip: str, started: datetime, onset_s: float | None) -> None:
    SCENARIO_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "camera": camera,
        "clip": clip,
        "started_at_utc": started.isoformat(),
        "accident_onset_s_in_clip": onset_s,
    }
    with SCENARIO_LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record) + "\n")


def start_server() -> None:
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start-server", action="store_true", help="docker compose up -d MediaMTX first"
    )
    parser.add_argument("--no-scenario", action="store_true")
    parser.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--config", type=Path, default=None, help="alternative streams YAML")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_streams_config(args.config)
    ffmpeg = find_ffmpeg(cfg.ffmpeg)
    if args.start_server:
        start_server()
    missing = [c for c in cfg.cameras if not clips(cfg, c, "normal")]
    if missing:
        raise SystemExit(
            f"no clips for {missing}; run: uv run python -m resqcity.streams.prepare_videos"
        )
    stop = threading.Event()
    pubs = {c: CameraPublisher(c, cfg, ffmpeg, stop) for c in cfg.cameras}
    for p in pubs.values():
        p.start()
    cues = [] if args.no_scenario else sorted(cfg.scenario, key=lambda c: c.at_s)
    t0 = time.monotonic()
    try:
        while args.duration is None or time.monotonic() - t0 < args.duration:
            elapsed = time.monotonic() - t0
            while cues and cues[0].at_s <= elapsed:
                cue = cues.pop(0)
                if cue.camera in pubs:
                    pubs[cue.camera].trigger.set()
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for p in pubs.values():
            p.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
