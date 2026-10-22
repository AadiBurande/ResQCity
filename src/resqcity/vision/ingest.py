"""Frame ingest from files, RTSP or HTTP (e.g. a phone IP camera), sampled to a target fps.

Timestamps: for files, ``t`` is media time (frame index / native fps), so evaluation is exact and
independent of processing speed. For live streams, ``t`` is wall-clock seconds since the first
frame.
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from resqcity.vision.config import IngestConfig
from resqcity.vision.types import FramePacket

log = logging.getLogger(__name__)
LIVE_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")
DEFAULT_FPS = 25.0


def is_live(source: str) -> bool:
    return str(source).lower().startswith(LIVE_PREFIXES)


def resize(image: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or image.shape[1] <= width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


LIVE_TIMEOUT_MS = 10_000  # fail (and reconnect) instead of blocking forever on a dead stream


def _open(source: str) -> cv2.VideoCapture:
    if is_live(source):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            LIVE_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            LIVE_TIMEOUT_MS,
        ]
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"cannot open video source {source!r}")
    return cap


def file_frames(
    path: str | Path, cfg: IngestConfig, use_pts: bool = False
) -> Iterator[FramePacket]:
    """Sampled frames of a video file with media-time timestamps: frame index / fps by default,
    or each frame's presentation timestamp with ``use_pts`` (variable-frame-rate clips, D-027)."""
    cap = _open(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        step = max(1, round(fps / cfg.sample_fps))
        index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if use_pts else index / fps
                yield FramePacket(t=t, image=resize(image, cfg.resize_width), index=index)
            index += 1
    finally:
        cap.release()


def live_frames(url: str, cfg: IngestConfig) -> Iterator[FramePacket]:
    """Sampled frames of a live stream, reconnecting on drops (up to ``max_reconnects``)."""
    period = 1.0 / cfg.sample_fps
    start: float | None = None
    last_emit = -np.inf
    reconnects = 0
    index = 0
    while reconnects <= cfg.max_reconnects:
        try:
            cap = _open(url)
        except OSError as exc:
            reconnects += 1
            log.warning(
                "%s; retry %d/%d in %.0f s",
                exc,
                reconnects,
                cfg.max_reconnects,
                cfg.reconnect_delay_s,
            )
            time.sleep(cfg.reconnect_delay_s)
            continue
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    raise ConnectionError("stream read failed")
                now = time.monotonic()
                start = now if start is None else start
                if now - last_emit >= period:
                    last_emit = now
                    yield FramePacket(
                        t=now - start, image=resize(image, cfg.resize_width), index=index
                    )
                    index += 1
        except ConnectionError as exc:
            reconnects += 1
            log.warning("%s: %s; reconnecting (%d/%d)", url, exc, reconnects, cfg.max_reconnects)
            time.sleep(cfg.reconnect_delay_s)
        finally:
            cap.release()
    raise ConnectionError(f"{url}: gave up after {cfg.max_reconnects} reconnects")


def frames(source: str | Path, cfg: IngestConfig) -> Iterator[FramePacket]:
    source = str(source)
    return live_frames(source, cfg) if is_live(source) else file_frames(source, cfg)
