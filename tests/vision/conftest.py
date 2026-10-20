"""Vision test fixtures: config, scripted fake tracker and synthetic scenes (no GPU, no models)."""

from collections.abc import Callable

import numpy as np
import pytest

from resqcity.vision.config import VisionConfig, load_vision_config
from resqcity.vision.types import Box, Detections, FramePacket, TrackObs

FPS = 5.0
H = 40.0  # box height (px) of a car; speeds in box-heights/s = px/s / H


@pytest.fixture(scope="session")
def vcfg() -> VisionConfig:
    return load_vision_config()


def car(
    tid: int, x: float, y: float = 200.0, w: float = 60.0, h: float = H, cls: str = "car"
) -> TrackObs:
    return TrackObs(tid, cls, 0.9, Box(x, y, x + w, y + h))


class ScriptedTracker:
    """Returns pre-scripted detections per frame time (stands in for YOLO + ByteTrack)."""

    def __init__(self, script: Callable[[float], list[TrackObs]]) -> None:
        self.script = script

    def update(self, frame: FramePacket) -> Detections:
        return Detections(t=frame.t, tracks=self.script(frame.t))


CONTACT_GAP = 30.0  # px of overlap at contact (IoU ~ 0.33 for 60x40 boxes)
CRASH_T = (600 - CONTACT_GAP) / (2 * 4 * H)  # time the two cars touch (~1.78 s)


def collision_script(t: float) -> list[TrackObs]:
    """Two cars approach head-on at 4 box-heights/s, touch at CRASH_T and both stop (a crash)."""
    tt = min(t, CRASH_T)
    return [car(1, 100 + 4 * H * tt), car(2, 700 - 4 * H * tt)]


def parked_script(t: float) -> list[TrackObs]:
    """Two parked cars overlapping (queue): no motion at all."""
    return [car(1, 300), car(2, 330)]


def free_flow_script(t: float) -> list[TrackObs]:
    """Two cars passing in separate lanes at constant speed: no accident cue."""
    speed = 3 * H
    return [car(1, 50 + speed * t, y=100), car(2, 800 - speed * t, y=300)]


def frames(duration: float, fps: float = FPS) -> list[FramePacket]:
    img = np.zeros((480, 960, 3), np.uint8)
    return [FramePacket(t=i / fps, image=img, index=i) for i in range(int(duration * fps))]
