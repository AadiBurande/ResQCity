"""Internal vision types (never leave ``vision/``; only ``common.schemas.Event`` does)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackObs:
    """One tracked object in one frame."""

    track_id: int
    cls: str
    conf: float
    box: Box


@dataclass(frozen=True)
class FramePacket:
    """A sampled frame and its timestamp in seconds from stream start."""

    t: float
    image: np.ndarray  # BGR, HxWx3, uint8
    index: int = 0


@dataclass
class Detections:
    """Tracker output for one frame."""

    t: float
    tracks: list[TrackObs] = field(default_factory=list)


class Tracker(Protocol):
    """Detection + tracking backend (YOLO+ByteTrack in production, fakes in tests)."""

    def update(self, frame: FramePacket) -> Detections: ...


class Classifier(Protocol):
    """Window-level accident probability from a sequence of frames."""

    def predict_window(self, frames: Sequence[np.ndarray]) -> float: ...
