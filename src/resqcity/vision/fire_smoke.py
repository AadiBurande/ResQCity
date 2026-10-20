"""Fire / smoke detection with temporal persistence (fine-tuned YOLO on D-Fire)."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from resqcity.vision.config import FireSmokeConfig
from resqcity.vision.types import Box, FramePacket


@dataclass
class FireObs:
    fire: float  # max confidence of a fire box in the frame (0 if none)
    smoke: float
    boxes: list[tuple[str, float, Box]]


class Persistence:
    """True once a class has been seen in ``n`` consecutive sampled frames."""

    def __init__(self, n: int) -> None:
        self.history: deque[bool] = deque(maxlen=n)

    def update(self, present: bool) -> bool:
        self.history.append(present)
        return len(self.history) == self.history.maxlen and all(self.history)


class FireSmokeDetector:
    def __init__(self, weights: Path, cfg: FireSmokeConfig, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.cfg = cfg
        self.device = device
        self.fire_persist = Persistence(cfg.persistence_frames)
        self.smoke_persist = Persistence(cfg.persistence_frames)

    def observe(self, frame: FramePacket) -> FireObs:
        result = self.model.predict(
            frame.image, conf=self.cfg.conf, device=self.device, verbose=False
        )[0]
        boxes = []
        best = {"fire": 0.0, "smoke": 0.0}
        if result.boxes is not None:
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                strict=True,
            ):
                name = result.names[int(cls)]
                boxes.append((name, float(conf), Box(*map(float, xyxy))))
                if name in best:
                    best[name] = max(best[name], float(conf))
        return FireObs(best["fire"], best["smoke"], boxes)

    def update(self, frame: FramePacket) -> tuple[FireObs, bool]:
        """Observation and whether fire OR smoke has persisted long enough to raise an event."""
        obs = self.observe(frame)
        fire = self.fire_persist.update(obs.fire > 0)
        smoke = self.smoke_persist.update(obs.smoke > 0)
        return obs, fire or smoke
