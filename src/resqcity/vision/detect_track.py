"""YOLO detection + ByteTrack tracking (Ultralytics, AGPL-3.0; accepted in D-019).

Imports of torch/ultralytics are lazy so the rest of the pipeline (and the tests) work without them.
"""

from pathlib import Path

from resqcity.common.config import PROJECT_ROOT
from resqcity.vision.config import DetectorConfig
from resqcity.vision.types import Box, Detections, FramePacket, TrackObs

MODEL_DIR = PROJECT_ROOT / "data" / "models"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloTracker:
    """Stateful tracker for ONE stream (ByteTrack keeps track ids across ``update`` calls)."""

    def __init__(self, cfg: DetectorConfig, weights: str | Path | None = None) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        w = Path(weights or cfg.weights)
        if not w.is_absolute() and not w.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            w = MODEL_DIR / w.name  # Ultralytics downloads the official weights here on first use
        self.model = YOLO(str(w))
        self.device = resolve_device(cfg.device)
        self.keep = set(cfg.vehicle_classes) | {cfg.person_class}

    def update(self, frame: FramePacket) -> Detections:
        results = self.model.track(
            frame.image,
            persist=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names
        tracks: list[TrackObs] = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, cls, conf, tid in zip(
                boxes.xyxy.tolist(),
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.id.tolist(),
                strict=True,
            ):
                label = names[int(cls)]
                if label in self.keep:
                    tracks.append(TrackObs(int(tid), label, float(conf), Box(*map(float, xyxy))))
        return Detections(t=frame.t, tracks=tracks)

    def reset(self) -> None:
        """Forget all tracks (call between independent videos)."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()
