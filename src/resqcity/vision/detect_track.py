"""YOLO detection + ByteTrack tracking (Ultralytics, AGPL-3.0; accepted in D-019).

Imports of torch/ultralytics are lazy so the rest of the pipeline (and the tests) work without them.
"""

from dataclasses import replace
from pathlib import Path

from resqcity.common.config import PROJECT_ROOT
from resqcity.vision.config import DetectorConfig, UvhDetectorConfig
from resqcity.vision.types import Box, Detections, FramePacket, TrackObs

MODEL_DIR = PROJECT_ROOT / "data" / "models"
PERSON_ID_OFFSET = 1_000_000  # keeps person track ids apart from vehicle ids (DualTracker)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloTracker:
    """Stateful tracker for ONE stream (ByteTrack keeps track ids across ``update`` calls)."""

    def __init__(
        self,
        cfg: DetectorConfig,
        weights: str | Path | None = None,
        class_map: dict[str, str] | None = None,
        keep: set[str] | None = None,
    ) -> None:
        """``class_map`` renames model classes to our vocabulary; ``keep`` limits the output
        classes (default: the configured vehicle classes and people)."""
        from ultralytics import YOLO

        self.cfg = cfg
        w = Path(weights or cfg.weights)
        if not w.is_absolute() and not w.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            w = MODEL_DIR / w.name  # Ultralytics downloads the official weights here on first use
        self.model = YOLO(str(w))
        self.device = resolve_device(cfg.device)
        self.class_map = class_map or {}
        self.keep = keep if keep is not None else set(cfg.vehicle_classes) | {cfg.person_class}

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
                label = self.class_map.get(names[int(cls)], names[int(cls)])
                if label in self.keep:
                    tracks.append(TrackObs(int(tid), label, float(conf), Box(*map(float, xyxy))))
        return Detections(t=frame.t, tracks=tracks)

    def reset(self) -> None:
        """Forget all tracks (call between independent videos)."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()


def uvh26_weights(cfg: UvhDetectorConfig) -> Path:
    """UVH-26 YOLO11 weights from the pinned Hugging Face revision (downloaded once)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(cfg.repo, cfg.weights_file, revision=cfg.revision))


def uvh26_vehicle_classes(cfg: DetectorConfig) -> set[str]:
    if cfg.uvh26 is None:
        raise ValueError("detector.uvh26 is not configured")
    return set(cfg.uvh26.class_map.values())


class DualTracker:
    """Vehicles from the UVH-26 model (Indian classes incl. auto-rickshaw; D-027) and people from
    the COCO model (UVH-26 has no person class). Each keeps its own ByteTrack state."""

    def __init__(self, cfg: DetectorConfig) -> None:
        if cfg.uvh26 is None:
            raise ValueError("detector.uvh26 is not configured")
        self.vehicles = YoloTracker(
            cfg,
            uvh26_weights(cfg.uvh26),
            class_map=cfg.uvh26.class_map,
            keep=uvh26_vehicle_classes(cfg),
        )
        self.persons = YoloTracker(cfg, keep={cfg.person_class})

    def update(self, frame: FramePacket) -> Detections:
        vehicles = self.vehicles.update(frame).tracks
        persons = [
            replace(o, track_id=o.track_id + PERSON_ID_OFFSET)
            for o in self.persons.update(frame).tracks
        ]
        return Detections(t=frame.t, tracks=vehicles + persons)

    def reset(self) -> None:
        self.vehicles.reset()
        self.persons.reset()
