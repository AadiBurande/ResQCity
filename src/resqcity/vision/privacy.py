"""Privacy by design: blur faces and number plates BEFORE a frame is stored, streamed or shown.

- Faces: OpenCV YuNet face detector (ONNX model, downloaded once to ``data/models``).
- Number plates: no reliable open plate detector is used; instead the lower band of every vehicle
  box (where plates are) is blurred (configurable band). This over-blurs rather than under-blurs.
- Persons: optionally blurred as a whole box (strong kernel), which covers faces the face model
  misses (small, side-on or blurred faces are common in CCTV).
Only event metadata leaves ``vision/`` (research-integrity rule 6).
"""

import logging
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from resqcity.common.config import PROJECT_ROOT
from resqcity.vision.config import PrivacyConfig
from resqcity.vision.types import Box, TrackObs

log = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MIN_KERNEL = 9


def _kernel(size: int, frac: float) -> int:
    k = max(MIN_KERNEL, int(size * frac))
    return k if k % 2 == 1 else k + 1


def _clip(box: Box, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(w, int(np.ceil(box.x2))), min(h, int(np.ceil(box.y2)))
    return x1, y1, x2, y2


def blur_region(image: np.ndarray, box: Box, frac: float) -> None:
    """Gaussian-blur ``box`` in place."""
    x1, y1, x2, y2 = _clip(box, image.shape)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = image[y1:y2, x1:x2]
    k = _kernel(max(x2 - x1, y2 - y1), frac)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def grow(box: Box, min_px: float) -> Box:
    """Grow a box (about its centre) so both sides are at least ``min_px``: a blur kernel cannot
    remove detail from a region only a few pixels wide."""
    cx, cy = box.center
    w, h = max(box.w, min_px), max(box.h, min_px)
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def plate_band(box: Box, band: tuple[float, float]) -> Box:
    lo, hi = band
    return Box(box.x1, box.y1 + lo * box.h, box.x2, box.y1 + hi * box.h)


class FaceDetector:
    """YuNet wrapper; returns face boxes. Falls back to no faces if the model is unavailable."""

    def __init__(self, cfg: PrivacyConfig, model_dir: Path = MODEL_DIR) -> None:
        self.cfg = cfg
        self.model_path = model_dir / cfg.face_model_file
        self._detector: cv2.FaceDetectorYN | None = None

    def _ensure(self, size: tuple[int, int]) -> cv2.FaceDetectorYN | None:
        if self._detector is None:
            if not self.model_path.exists():
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    urllib.request.urlretrieve(self.cfg.face_model_url, self.model_path)
                except OSError as exc:
                    log.error("face model download failed (%s); relying on person pixelation", exc)
                    return None
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path), "", size, self.cfg.face_score_threshold
            )
        self._detector.setInputSize(size)
        return self._detector

    def detect(self, image: np.ndarray) -> list[Box]:
        detector = self._ensure((image.shape[1], image.shape[0]))
        if detector is None:
            return []
        _, faces = detector.detect(image)
        if faces is None:
            return []
        return [
            Box(float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])) for f in faces
        ]


class Anonymizer:
    """Applies all privacy filters to a frame (returns a new, anonymised copy)."""

    def __init__(
        self,
        cfg: PrivacyConfig,
        vehicle_classes: Iterable[str],
        person_class: str,
        faces: FaceDetector | None = None,
    ) -> None:
        self.cfg = cfg
        self.vehicle_classes = set(vehicle_classes)
        self.person_class = person_class
        self.faces = faces if faces is not None else FaceDetector(cfg)

    def regions(self, image: np.ndarray, tracks: Iterable[TrackObs]) -> dict[str, list[Box]]:
        tracks = list(tracks)
        m = self.cfg.min_region_px
        return {
            "faces": [grow(b, m) for b in self.faces.detect(image)],
            "plates": [
                grow(plate_band(t.box, self.cfg.plate_band), m)
                for t in tracks
                if t.cls in self.vehicle_classes
            ],
            "persons": [grow(t.box, m) for t in tracks if t.cls == self.person_class]
            if self.cfg.blur_persons
            else [],
        }

    def apply(self, image: np.ndarray, tracks: Iterable[TrackObs]) -> np.ndarray:
        out = image.copy()
        regions = self.regions(out, tracks)
        for box in regions["persons"]:
            blur_region(out, box, self.cfg.person_blur_kernel_frac)
        for box in regions["faces"] + regions["plates"]:
            blur_region(out, box, self.cfg.blur_kernel_frac)
        return out
