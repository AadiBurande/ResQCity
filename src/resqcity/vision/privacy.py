"""Privacy by design: blur faces and number plates BEFORE a frame is stored, streamed or shown.

Modes (``privacy.mode``, D-026):
- ``faces_plates`` (default): faces (OpenCV YuNet, ONNX model downloaded once to ``data/models``)
  and the plate region of every vehicle (no reliable open plate detector exists, so the lower-centre
  part of the vehicle box is blurred). Vehicles, people's bodies and the scene stay clear.
- ``full``: additionally blurs every person box (covers faces the face model misses).
- ``off``: no blurring; allowed only for public-dataset footage (:func:`unblurred_allowed`),
  enforced when a ``CameraPipeline`` is built (:func:`require_source_allowed`).
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


def unblurred_allowed(source: str, cfg: PrivacyConfig) -> bool:
    """Whether mode ``off`` may be used for ``source`` (D-026): a local file under one of
    ``cfg.unblurred_local_dirs`` (public-dataset footage), or a simulated camera
    (``cfg.unblurred_streams``: loopback base + simulated camera id). A phone or IP camera, any
    other file, or a relay under another name is refused. Footage deliberately copied into those
    folders or published under a simulated camera's name cannot be detected."""
    from resqcity.vision.ingest import is_live

    if not is_live(source):
        path = Path(source).resolve()
        return any(
            path.is_relative_to((PROJECT_ROOT / d).resolve()) for d in cfg.unblurred_local_dirs
        )
    return source.rstrip("/") in cfg.unblurred_streams.urls()


def require_source_allowed(cfg: PrivacyConfig, source: str | None) -> None:
    """Raise unless the privacy mode may be used for ``source`` (mode ``off`` needs a source that
    passes :func:`unblurred_allowed`)."""
    if cfg.mode == "off" and (source is None or not unblurred_allowed(source, cfg)):
        raise PermissionError(
            f"privacy mode 'off' is only for public-dataset footage (files under "
            f"{cfg.unblurred_local_dirs} or the simulated cameras); refused for {source!r}"
        )


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


def plate_band(
    box: Box, band: tuple[float, float], band_x: tuple[float, float] = (0.0, 1.0)
) -> Box:
    lo, hi = band
    xlo, xhi = band_x
    return Box(box.x1 + xlo * box.w, box.y1 + lo * box.h, box.x1 + xhi * box.w, box.y1 + hi * box.h)


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
        if self.cfg.mode == "off":
            return {"faces": [], "plates": [], "persons": []}
        tracks = list(tracks)
        m = self.cfg.min_region_px
        return {
            "faces": [grow(b, m) for b in self.faces.detect(image)],
            "plates": [
                grow(plate_band(t.box, self.cfg.plate_band, self.cfg.plate_band_x), m)
                for t in tracks
                if t.cls in self.vehicle_classes
            ],
            "persons": [grow(t.box, m) for t in tracks if t.cls == self.person_class]
            if self.cfg.mode == "full"
            else [],
        }

    def apply(self, image: np.ndarray, tracks: Iterable[TrackObs]) -> np.ndarray:
        out = image.copy()
        if self.cfg.mode == "off":
            return out
        regions = self.regions(out, tracks)
        for box in regions["persons"]:
            blur_region(out, box, self.cfg.person_blur_kernel_frac)
        for box in regions["faces"] + regions["plates"]:
            blur_region(out, box, self.cfg.blur_kernel_frac)
        return out
