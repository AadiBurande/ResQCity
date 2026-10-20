"""Typed models for ``configs/vision.yaml``."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from resqcity.common.config import CONFIG_DIR, ConfigModel, load_model


class IngestConfig(ConfigModel):
    sample_fps: float = Field(gt=0)
    reconnect_delay_s: float = Field(ge=0)
    max_reconnects: int = Field(ge=0)
    resize_width: int | None = Field(default=None, gt=0)


class DetectorConfig(ConfigModel):
    weights: str
    tracker: str
    device: str
    conf: float = Field(gt=0, lt=1)
    iou: float = Field(gt=0, lt=1)
    vehicle_classes: list[str] = Field(min_length=1)
    person_class: str


class PrivacyConfig(ConfigModel):
    face_model_url: str
    face_model_file: str
    face_score_threshold: float = Field(gt=0, lt=1)
    mode: Literal["faces_plates", "full", "off"]
    plate_band: tuple[float, float]
    plate_band_x: tuple[float, float]
    person_blur_kernel_frac: float = Field(gt=0, le=1)
    min_region_px: int = Field(ge=1)
    blur_kernel_frac: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _band(self) -> Self:
        for name in ("plate_band", "plate_band_x"):
            lo, hi = getattr(self, name)
            if not 0 <= lo < hi <= 1:
                raise ValueError(f"{name} must satisfy 0 <= low < high <= 1")
        return self


class HeuristicWeights(ConfigModel):
    deceleration: float = Field(ge=0)
    collision: float = Field(ge=0)
    orientation: float = Field(ge=0)
    person_down: float = Field(ge=0)

    def total(self) -> float:
        return self.deceleration + self.collision + self.orientation + self.person_down


class HeuristicsConfig(ConfigModel):
    window_s: float = Field(gt=0)
    min_track_frames: int = Field(ge=2)
    decel_drop_frac: float = Field(gt=0, lt=1)
    decel_window_s: float = Field(gt=0)
    moving_speed: float = Field(gt=0)
    stopped_speed: float = Field(ge=0)
    overlap_iou: float = Field(gt=0, lt=1)
    stop_after_overlap_s: float = Field(gt=0)
    aspect_flip_ratio: float = Field(gt=1)
    person_down_aspect: float = Field(gt=1)
    person_down_still_s: float = Field(gt=0)
    weights: HeuristicWeights

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.stopped_speed >= self.moving_speed:
            raise ValueError("stopped_speed must be below moving_speed")
        if self.weights.total() <= 0:
            raise ValueError("heuristic weights must not all be zero")
        return self


class ClipClassifierConfig(ConfigModel):
    backbone: str
    input_size: int = Field(gt=0)
    clip_frames: int = Field(gt=1)
    window_s: float = Field(gt=0)
    stride_s: float = Field(gt=0)
    hidden: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    calibration: Literal["temperature", "isotonic"]
    mil_weight: float = Field(ge=0)
    mil_sparsity: float = Field(ge=0)
    mil_smoothness: float = Field(ge=0)


class FireSmokeConfig(ConfigModel):
    base_weights: str
    epochs: int = Field(gt=0)
    imgsz: int = Field(gt=0)
    batch: int = Field(gt=0)
    conf: float = Field(gt=0, lt=1)
    persistence_frames: int = Field(ge=1)
    workers: int = Field(ge=0)


class AlarmConfig(ConfigModel):
    threshold: float = Field(gt=0, lt=1)
    consecutive_windows: int = Field(ge=1)
    cooldown_s: float = Field(ge=0)


class SplitConfig(ConfigModel):
    train: float = Field(gt=0)
    val: float = Field(gt=0)
    test: float = Field(gt=0)
    seed: int
    # Accident videos with annotated onsets are scarce (UCF-Crime: 23) and are the only ones that
    # measure detection delay: keep them for val/test; training uses weak (MIL) labels instead.
    onset_videos_eval_only: bool = True

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class UcfCrimeConfig(ConfigModel):
    base_url: str
    accident_zip: str
    accident_prefix: str
    normal_zip: str
    normal_max_videos: int = Field(ge=0)
    temporal_annotations: str
    licence: str


class CadpConfig(ConfigModel):
    annotations_url: str
    frames_gdrive_id: str
    licence: str


class DFireConfig(ConfigModel):
    hf_repo: str
    licence: str


class DatasetsConfig(ConfigModel):
    cap_gb: float = Field(gt=0)
    split: SplitConfig
    ucf_crime: UcfCrimeConfig
    cadp: CadpConfig
    dfire: DFireConfig


class VisionConfig(ConfigModel):
    ingest: IngestConfig
    detector: DetectorConfig
    privacy: PrivacyConfig
    heuristics: HeuristicsConfig
    clip_classifier: ClipClassifierConfig
    fire_smoke: FireSmokeConfig
    alarm: AlarmConfig
    datasets: DatasetsConfig


def load_vision_config(path: Path | None = None) -> VisionConfig:
    return load_model(VisionConfig, path or CONFIG_DIR / "vision.yaml")
