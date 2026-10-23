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


class HfWeights(ConfigModel):
    repo: str
    revision: str = Field(min_length=40, max_length=40)


class UvhDetectorConfig(HfWeights):
    weights_file: str
    class_map: dict[str, str] = Field(min_length=1)


class DetectorConfig(ConfigModel):
    weights: str
    tracker: str
    device: str
    conf: float = Field(gt=0, lt=1)
    iou: float = Field(gt=0, lt=1)
    vehicle_classes: list[str] = Field(min_length=1)
    person_class: str
    uvh26: UvhDetectorConfig | None = None


class SimulatedStreams(ConfigModel):
    rtsp_base: str
    cameras: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _loopback(self) -> Self:
        from urllib.parse import urlparse

        if urlparse(self.rtsp_base).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("unblurred_streams.rtsp_base must be a loopback address")
        return self

    def urls(self) -> set[str]:
        return {f"{self.rtsp_base.rstrip('/')}/{c}" for c in self.cameras}


class PrivacyConfig(ConfigModel):
    face_model_url: str
    face_model_file: str
    face_score_threshold: float = Field(gt=0, lt=1)
    mode: Literal["faces_plates", "full", "off"]
    unblurred_local_dirs: list[str] = Field(min_length=1)
    unblurred_streams: SimulatedStreams
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
    siglip2: HfWeights | None = None
    vjepa2: HfWeights | None = None
    text_prompts: list[str] = Field(default_factory=list)
    labels: Literal["mil", "interval"] = "mil"
    sources: list[str] = Field(default_factory=lambda: ["ucf_crime"], min_length=1)
    train_sources: list[str] = Field(default_factory=lambda: ["ucf_crime"], min_length=1)
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


class VerifierCandidate(HfWeights):
    quantization: Literal["4bit", "8bit", "none"]
    remote_code: bool = False  # model code shipped with the weights (trust_remote_code)
    processor: HfWeights | None = None  # load the processor from another repo (default: repo)
    cpu_modules: list[str] = Field(default_factory=list)  # kept in system RAM (e.g. embeddings)
    system_prompt: str | None = None  # candidate-specific prompts (default: verifier.prompt)
    prompt: str | None = None
    positive: list[str] = Field(default_factory=lambda: ["yes", "Yes", " yes", " Yes", "YES"])
    negative: list[str] = Field(default_factory=lambda: ["no", "No", " no", " No", "NO"])


class VerifierConfig(ConfigModel):
    prompt: str
    describe_prompt: str
    frames: int = Field(ge=1)
    span_s: float = Field(gt=0)
    frame_long_side: int = Field(gt=0)
    min_detection_retention: float = Field(gt=0, le=1)
    max_latency_s: float = Field(gt=0)
    thresholds: list[float] = Field(min_length=1)
    candidates: dict[str, VerifierCandidate] = Field(min_length=1)


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
    exclude_videos: list[str] = Field(default_factory=list)
    temporal_annotations: str
    licence: str


class CadpConfig(ConfigModel):
    annotations_url: str
    frames_gdrive_id: str
    licence: str


class DFireConfig(ConfigModel):
    hf_repo: str
    licence: str


class AccidentConfig(ConfigModel):
    kaggle_dataset: str
    files_prefix: str
    metadata: list[str] = Field(min_length=1)
    licence: str
    val_fraction: float = Field(gt=0, lt=1)
    split_seed: int = Field(ge=0)
    offset_after_impact_s: float = Field(ge=0)


class UcaConfig(ConfigModel):
    repo: str
    commit: str = Field(min_length=40, max_length=40)
    files: list[str] = Field(min_length=1)
    licence: str
    accident_keywords: list[str] = Field(min_length=1)


class UaDetracConfig(ConfigModel):
    hf_repo: str
    zips: list[str] = Field(min_length=1)
    source_fps: float = Field(gt=0)
    keep_every: int = Field(ge=1)
    licence: str


class DatasetsConfig(ConfigModel):
    cap_gb: float = Field(gt=0)
    split: SplitConfig
    ucf_crime: UcfCrimeConfig
    cadp: CadpConfig
    accident: AccidentConfig
    uca: UcaConfig
    ua_detrac: UaDetracConfig | None = None
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
    verifier: VerifierConfig | None = None


def load_vision_config(path: Path | None = None) -> VisionConfig:
    return load_model(VisionConfig, path or CONFIG_DIR / "vision.yaml")
