"""Stage 2 accident classifier on short windows.

Two kinds of frozen backbone (``clip_classifier.backbone``):
- image: ``mobilenet_v3_small`` (ImageNet, 576-d) or ``siglip2_base`` (image-text, D-027: 768-d
  normalised embedding + cosine similarity to each of ``text_prompts``), cached per sampled frame;
  window feature = [mean, max, mean |diff|] over the window's frames;
- video (``r2plus1d_18``, Kinetics-400): the window's frames are resampled to ``clip_frames``,
  resized to ``input_size`` and embedded as ONE clip (512-d), so motion is modelled directly.
Head: 2-layer MLP.

Training signal:
- windows of videos WITH accident onset/offset annotations get exact labels (window overlaps the
  accident interval -> 1, otherwise 0; frames after the accident ends are excluded);
- accident videos WITHOUT temporal annotations contribute through multiple-instance learning
  (Sultani et al., 2018): the mean of the top-k window scores of the video is pushed to 1;
- normal videos: every window is a negative.
Probabilities are calibrated with temperature scaling on the validation split.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resqcity.vision.config import ClipClassifierConfig

FEATURE_DIM = 576  # mobilenet_v3_small final channels
IMAGE_BACKBONES = {"mobilenet_v3_small": FEATURE_DIM, "siglip2_base": 768}
TEXT_BACKBONES = {"siglip2_base"}  # image-text models: prompt similarities are appended
SIGLIP_TEXT_LENGTH = 64  # SigLIP2 text inputs are padded to this length (model card)
VIDEO_BACKBONES = {"r2plus1d_18": 512}  # backbone -> clip embedding size
MIL_TOP_K = 3
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
VIDEO_RESIZE_SHORT = 128  # Kinetics recipe: resize short side to 128, centre-crop input_size


def is_video_backbone(name: str) -> bool:
    return name in VIDEO_BACKBONES


def frame_dim(cfg: ClipClassifierConfig) -> int:
    """Per-frame feature size of an image backbone."""
    extra = len(cfg.text_prompts) if cfg.backbone in TEXT_BACKBONES else 0
    return IMAGE_BACKBONES[cfg.backbone] + extra


def head_input_dim(cfg: ClipClassifierConfig) -> int:
    if is_video_backbone(cfg.backbone):
        return VIDEO_BACKBONES[cfg.backbone]
    return 3 * frame_dim(cfg)


def feature_tag(cfg: ClipClassifierConfig) -> str:
    """Feature-cache folder: the backbone, plus a hash of the prompts when they are features (so a
    prompt change can never reuse stale features)."""
    if cfg.backbone in TEXT_BACKBONES:
        digest = hashlib.sha256("\n".join(cfg.text_prompts).encode("utf-8")).hexdigest()[:8]
        return f"{cfg.backbone}_{digest}"
    return cfg.backbone


def sample_indices(n: int, k: int) -> np.ndarray:
    """``k`` indices spread evenly over ``n`` frames (repeats when n < k)."""
    return np.linspace(0, n - 1, k).round().astype(int)


def resize_short(image_bgr: np.ndarray, short: int = VIDEO_RESIZE_SHORT) -> np.ndarray:
    import cv2

    h, w = image_bgr.shape[:2]
    scale = short / min(h, w)
    return cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def window_features(frame_feats: np.ndarray) -> np.ndarray:
    """(T, D) per-frame features -> (3D,) window feature."""
    if frame_feats.ndim != 2 or len(frame_feats) == 0:
        raise ValueError("expected (T, D) features with T >= 1")
    diff = (
        np.abs(np.diff(frame_feats, axis=0)).mean(axis=0)
        if len(frame_feats) > 1
        else np.zeros(frame_feats.shape[1])
    )
    return np.concatenate([frame_feats.mean(axis=0), frame_feats.max(axis=0), diff]).astype(
        np.float32
    )


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    first: int  # frame-feature row indices [first, last)
    last: int


def make_windows(times: np.ndarray, window_s: float, stride_s: float) -> list[Window]:
    """Windows ending every ``stride_s`` once ``window_s`` of frames is available."""
    if len(times) == 0:
        return []
    out = []
    end = times[0] + window_s
    while end <= times[-1] + 1e-9:
        first = int(np.searchsorted(times, end - window_s, side="left"))
        last = int(np.searchsorted(times, end, side="right"))
        if last - first >= 2:
            out.append(Window(float(end - window_s), float(end), first, last))
        end += stride_s
    return out


def window_label(w: Window, onset_s: float | None, offset_s: float | None) -> float | None:
    """1 if the window overlaps [onset, offset], 0 if entirely before onset, None if after offset
    (post-accident aftermath is ambiguous for detection and excluded) or unknown."""
    if onset_s is None:
        return None
    off = offset_s if offset_s is not None else onset_s
    if w.end_s < onset_s:
        return 0.0
    if w.start_s <= off:
        return 1.0
    return None


class FrameEncoder:
    """Frozen per-frame feature extractor: MobileNetV3-small or SigLIP2 (torch imported lazily)."""

    def __init__(self, cfg: ClipClassifierConfig, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dim = frame_dim(cfg) if cfg.backbone in IMAGE_BACKBONES else 0
        if cfg.backbone == "siglip2_base":
            self._init_siglip2(cfg)
            return
        if cfg.backbone != "mobilenet_v3_small":
            raise ValueError(f"unsupported backbone {cfg.backbone!r}")
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.siglip = None
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        self.net = torch.nn.Sequential(
            model.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )
        self.net.eval().to(device)
        self.size = cfg.input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _init_siglip2(self, cfg: ClipClassifierConfig) -> None:
        from transformers import AutoModel, AutoProcessor

        torch, w = self._torch, cfg.siglip2
        if w is None or not cfg.text_prompts:
            raise ValueError("siglip2_base needs clip_classifier.siglip2 and text_prompts")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(w.repo, revision=w.revision)
        model = AutoModel.from_pretrained(w.repo, revision=w.revision, dtype=self.dtype)
        self.siglip = model.eval().to(self.device)
        text = self.processor(
            text=list(cfg.text_prompts),
            padding="max_length",
            max_length=SIGLIP_TEXT_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            self.text = torch.nn.functional.normalize(self._features(text, "text"), dim=-1)

    def _features(self, inputs: object, kind: str):  # noqa: ANN202 (torch tensor)
        """Pooled embedding (transformers may return a tensor or an output object)."""
        fn = self.siglip.get_text_features if kind == "text" else self.siglip.get_image_features
        out = fn(**inputs)  # type: ignore[arg-type]
        out = getattr(out, "pooler_output", out)
        return out.float()

    def _encode_siglip(self, rgb: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            emb = torch.nn.functional.normalize(self._features(inputs, "image"), dim=-1)
            sims = emb @ self.text.T
        return torch.cat([emb, sims], dim=1).cpu().numpy()

    def encode(self, images_bgr: Sequence[np.ndarray], batch: int = 64) -> np.ndarray:
        import cv2

        torch = self._torch
        if self.siglip is not None:
            parts = [
                self._encode_siglip(
                    [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr[i : i + batch]]
                )
                for i in range(0, len(images_bgr), batch)
            ]
            return np.concatenate(parts) if parts else np.zeros((0, self.dim), np.float32)
        feats = []
        for i in range(0, len(images_bgr), batch):
            chunk = [
                cv2.resize(
                    cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                    (self.size, self.size),
                    interpolation=cv2.INTER_AREA,
                )
                for im in images_bgr[i : i + batch]
            ]
            x = (
                torch.from_numpy(np.stack(chunk)).to(self.device).permute(0, 3, 1, 2).float()
                / 255.0
            )
            x = (x - self.mean) / self.std
            with torch.inference_mode():
                feats.append(self.net(x).float().cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, self.dim), np.float32)


class ClipEncoder:
    """Frozen Kinetics-400 video backbone: a window of frames -> one clip embedding."""

    def __init__(self, cfg: ClipClassifierConfig, device: str = "cpu") -> None:
        import torch
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

        if cfg.backbone != "r2plus1d_18":
            raise ValueError(f"unsupported video backbone {cfg.backbone!r}")
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()
        self.net = model.eval().to(device)
        self.device = device
        self.crop = cfg.input_size
        self.k = cfg.clip_frames
        self.mean = torch.tensor(KINETICS_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(KINETICS_STD, device=device).view(1, 3, 1, 1, 1)
        self._torch = torch

    def _clip_array(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        chosen = [frames_bgr[i] for i in sample_indices(len(frames_bgr), self.k)]
        out = []
        for im in chosen:
            if min(im.shape[:2]) != VIDEO_RESIZE_SHORT:
                im = resize_short(im)
            h, w = im.shape[:2]
            y0, x0 = (h - self.crop) // 2, (w - self.crop) // 2
            crop = im[y0 : y0 + self.crop, x0 : x0 + self.crop]
            out.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return np.stack(out)  # (T, H, W, 3)

    def encode_clips(self, clips: Sequence[Sequence[np.ndarray]], batch: int = 16) -> np.ndarray:
        torch = self._torch
        feats = []
        for i in range(0, len(clips), batch):
            arr = np.stack([self._clip_array(c) for c in clips[i : i + batch]])  # (B,T,H,W,3)
            x = torch.from_numpy(arr).to(self.device).permute(0, 4, 1, 2, 3).float() / 255.0
            x = (x - self.mean) / self.std
            with torch.inference_mode():
                feats.append(self.net(x).float().cpu().numpy())
        dim = VIDEO_BACKBONES["r2plus1d_18"]
        return np.concatenate(feats) if feats else np.zeros((0, dim), np.float32)


def build_head(cfg: ClipClassifierConfig, in_dim: int | None = None):  # noqa: ANN201 (torch type)
    import torch

    in_dim = head_input_dim(cfg) if in_dim is None else in_dim
    return torch.nn.Sequential(
        torch.nn.LayerNorm(in_dim),
        torch.nn.Linear(in_dim, cfg.hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(cfg.dropout),
        torch.nn.Linear(cfg.hidden, 1),
    )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """Temperature T minimising the negative log-likelihood of sigmoid(logits / T) (grid search;
    robust and deterministic for one parameter)."""
    grid = np.exp(np.linspace(math.log(0.05), math.log(20.0), 400)) if grid is None else grid
    eps = 1e-7
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = np.clip(sigmoid(logits / t), eps, 1 - eps)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


class ClipClassifier:
    """Live inference wrapper for one camera: buffers frames (or frame features) for one window
    and scores it. The backbone and window settings come from the training run's metadata, so a
    run is always used with the exact settings it was trained with."""

    def __init__(self, run_dir: Path, cfg: ClipClassifierConfig, device: str = "cpu") -> None:
        import json
        from collections import deque

        import torch

        meta = json.loads((run_dir / "clip_classifier.json").read_text(encoding="utf-8"))
        cfg = cfg.model_copy(update=meta.get("clip_classifier", {}))
        self.cfg = cfg
        self.temperature = float(meta["temperature"])
        self.head = build_head(cfg)
        self.head.load_state_dict(torch.load(run_dir / "clip_head.pt", map_location=device))
        self.head.eval().to(device)
        self.video = is_video_backbone(cfg.backbone)
        self.encoder: FrameEncoder | ClipEncoder = (
            ClipEncoder(cfg, device) if self.video else FrameEncoder(cfg, device)
        )
        self.buffer: deque[tuple[float, np.ndarray]] = deque()
        self.device = device
        self._torch = torch

    def observe(self, t: float, image_bgr: np.ndarray) -> None:
        """Add one sampled frame (stored resized for video backbones, as features otherwise)."""
        if self.video:  # noqa: SIM108 (ClipEncoder has no per-frame encode)
            item = resize_short(image_bgr)
        else:
            item = self.encoder.encode([image_bgr])[0]  # type: ignore[union-attr]
        self.buffer.append((t, item))
        while self.buffer and self.buffer[0][0] < t - self.cfg.window_s:
            self.buffer.popleft()

    def ready(self) -> bool:
        return len(self.buffer) >= 2

    def logit_now(self) -> float:
        torch = self._torch
        items = [x for _, x in self.buffer]
        if self.video:
            feat = self.encoder.encode_clips([items])[0]  # type: ignore[union-attr]
        else:
            feat = window_features(np.stack(items))
        x = torch.from_numpy(np.asarray(feat, np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return float(self.head(x).squeeze().cpu())

    def proba_now(self) -> float:
        return float(sigmoid(self.logit_now() / self.temperature))
