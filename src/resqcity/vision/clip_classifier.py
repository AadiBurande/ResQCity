"""Stage 2 accident classifier on short windows.

Frame features: frozen ImageNet MobileNetV3-small (torchvision), global-pooled (576-d), computed
once per sampled frame and cached. Window features: [mean, max, mean |diff|] over the frames of a
``window_s`` window (motion enters through the frame-to-frame differences). Head: 2-layer MLP.

Training signal:
- windows of videos WITH accident onset/offset annotations get exact labels (window overlaps the
  accident interval -> 1, otherwise 0; frames after the accident ends are excluded);
- accident videos WITHOUT temporal annotations contribute through multiple-instance learning
  (Sultani et al., 2018): the mean of the top-k window scores of the video is pushed to 1;
- normal videos: every window is a negative.
Probabilities are calibrated with temperature scaling on the validation split.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resqcity.vision.config import ClipClassifierConfig

FEATURE_DIM = 576  # mobilenet_v3_small final channels
MIL_TOP_K = 3


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
    """Frozen MobileNetV3-small feature extractor (torch imported lazily)."""

    def __init__(self, cfg: ClipClassifierConfig, device: str = "cpu") -> None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        if cfg.backbone != "mobilenet_v3_small":
            raise ValueError(f"unsupported backbone {cfg.backbone!r}")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        self.net = torch.nn.Sequential(
            model.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )
        self.net.eval().to(device)
        self.device = device
        self.size = cfg.input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self._torch = torch

    def encode(self, images_bgr: Sequence[np.ndarray], batch: int = 64) -> np.ndarray:
        import cv2

        torch = self._torch
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
        return np.concatenate(feats) if feats else np.zeros((0, FEATURE_DIM), np.float32)


def build_head(cfg: ClipClassifierConfig, in_dim: int = 3 * FEATURE_DIM):  # noqa: ANN201 (torch type)
    import torch

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
    """Inference wrapper: encoder + head + temperature (loaded from a training run)."""

    def __init__(self, run_dir: Path, cfg: ClipClassifierConfig, device: str = "cpu") -> None:
        import json

        import torch

        meta = json.loads((run_dir / "clip_classifier.json").read_text(encoding="utf-8"))
        self.temperature = float(meta["temperature"])
        self.head = build_head(cfg)
        self.head.load_state_dict(torch.load(run_dir / "clip_head.pt", map_location=device))
        self.head.eval().to(device)
        self.encoder = FrameEncoder(cfg, device)
        self.device = device
        self._torch = torch

    def logit_from_features(self, frame_feats: np.ndarray) -> float:
        torch = self._torch
        x = torch.from_numpy(window_features(frame_feats)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return float(self.head(x).squeeze().cpu())

    def proba_from_features(self, frame_feats: np.ndarray) -> float:
        return float(sigmoid(self.logit_from_features(frame_feats) / self.temperature))

    def predict_window(self, frames: Sequence[np.ndarray]) -> float:
        return self.proba_from_features(self.encoder.encode(frames))
