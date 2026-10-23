"""Second-stage alarm verifier: a small offline vision-language model (phase 3b, D-027).

When the detector raises a candidate alarm, ``frames`` anonymised frames from the last ``span_s``
seconds are shown to the model with the pre-registered yes/no prompt. The verifier score is the
probability of "yes" against "no" for the first answer token (thinking disabled), so no free text
has to be parsed. ``describe`` returns a one-sentence description for the operator (optional
``Event`` metadata). Everything runs locally; frames never leave ``vision/``.

Backends: Hugging Face image-text-to-text models (Qwen3.5, Qwen3-VL, Gemma 4, TAU-R1, and
Molmo2 with its own code) loaded in 4/8-bit; a candidate may bring its own prompts and answers.
"""

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from resqcity.vision.config import VerifierCandidate, VerifierConfig
from resqcity.vision.privacy import Anonymizer
from resqcity.vision.types import FramePacket

log = logging.getLogger(__name__)
DESCRIBE_TOKENS = 48
IMAGES = {".jpg", ".jpeg", ".png"}


def resize_long(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def clip_frames(
    path: Path,
    t_end: float,
    cfg: VerifierConfig,
    anonymizer: Anonymizer | None = None,
    tracker: object | None = None,
    fps: float | None = None,
) -> list[np.ndarray]:
    """``cfg.frames`` BGR frames evenly spaced in [t_end - span_s, t_end] of a video file or of
    a folder of frames (``fps`` = rate of the stored frames), anonymised (``tracker`` supplies
    the vehicle/person boxes the anonymiser needs)."""
    folder = (
        sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGES) if path.is_dir() else []
    )
    cap = None if path.is_dir() else cv2.VideoCapture(str(path))
    out = []
    try:
        for t in np.linspace(max(0.0, t_end - cfg.span_s), t_end, cfg.frames):
            if cap is None:
                i = min(len(folder) - 1, round(float(t) * (fps or 1.0)))
                image = cv2.imread(str(folder[i])) if folder else None
                ok = image is not None
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ok, image = cap.read()
            if not ok:
                continue
            if anonymizer is not None:
                tracks = (
                    tracker.update(FramePacket(t=float(t), image=image)).tracks if tracker else []
                )  # type: ignore[attr-defined]
                image = anonymizer.apply(image, tracks)
            out.append(resize_long(image, cfg.frame_long_side))
    finally:
        if cap is not None:
            cap.release()
    return out


class HfVerifier:
    """Image-text-to-text model from the Hugging Face hub, pinned revision, quantised."""

    def __init__(
        self, cand: VerifierCandidate, cfg: VerifierConfig, device: str = "cuda:0"
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.cfg, self.device, self._torch = cfg, device, torch
        self.cand = cand
        quant = None
        if cand.quantization == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        elif cand.quantization == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        remote = {"trust_remote_code": True} if cand.remote_code else {}
        self.processor = AutoProcessor.from_pretrained(cand.repo, revision=cand.revision, **remote)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cand.repo,
            revision=cand.revision,
            dtype=torch.float16,
            quantization_config=quant,
            device_map={"": device},
            **remote,
        ).eval()
        tok = self.processor.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in cand.positive})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in cand.negative})

    def _inputs(self, frames_bgr: list[np.ndarray], prompt: str):  # noqa: ANN202 (BatchFeature)
        from PIL import Image

        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if self.cand.system_prompt:
            system = [{"type": "text", "text": self.cand.system_prompt}]
            messages.insert(0, {"role": "system", "content": system})
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

    def score(self, frames_bgr: list[np.ndarray]) -> float:
        """P(positive answer) vs P(negative answers) for the first answer token (default yes/no;
        TAU-R1: label C vs A, B, D)."""
        torch = self._torch
        if not frames_bgr:
            return 0.0
        inputs = self._inputs(frames_bgr, self.cand.prompt or self.cfg.prompt)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        lse_yes = torch.logsumexp(logits[self.yes], 0)
        lse_no = torch.logsumexp(logits[self.no], 0)
        return float(1.0 / (1.0 + math.exp(float(lse_no - lse_yes))))

    def describe(self, frames_bgr: list[np.ndarray]) -> str:
        torch = self._torch
        inputs = self._inputs(frames_bgr, self.cfg.describe_prompt)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=DESCRIBE_TOKENS, do_sample=False)
        new = ids[0, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()


def load_verifier(name: str, cfg: VerifierConfig, device: str = "cuda:0") -> HfVerifier:
    cand = cfg.candidates[name]
    t0 = time.perf_counter()
    verifier = HfVerifier(cand, cfg, device)
    log.info("verifier %s loaded in %.1f s", name, time.perf_counter() - t0)
    return verifier
