"""Fine-tune the alarm verifier on TRAINING clips (phase 3b amendment 3, D-027; one attempt).

    uv run python -m resqcity.vision.train_verifier

Examples come from the train split only (asserted): ACCIDENT train clips, UCF-Crime train
accident videos with UCA intervals, and UCF-Crime train normal videos. Each example is an
alarm-like end time t; its input is exactly what the verifier sees at inference (frames of
[t - span_s, t], anonymised, the pre-registered prompt) and its target is the first answer
token "yes" (t in [t* + 0.5 s, t* + 8 s]) or "no" (t before t* - 1 s, normal footage, or a
false alarm of the deployed detector on a training video: hard negatives). Classes are
balanced 1:1 by seeded sampling. QLoRA settings: ``verifier.finetune`` in configs/vision.yaml.

Outputs: the LoRA adapter in ``data/models/verifier/<name>/`` and
``results/vision/v5/verifier_ft/`` (examples.csv, train_log.csv, meta).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.alarm import alarm_times
from resqcity.vision.config import VisionConfig, load_vision_config
from resqcity.vision.eval.evaluate import add_scores, alarm_cfg
from resqcity.vision.eval.select_v5 import SWEEPS, run_config, validate
from resqcity.vision.scoring import logit

log = logging.getLogger(__name__)
TRAIN = "train"


def _uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> list[float]:
    return [float(x) for x in rng.uniform(lo, hi, n)] if hi > lo else []


def hard_negatives(cache: Path, ann: pd.DataFrame, intervals: pd.DataFrame) -> list[dict]:
    """False alarms of the deployed (stage C) detector on TRAIN videos."""
    choice = json.loads((PROJECT_ROOT / "results/vision/v5/stage_c.json").read_text("utf-8"))
    run, tag = choice["chosen_run"], choice["detector_tag"]
    op = choice["operating_point"]
    cfg = run_config(run)
    _, _, stacker = validate(run, cache, ann, cfg, tag)  # fitted on validation, as deployed
    w = pd.read_parquet(cache / f"windows_{run}.parquet")
    w = add_scores(w[w["split"].eq(TRAIN)], cache, tag)
    w["clf_logit"] = logit(w["clf_p"])
    w["combined_p"] = stacker.proba(w)
    onset = dict(zip(intervals["video_id"], intervals["train_onset_s"], strict=True))
    out = []
    for vid, g in w.groupby("video_id"):
        v = ann.loc[ann["video_id"].eq(vid)].iloc[0]
        if v["label"] == "normal":
            limit = np.inf
        elif v["has_onset"]:
            limit = float(v["accident_onset_s"]) - 1.0
        elif vid in onset:
            limit = float(onset[vid]) - 1.0
        else:
            continue  # weak video without any interval: alarm timing cannot be judged
        g = g.sort_values("end_s")
        for t in alarm_times(
            g["end_s"].tolist(),
            g["combined_p"].tolist(),
            alarm_cfg(cfg, op["consecutive_windows"]),
            op["threshold"],
        ):
            if t < limit:
                out.append({"video_id": vid, "t_end": t, "label": 0, "origin": "hard_negative"})
    return out


def build_examples(cfg: VisionConfig, cache: Path) -> pd.DataFrame:
    ft = cfg.verifier.finetune  # type: ignore[union-attr]
    rng = np.random.default_rng(ft.seed)
    ann = pd.read_parquet(cache / "annotations.parquet")
    intervals = pd.read_parquet(cache / "uca_intervals.parquet")
    train = ann[ann["split"].eq(TRAIN) & ann["source"].isin(["accident", "ucf_crime"])]
    onset = intervals.set_index("video_id")
    lo, hi = ft.positive_span_s
    rows: list[dict] = []
    for _, v in train.iterrows():
        dur = float(v["duration_s"])
        if v["label"] == "normal":
            for t in _uniform(rng, cfg.verifier.span_s, dur, ft.normal_negatives_per_video):  # type: ignore[union-attr]
                rows.append({"video_id": v["video_id"], "t_end": t, "label": 0, "origin": "normal"})
            continue
        if v["has_onset"]:
            t0 = float(v["accident_onset_s"])
        elif v["video_id"] in onset.index:
            t0 = float(onset.loc[v["video_id"], "train_onset_s"])
        else:
            continue
        for t in _uniform(rng, t0 + lo, min(t0 + hi, dur), ft.max_positives_per_clip):
            rows.append({"video_id": v["video_id"], "t_end": t, "label": 1, "origin": "crash"})
        for t in _uniform(rng, 1.0, t0 - ft.negative_margin_s, 1):
            rows.append({"video_id": v["video_id"], "t_end": t, "label": 0, "origin": "pre_crash"})
    rows += hard_negatives(cache, ann, intervals)
    ex = pd.DataFrame(rows)
    pos, neg = ex[ex["label"].eq(1)], ex[ex["label"].eq(0)]
    n = min(len(pos), len(neg))
    ex = pd.concat(
        [pos.sample(n, random_state=ft.seed), neg.sample(n, random_state=ft.seed)],
        ignore_index=True,
    )
    meta = ann.set_index("video_id").loc[ex["video_id"], ["split", "path", "fps"]]
    if not meta["split"].eq(TRAIN).all():
        raise RuntimeError("verifier training examples must come from the train split only")
    ex["path"], ex["fps"] = meta["path"].to_numpy(), meta["fps"].to_numpy()
    ex = ex.sample(frac=1.0, random_state=ft.seed).reset_index(drop=True)
    ex["example_id"] = [f"ex{i:05d}" for i in range(len(ex))]
    return ex


def cache_frames(ex: pd.DataFrame, cfg: VisionConfig, raw: Path, out: Path) -> None:
    """Anonymised verifier frames per example, saved as JPEG (computed once)."""
    from resqcity.vision.detect_track import YoloTracker
    from resqcity.vision.privacy import Anonymizer
    from resqcity.vision.verifier import clip_frames

    d = cfg.detector
    anon, tracker = Anonymizer(cfg.privacy, d.vehicle_classes, d.person_class), YoloTracker(d)
    for i, r in enumerate(ex.itertuples(), start=1):
        folder = out / r.example_id
        if folder.exists() and any(folder.iterdir()):
            continue
        tracker.reset()
        frames = clip_frames(raw / r.path, r.t_end, cfg.verifier, anon, tracker, fps=float(r.fps))  # type: ignore[arg-type]
        folder.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(frames):
            cv2.imwrite(str(folder / f"f{j}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 100 == 0:
            log.info("frames: %d/%d examples", i, len(ex))


def train(ex: pd.DataFrame, cfg: VisionConfig, frames_dir: Path, adapter_dir: Path) -> pd.DataFrame:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    v = cfg.verifier
    ft = v.finetune  # type: ignore[union-attr]
    base = v.candidates[ft.base]  # type: ignore[union-attr]
    torch.manual_seed(ft.seed)
    processor = AutoProcessor.from_pretrained(base.repo, revision=base.revision)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base.repo,
        revision=base.revision,
        dtype=torch.bfloat16,
        quantization_config=quant,
        device_map={"": "cuda:0"},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=ft.lora_rank,
            lora_alpha=ft.lora_alpha,
            lora_dropout=ft.lora_dropout,
            target_modules=ft.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    tok = processor.tokenizer
    answer = {1: tok.encode("yes", add_special_tokens=False)[0],
              0: tok.encode("no", add_special_tokens=False)[0]}  # fmt: skip
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=ft.lr)
    rng = np.random.default_rng(ft.seed)
    log_rows, step = [], 0
    model.train()
    for epoch in range(ft.epochs):
        order = rng.permutation(len(ex))
        losses, t0 = [], time.time()
        for n, i in enumerate(order, start=1):
            r = ex.iloc[i]
            from PIL import Image

            images = [
                Image.open(p).convert("RGB")
                for p in sorted((frames_dir / r["example_id"]).glob("*.jpg"))
            ]
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": v.prompt})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("cuda:0")
            logits = model(**inputs).logits[0, -1].float()
            target = torch.tensor(answer[int(r["label"])], device="cuda:0")
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            (loss / ft.grad_accum).backward()
            losses.append(float(loss.detach()))
            if n % ft.grad_accum == 0 or n == len(order):
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if n % 100 == 0:
                log.info("epoch %d: %d/%d, mean loss %.4f, %.1f s/example", epoch, n, len(order),
                         float(np.mean(losses[-100:])), (time.time() - t0) / n)  # fmt: skip
        log_rows.append(
            {"epoch": epoch, "mean_loss": float(np.mean(losses)), "optimizer_steps": step}
        )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    return pd.DataFrame(log_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prepare-only", action="store_true", help="build examples and cache frames, no training"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    ft = cfg.verifier.finetune  # type: ignore[union-attr]
    cache = base.paths.resolve("data_processed") / "vision"
    out = base.paths.resolve("results") / "vision" / "v5" / "verifier_ft"
    out.mkdir(parents=True, exist_ok=True)
    ex = build_examples(cfg, cache)
    ex.to_csv(out / "examples.csv", index=False, lineterminator="\n")
    log.info("examples: %s", ex.groupby(["label", "origin"]).size().to_dict())
    frames_dir = cache / "verifier_train"
    cache_frames(ex, cfg, base.paths.resolve("data_raw") / "vision", frames_dir)
    if args.prepare_only:
        return 0
    adapter_dir = PROJECT_ROOT / cfg.verifier.candidates[ft.name].adapter  # type: ignore[union-attr,operator]
    log_df = train(ex, cfg, frames_dir, adapter_dir)
    log_df.to_csv(out / "train_log.csv", index=False, lineterminator="\n")
    meta = {
        "examples": len(ex),
        "by_origin": ex.groupby("origin").size().to_dict(),
        **ft.model_dump(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    configs = [CONFIG_DIR / "vision.yaml", SWEEPS / "v5_vjepa_mil_all.yaml"]
    for f in (out / "examples.csv", out / "train_log.csv", out / "meta.json"):
        write_meta(f, configs=configs, synthetic=False, seeds={"seed": ft.seed})
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
