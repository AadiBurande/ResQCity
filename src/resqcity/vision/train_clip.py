"""Train the stage-2 window classifier (GPU if available); early stopping and calibration on the
FIT split (``val`` by default; ``test`` for the second cross-fitting fold, D-026).

    uv run python -m resqcity.vision.train_clip [--run-id NAME] [--overlay sweep.yaml]
                                                 [--fit-split val|test]

Outputs:
- ``data/models/clip/<run_id>/clip_head.pt`` + ``clip_classifier.json`` (temperature, weight hash);
- ``results/vision/<run_id>/config.yaml`` (hyper-parameters), ``train_log.csv``,
  ``windows.parquet`` (classifier logit/probability for every window of every video, all splits).
Seeds are fixed; splits are by video (``data/processed/vision/annotations.parquet``).
"""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config, load_model
from resqcity.common.provenance import write_meta
from resqcity.vision.clip_classifier import (
    MIL_TOP_K,
    ClipEncoder,
    FrameEncoder,
    Window,
    build_head,
    fit_temperature,
    is_video_backbone,
    make_windows,
    sigmoid,
    window_features,
    window_label,
)
from resqcity.vision.config import VisionConfig
from resqcity.vision.offline import (
    extract_features,
    extract_window_features,
    feature_path,
    load_features,
    load_window_features,
)

log = logging.getLogger(__name__)
SEED = 20261020
PATIENCE = 8


def build_windows(
    ann: pd.DataFrame, cfg: VisionConfig, cache: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per window (all videos) + the matching feature matrix."""
    cc = cfg.clip_classifier
    video = is_video_backbone(cc.backbone)
    rows, feats = [], []
    for _, v in ann.iterrows():
        path = feature_path(cache, cc.backbone, v["video_id"])
        if video:
            starts, ends, wf = load_window_features(path)
            windows = [
                Window(float(s), float(e), i, i + 1)
                for i, (s, e) in enumerate(zip(starts, ends, strict=True))
            ]
        else:
            times, f = load_features(path)
            windows = make_windows(times, cc.window_s, cc.stride_s)
        onset = v["accident_onset_s"] if v["has_onset"] else None
        offset = v["accident_offset_s"] if v["has_onset"] else None
        for w in windows:
            if v["label"] == "normal":
                label = 0.0
            elif v["has_onset"]:
                label = window_label(w, onset, offset)
            else:
                label = None  # weak (MIL) bag member
            rows.append(
                {
                    "video_id": v["video_id"],
                    "source": v["source"],
                    "split": v["split"],
                    "video_label": v["label"],
                    "start_s": w.start_s,
                    "end_s": w.end_s,
                    "label": np.nan if label is None else label,
                }
            )
            feats.append(wf[w.first] if video else window_features(f[w.first : w.last]))
    return pd.DataFrame(rows), np.stack(feats).astype(np.float32)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def train(
    windows: pd.DataFrame, x: np.ndarray, cfg: VisionConfig, device: str, fit_split: str = "val"
) -> tuple[object, pd.DataFrame]:
    import torch

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cc = cfg.clip_classifier
    head = build_head(cc, x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)

    tr = windows["split"].eq("train")
    sup = (tr & windows["label"].notna()).to_numpy()
    y_sup = windows.loc[sup, "label"].to_numpy(np.float32)
    pos_weight = torch.tensor([(y_sup == 0).sum() / max(1, (y_sup == 1).sum())], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bags = [
        np.flatnonzero((windows["video_id"] == vid).to_numpy())
        for vid in windows.loc[
            tr & windows["label"].isna() & windows["video_label"].eq("accident"), "video_id"
        ].unique()
    ]
    val = (windows["split"].eq(fit_split) & windows["label"].notna()).to_numpy()
    # Model selection: discrimination WITHIN onset-annotated accident videos (pre-onset vs
    # accident windows of the same scene). Overall AUROC is dominated by scene type (D-021).
    val_within = val & windows["video_label"].eq("accident").to_numpy()
    xt = torch.from_numpy(x).to(device)
    sup_idx = np.flatnonzero(sup)

    best_auc, best_state, stale, log_rows = -1.0, None, 0, []
    for epoch in range(cc.epochs):
        head.train()
        order = rng.permutation(sup_idx)
        losses = []
        for i in range(0, len(order), cc.batch_size):
            idx = order[i : i + cc.batch_size]
            yb = torch.from_numpy(windows["label"].to_numpy(np.float32)[idx]).to(device)
            loss = bce(head(xt[idx]).squeeze(1), yb)
            if bags:  # MIL (Sultani et al. 2018): top-k of weak accident videos -> positive,
                # plus sparsity (accidents are brief) and temporal smoothness within the video
                chosen = rng.choice(len(bags), size=min(4, len(bags)), replace=False)
                bag_logits, sparsity, smooth = [], [], []
                for b in chosen:
                    logits = head(xt[bags[b]]).squeeze(1)
                    k = min(MIL_TOP_K, len(logits))
                    bag_logits.append(torch.topk(logits, k).values.mean())
                    probs = torch.sigmoid(logits)
                    sparsity.append(probs.mean())
                    if len(probs) > 1:
                        smooth.append(((probs[1:] - probs[:-1]) ** 2).mean())
                mil = torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.stack(bag_logits), torch.ones(len(bag_logits), device=device)
                )
                loss = loss + cc.mil_weight * mil + cc.mil_sparsity * torch.stack(sparsity).mean()
                if smooth:
                    loss = loss + cc.mil_smoothness * torch.stack(smooth).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.inference_mode():
            val_logits = head(xt[val]).squeeze(1).cpu().numpy()
            within_logits = head(xt[val_within]).squeeze(1).cpu().numpy()
        val_all = auroc(windows.loc[val, "label"].to_numpy(), val_logits)
        val_auc = auroc(windows.loc[val_within, "label"].to_numpy(), within_logits)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_auroc_all": val_all,
                "val_auroc_within": val_auc,
            }
        )
        log.info(
            "epoch %d loss %.4f val AUROC all %.4f within %.4f",
            epoch,
            np.mean(losses),
            val_all,
            val_auc,
        )
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(best_state)
    return head, pd.DataFrame(log_rows)


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--overlay", type=Path, default=None, help="YAML merged over configs/vision.yaml (sweeps)"
    )
    parser.add_argument("--fit-split", choices=["val", "test"], default="val")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    vision_paths = [CONFIG_DIR / "vision.yaml", *([args.overlay] if args.overlay else [])]
    base, cfg = load_base_config(), load_model(VisionConfig, *vision_paths)
    run_id = args.run_id or datetime.now(UTC).strftime("clip_%Y%m%dT%H%M")
    raw = base.paths.resolve("data_raw") / "vision"
    cache = base.paths.resolve("data_processed") / "vision"
    ann = pd.read_parquet(cache / "annotations.parquet")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    video = is_video_backbone(cfg.clip_classifier.backbone)
    encoder = (ClipEncoder if video else FrameEncoder)(cfg.clip_classifier, device)
    for i, (_, row) in enumerate(ann.iterrows(), start=1):
        if video:
            extract_window_features(raw, row, cfg, encoder, cache)  # type: ignore[arg-type]
        else:
            extract_features(raw, row, cfg, encoder, cache)  # type: ignore[arg-type]
        if i % 20 == 0:
            log.info("features: %d/%d videos", i, len(ann))

    windows, x = build_windows(ann, cfg, cache)
    head, train_log = train(windows, x, cfg, device, args.fit_split)
    with torch.inference_mode():
        logits = head(torch.from_numpy(x).to(device)).squeeze(1).cpu().numpy()
    val = (windows["split"].eq(args.fit_split) & windows["label"].notna()).to_numpy()
    temperature = fit_temperature(logits[val], windows.loc[val, "label"].to_numpy())
    windows["clf_logit"] = logits
    windows["clf_p"] = sigmoid(logits / temperature)

    model_dir = PROJECT_ROOT / "data" / "models" / "clip" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "clip_head.pt"
    torch.save(head.cpu().state_dict(), weights)
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    meta = {
        "run_id": run_id,
        "temperature": temperature,
        "weights_sha256_16": weight_hash,
        "device": device,
        "fit_split": args.fit_split,
        "clip_classifier": cfg.clip_classifier.model_dump(),
    }
    (model_dir / "clip_classifier.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )

    out = base.paths.resolve("results") / "vision" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {"seed": SEED, **meta},
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    train_log.to_csv(out / "train_log.csv", index=False, lineterminator="\n")
    windows.to_parquet(cache / f"windows_{run_id}.parquet", index=False)
    write_meta(
        out / "train_log.csv",
        configs=vision_paths,
        synthetic=False,
        seeds={"seed": SEED},
        extra=meta,
    )
    log.info("run %s: temperature %.3f, weights %s", run_id, temperature, weight_hash)
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
