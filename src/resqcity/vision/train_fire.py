"""Fine-tune a small YOLO on D-Fire (fire, smoke) and evaluate on its test split.

    uv run python -m resqcity.vision.train_fire [--epochs N] [--run-id NAME]

Weights: ``data/models/fire/<run_id>/best.pt`` (+ content hash). Metrics (mAP50, mAP50-95,
precision, recall per class on the TEST split) go to ``results/vision/<run_id>/fire_metrics.json``.
"""

import argparse
import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime

import yaml

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import load_vision_config
from resqcity.vision.detect_track import MODEL_DIR, resolve_device

log = logging.getLogger(__name__)
SEED = 20261020


def main(argv: list[str] | None = None) -> int:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    fc = cfg.fire_smoke
    run_id = args.run_id or datetime.now(UTC).strftime("fire_%Y%m%dT%H%M")
    data_yaml = base.paths.resolve("data_processed") / "vision" / "dfire_yolo" / "dataset.yaml"
    work = PROJECT_ROOT / "data" / "models" / "fire_runs"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or fc.epochs
    last = work / run_id / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise FileNotFoundError(f"nothing to resume: {last}")
        YOLO(str(last)).train(resume=True)  # restores the saved args (epochs, data, seed, ...)
    else:
        YOLO(str(MODEL_DIR / fc.base_weights)).train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=fc.imgsz,
            batch=fc.batch,
            seed=SEED,
            deterministic=True,
            device=resolve_device(cfg.detector.device),
            project=str(work),
            name=run_id,
            exist_ok=True,
            verbose=False,
            workers=fc.workers,
        )
    best = work / run_id / "weights" / "best.pt"
    dest_dir = PROJECT_ROOT / "data" / "models" / "fire" / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest_dir / "best.pt")
    weight_hash = hashlib.sha256((dest_dir / "best.pt").read_bytes()).hexdigest()[:16]

    metrics = YOLO(str(dest_dir / "best.pt")).val(
        data=str(data_yaml),
        split="test",
        imgsz=fc.imgsz,
        verbose=False,
        project=str(work),
        name=f"{run_id}_test",
        exist_ok=True,
    )
    names = metrics.names
    per_class = {
        names[int(c)]: {
            "precision": float(p),
            "recall": float(r),
            "map50": float(m50),
            "map50_95": float(m),
        }
        for c, p, r, m50, m in zip(
            metrics.box.ap_class_index,
            metrics.box.p,
            metrics.box.r,
            metrics.box.ap50,
            metrics.box.ap,
            strict=True,
        )
    }
    result = {
        "run_id": run_id,
        "weights_sha256_16": weight_hash,
        "epochs": epochs,
        "split": "test",
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "dataset": "D-Fire (Venancio et al. 2022) via HF mirror; see docs/data-sources.md",
    }
    out = base.paths.resolve("results") / "vision" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "fire_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {"fire_smoke": fc.model_dump(), "seed": SEED, "epochs": epochs}, sort_keys=False
        ),
        encoding="utf-8",
        newline="\n",
    )
    write_meta(
        out / "fire_metrics.json",
        configs=[CONFIG_DIR / "vision.yaml"],
        synthetic=False,
        seeds={"seed": SEED},
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
