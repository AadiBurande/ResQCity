"""Convert the D-Fire parquet mirror to an Ultralytics YOLO dataset.

    uv run python -m resqcity.vision.datasets.prepare_dfire

Class ids: the D-Fire README gives box totals (fire 14,692; smoke 11,865) but not the id mapping.
The mapping is inferred by matching per-id box counts to those totals and the check is written
to ``dataset_check.json`` (it fails loudly if the counts do not match either way).
Splits: the mirror's ``test`` split is kept as test; ``val`` is a seeded 10 % of ``train``.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from resqcity.common.config import load_base_config
from resqcity.vision.config import load_vision_config

log = logging.getLogger(__name__)
README_BOX_TOTALS = {"fire": 14692, "smoke": 11865}
COUNT_TOLERANCE = 0.02  # mirror may differ slightly from the README totals
VAL_FRACTION = 0.1


def box_counts(shards: list[Path]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for shard in shards:
        for label in pq.read_table(shard, columns=["label"]).column("label").to_pylist():
            for line in (label or "").strip().splitlines():
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def infer_names(counts: dict[int, int]) -> dict[int, str]:
    for fire_id, smoke_id in ((1, 0), (0, 1)):
        ok = all(
            abs(counts.get(i, 0) - README_BOX_TOTALS[name])
            <= COUNT_TOLERANCE * README_BOX_TOTALS[name]
            for i, name in ((fire_id, "fire"), (smoke_id, "smoke"))
        )
        if ok:
            return {fire_id: "fire", smoke_id: "smoke"}
    raise ValueError(
        f"box counts {counts} match neither id mapping of the README totals {README_BOX_TOTALS}"
    )


def write_split(shards: list[Path], out: Path, split_of: "dict[str, str] | str") -> int:
    n = 0
    for shard in shards:
        table = pq.read_table(shard)
        for image, label, filename in zip(
            table.column("image").to_pylist(),
            table.column("label").to_pylist(),
            table.column("filename").to_pylist(),
            strict=True,
        ):
            split = split_of if isinstance(split_of, str) else split_of[filename]
            stem = Path(filename).stem
            img_dir, lbl_dir = out / split / "images", out / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / f"{stem}.jpg").write_bytes(image["bytes"])
            (lbl_dir / f"{stem}.txt").write_text(
                (label or "").strip() + "\n" if label else "", encoding="utf-8"
            )
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    src = base.paths.resolve("data_raw") / "vision" / "dfire"
    out = base.paths.resolve("data_processed") / "vision" / "dfire_yolo"
    train_shards = sorted(src.glob("train-*.parquet"))
    test_shards = sorted(src.glob("test-*.parquet"))
    if not train_shards or not test_shards:
        raise FileNotFoundError(
            f"D-Fire shards missing in {src}; run resqcity.vision.datasets.download --only dfire"
        )

    counts = box_counts(train_shards + test_shards)
    names = infer_names(counts)
    filenames = [
        f
        for s in train_shards
        for f in pq.read_table(s, columns=["filename"]).column("filename").to_pylist()
    ]
    rng = np.random.default_rng(cfg.datasets.split.seed)
    val = set(
        rng.choice(filenames, size=round(VAL_FRACTION * len(filenames)), replace=False).tolist()
    )
    split_of = {f: ("val" if f in val else "train") for f in filenames}
    n_train = write_split(train_shards, out, split_of)
    n_test = write_split(test_shards, out, "test")

    data_yaml = {
        "path": out.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: names[i] for i in sorted(names)},
    }
    (out / "dataset.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8", newline="\n"
    )
    check = {
        "box_counts_by_id": counts,
        "readme_totals": README_BOX_TOTALS,
        "names": names,
        "images": {"train+val": n_train, "test": n_test, "val": len(val)},
    }
    (out / "dataset_check.json").write_text(
        json.dumps(check, indent=2), encoding="utf-8", newline="\n"
    )
    log.info("D-Fire YOLO dataset: %s", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
