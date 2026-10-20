"""Assign public-dataset clips to the simulated cameras.

    uv run python -m resqcity.streams.prepare_videos

For each camera in ``configs/streams.yaml``: normal clips (UCF-Crime normal videos) and accident
clips (UCF-Crime RoadAccidents TEST-split videos with annotated onsets, so a demo alarm can be
checked against the true onset) are copied to ``data/videos/<camera_id>/{normal,accident}/``.
A ``clips.yaml`` per camera records the source video ids. Deterministic (sorted, round-robin).
"""

import argparse
import shutil

import pandas as pd
import yaml

from resqcity.common.config import load_base_config
from resqcity.streams.config import load_streams_config


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base, cfg = load_base_config(), load_streams_config()
    raw = base.paths.resolve("data_raw") / "vision"
    ann = pd.read_parquet(base.paths.resolve("data_processed") / "vision" / "annotations.parquet")
    ucf = ann[(ann["source"] == "ucf_crime") & (ann["kind"] == "video")]
    pools = {
        "accident": ucf[
            (ucf["label"] == "accident") & ucf["has_onset"] & (ucf["split"] == "test")
        ].sort_values("video_id"),
        "normal": ucf[(ucf["label"] == "normal") & (ucf["split"] == "test")].sort_values(
            "video_id"
        ),
    }
    for kind, pool in pools.items():
        if len(pool) == 0:
            raise RuntimeError(f"no {kind} test clips available; run the vision data steps first")
    for i, cam in enumerate(cfg.cameras):
        record: dict[str, list[dict[str, object]]] = {}
        for kind, pool in pools.items():
            n = cfg.clips_per_camera.get(kind, 0)
            chosen = [pool.iloc[(i * n + j) % len(pool)] for j in range(n)]
            dest = cfg.camera_dir(cam) / kind
            dest.mkdir(parents=True, exist_ok=True)
            record[kind] = []
            for row in chosen:
                src = raw / row["path"]
                shutil.copy2(src, dest / src.name)
                record[kind].append(
                    {
                        "file": src.name,
                        "video_id": row["video_id"],
                        "accident_onset_s": None
                        if pd.isna(row["accident_onset_s"])
                        else float(row["accident_onset_s"]),
                    }
                )
        (cfg.camera_dir(cam) / "clips.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n"
        )
        print(cam, {k: [c["video_id"] for c in v] for k, v in record.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
