"""Assign public-dataset clips to the simulated cameras (road footage only).

    uv run python -m resqcity.streams.prepare_videos

For each camera in ``configs/streams.yaml``:
- accident clip: a UCF-Crime RoadAccidents TEST-split video with an annotated onset (so a live
  alarm can be checked against the true onset and offset);
- normal clips: ROAD footage = the part of OTHER onset-annotated test videos that ends
  ``NORMAL_MARGIN_S`` before their accident onset (trimmed with ffmpeg). Using road scenes avoids a
  scene change (e.g. shop interior -> road) masquerading as an accident (research audit, D-025).
Clips go to ``data/videos/<camera_id>/{normal,accident}/`` with a ``clips.yaml`` record.
Deterministic (sorted, round-robin).
"""

import argparse
import shutil
import subprocess

import pandas as pd
import yaml

from resqcity.common.config import load_base_config
from resqcity.streams.config import find_ffmpeg, load_streams_config

NORMAL_MARGIN_S = 1.0  # normal clips end this long before the other video's accident onset
MIN_NORMAL_S = 5.0  # shorter pre-onset segments are not used as normal footage


def trim(ffmpeg: str, src: str, dest: str, seconds: float) -> None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src,
        "-t",
        f"{seconds:.2f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        dest,
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base, cfg = load_base_config(), load_streams_config()
    ffmpeg = find_ffmpeg(cfg.ffmpeg)
    raw = base.paths.resolve("data_raw") / "vision"
    ann = pd.read_parquet(base.paths.resolve("data_processed") / "vision" / "annotations.parquet")
    onset_videos = ann[
        (ann["source"] == "ucf_crime")
        & (ann["label"] == "accident")
        & ann["has_onset"]
        & (ann["split"] == "test")
        & (ann["kind"] == "video")
    ].sort_values("video_id")
    normal_pool = onset_videos[onset_videos["accident_onset_s"] - NORMAL_MARGIN_S >= MIN_NORMAL_S]
    if onset_videos.empty or normal_pool.empty:
        raise RuntimeError("no onset-annotated test videos; run the vision data steps first")

    n_normal = cfg.clips_per_camera.get("normal", 0)
    for i, cam in enumerate(cfg.cameras):
        cam_dir = cfg.camera_dir(cam)
        if cam_dir.exists():
            shutil.rmtree(cam_dir)  # regenerate from scratch (old non-road clips must not remain)
        acc = onset_videos.iloc[i % len(onset_videos)]
        (cam_dir / "accident").mkdir(parents=True)
        (cam_dir / "normal").mkdir(parents=True)
        src = raw / acc["path"]
        shutil.copy2(src, cam_dir / "accident" / src.name)
        record: dict[str, list[dict[str, object]]] = {
            "accident": [
                {
                    "file": src.name,
                    "video_id": acc["video_id"],
                    "accident_onset_s": float(acc["accident_onset_s"]),
                    "accident_offset_s": float(acc["accident_offset_s"]),
                }
            ],
            "normal": [],
        }
        others = normal_pool[normal_pool["video_id"] != acc["video_id"]]
        for j in range(n_normal):
            other = others.iloc[(i * n_normal + j) % len(others)]
            length = float(other["accident_onset_s"]) - NORMAL_MARGIN_S
            name = f"road_{other['video_id']}_pre_onset.mp4"
            trim(ffmpeg, str(raw / other["path"]), str(cam_dir / "normal" / name), length)
            record["normal"].append(
                {
                    "file": name,
                    "video_id": other["video_id"],
                    "segment_s": [0.0, round(length, 2)],
                    "accident_onset_s": None,
                    "accident_offset_s": None,
                }
            )
        (cam_dir / "clips.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n"
        )
        print(cam, {k: [c["video_id"] for c in v] for k, v in record.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
