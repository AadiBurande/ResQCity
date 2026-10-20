"""Privacy verification on real sample frames (acceptance: results/vision/privacy_check.png).

    uv run python -m resqcity.vision.eval.privacy_check

Picks test-split frames with people and vehicles, runs detection and the ``Anonymizer``, and saves
ONLY the anonymised frame (with outlines marking each blurred region; the raw frame is never
written). For every region it records the image detail (variance of the Laplacian) before and
after, and the check passes when detail drops by >= ``MIN_REDUCTION`` in every region.
"""

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.vision.config import load_vision_config
from resqcity.vision.ingest import file_frames
from resqcity.vision.privacy import Anonymizer
from resqcity.vision.types import Box

MIN_REDUCTION = 0.8  # detail must fall by at least 80 % inside every blurred region
MAX_FRAMES_SCANNED = 400
OUTLINE = {"faces": (0, 0, 255), "plates": (0, 200, 255), "persons": (255, 0, 255)}


def detail(image: np.ndarray, box: Box) -> float:
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(image.shape[1], int(box.x2)), min(image.shape[0], int(box.y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return float("nan")
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main(argv: list[str] | None = None) -> int:
    from resqcity.vision.detect_track import YoloTracker

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    base, cfg = load_base_config(), load_vision_config()
    raw = base.paths.resolve("data_raw") / "vision"
    ann = pd.read_parquet(base.paths.resolve("data_processed") / "vision" / "annotations.parquet")
    anon = Anonymizer(cfg.privacy, cfg.detector.vehicle_classes, cfg.detector.person_class)
    best = None
    for _, row in ann[ann["split"] == "test"].sort_values("video_id").iterrows():
        tracker = YoloTracker(cfg.detector)
        for i, frame in enumerate(file_frames(raw / row["path"], cfg.ingest)):
            if i >= MAX_FRAMES_SCANNED:
                break
            det = tracker.update(frame)
            kinds = {o.cls for o in det.tracks}
            score = len(det.tracks) + 5 * (cfg.detector.person_class in kinds)
            if (
                cfg.detector.person_class in kinds
                and kinds & set(cfg.detector.vehicle_classes)
                and (best is None or score > best[0])
            ):
                best = (score, row["video_id"], frame, det)
        if best is not None and best[0] >= 10:
            break
    if best is None:
        raise RuntimeError("no test frame with both people and vehicles found")
    _, video_id, frame, det = best
    regions = anon.regions(frame.image, det.tracks)
    out = anon.apply(frame.image, det.tracks)
    checks = []
    for kind, boxes in regions.items():
        for b in boxes:
            before, after = detail(frame.image, b), detail(out, b)
            if np.isnan(before) or before == 0:
                continue
            checks.append(
                {
                    "kind": kind,
                    "w": round(b.w, 1),
                    "h": round(b.h, 1),
                    "before": before,
                    "after": after,
                    "reduction": 1 - after / before,
                }
            )
    # Outlines are drawn only AFTER all measurements (drawn edges would inflate detail).
    for kind, boxes in regions.items():
        for b in boxes:
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), OUTLINE[kind], 1)
    cv2.putText(
        out,
        "ANONYMISED (outlines = blurred regions)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    results = base.paths.resolve("results") / "vision"
    results.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(results / "privacy_check.png"), out)
    passed = bool(checks) and all(c["reduction"] >= MIN_REDUCTION for c in checks)
    report = {
        "video_id": video_id,
        "t": frame.t,
        "regions": checks,
        "min_reduction_required": MIN_REDUCTION,
        "passed": passed,
        "counts": {k: len(v) for k, v in regions.items()},
    }
    (results / "privacy_check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    write_meta(
        results / "privacy_check.json", configs=[CONFIG_DIR / "vision.yaml"], synthetic=False
    )
    print(json.dumps({k: v for k, v in report.items() if k != "regions"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
