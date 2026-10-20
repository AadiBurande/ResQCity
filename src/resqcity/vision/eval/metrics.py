"""Metric functions for the accident detector (pure numpy/pandas; unit-tested on toy inputs)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from resqcity.vision.alarm import alarm_times
from resqcity.vision.config import AlarmConfig

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    positives: int
    precision: float
    recall: float
    f1: float
    auroc: float


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> BinaryMetrics:
    y = np.asarray(y).astype(int)
    pred = np.asarray(score) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return BinaryMetrics(len(y), int(y.sum()), precision, recall, f1, auroc(y, score))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie handling; NaN if only one class."""
    y = np.asarray(y).astype(int)
    pos, neg = np.asarray(score)[y == 1], np.asarray(score)[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration table: mean predicted vs observed frequency per probability bin + ECE."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "lo": edges[b],
                    "hi": edges[b + 1],
                    "n": int(m.sum()),
                    "mean_p": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.attrs["ece"] = float(
            (table["n"] * (table["mean_p"] - table["observed"]).abs()).sum() / len(p)
        )
    return table


def video_alarm_summary(
    windows: pd.DataFrame, videos: pd.DataFrame, score_col: str, cfg: AlarmConfig, threshold: float
) -> pd.DataFrame:
    """Per video: alarms, first alarm after onset, delay, and false alarms on the normal portion.

    ``windows`` needs video_id, end_s and ``score_col``; alarms are timestamped at window end.
    Normal portion = whole normal video, or [0, onset) of an annotated accident video.
    """
    rows = []
    for _, v in videos.iterrows():
        w = windows[windows["video_id"] == v["video_id"]].sort_values("end_s")
        times = w["end_s"].tolist()
        alarms = alarm_times(times, w[score_col].tolist(), cfg, threshold)
        onset = v["accident_onset_s"] if v.get("has_onset", False) else None
        if v["label"] == "normal":
            normal_s, false_alarms, detected, delay = (
                float(v["duration_s"]),
                len(alarms),
                False,
                None,
            )
        elif onset is not None:
            early = [a for a in alarms if a < onset]
            late = [a for a in alarms if a >= onset]
            normal_s, false_alarms = float(onset), len(early)
            detected = bool(late)
            delay = (late[0] - onset) if late else None
        else:  # accident video without onset: detection only
            normal_s, false_alarms, detected, delay = 0.0, 0, bool(alarms), None
        rows.append(
            {
                "video_id": v["video_id"],
                "source": v["source"],
                "label": v["label"],
                "has_onset": bool(onset is not None),
                "threshold": threshold,
                "n_alarms": len(alarms),
                "detected": detected,
                "delay_s": delay,
                "false_alarms": false_alarms,
                "normal_hours": normal_s / SECONDS_PER_HOUR,
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion (well-behaved for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def poisson_rate_interval(count: int, exposure: float) -> tuple[float, float]:
    """Exact (Garwood) 95 % interval for a Poisson rate = count / exposure."""
    from scipy.stats import chi2

    if exposure <= 0:
        return float("nan"), float("nan")
    lo = chi2.ppf(0.025, 2 * count) / 2 if count > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (count + 1)) / 2
    return float(lo / exposure), float(hi / exposure)


def _ci(name: str, interval: tuple[float, float]) -> dict[str, float]:
    return {f"{name}_ci95_lo": interval[0], f"{name}_ci95_hi": interval[1]}


def summarise_alarms(per_video: pd.DataFrame) -> dict[str, float]:
    acc = per_video[per_video["label"] == "accident"]
    with_onset = acc[acc["has_onset"]]
    delays = with_onset["delay_s"].dropna().to_numpy(float)
    normal_vids = per_video[per_video["label"] == "normal"]
    hours = per_video["normal_hours"].sum()
    tp = int(acc["detected"].sum())
    fp_videos = int((normal_vids["n_alarms"] > 0).sum())
    precision = tp / (tp + fp_videos) if tp + fp_videos else float("nan")
    recall = tp / len(acc) if len(acc) else float("nan")
    return {
        "videos_accident": len(acc),
        "videos_normal": len(normal_vids),
        "video_recall": recall,
        "video_precision": precision,
        "video_f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "onset_videos": len(with_onset),
        "onset_detected": int(with_onset["detected"].sum()),
        "delay_median_s": float(np.median(delays)) if len(delays) else float("nan"),
        "delay_p90_s": float(np.quantile(delays, 0.9)) if len(delays) else float("nan"),
        "false_alarms": int(per_video["false_alarms"].sum()),
        "normal_hours": float(hours),
        "false_alarms_per_camera_hour": float(per_video["false_alarms"].sum() / hours)
        if hours > 0
        else float("nan"),
        **_ci(
            "false_alarms_per_camera_hour",
            poisson_rate_interval(int(per_video["false_alarms"].sum()), float(hours)),
        ),
        **_ci("video_recall", wilson_interval(tp, len(acc))),
        **_ci(
            "onset_detected", wilson_interval(int(with_onset["detected"].sum()), len(with_onset))
        ),
    }
