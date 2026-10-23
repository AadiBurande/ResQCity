"""Metric functions for the accident detector (pure numpy/pandas; unit-tested on toy inputs)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from resqcity.vision.alarm import alarm_times, verified_alarm_times
from resqcity.vision.config import AlarmConfig

SECONDS_PER_HOUR = 3600.0
# An alarm counts as a detection if it falls in [onset, offset + DETECTION_GRACE_S]. Fixed before
# the v3 evaluation (D-025): covers the 3 s window plus up to four windows of alarm persistence.
DETECTION_GRACE_S = 10.0
# ACCIDENT clips (evaluation v5, pre-registered in docs/plans/phase-3b.md): only the impact time t*
# is annotated, so detection = an alarm in [t* - ACCIDENT_PRE_TOLERANCE_S, t* + DETECTION_GRACE_S].
ACCIDENT_PRE_TOLERANCE_S = 1.0
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261020


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


def _alarms(
    w: pd.DataFrame,
    score_col: str,
    cfg: AlarmConfig,
    threshold: float,
    video_id: str,
    verify: Callable[[str, float], bool] | None,
) -> list[float]:
    times, scores = w["end_s"].tolist(), w[score_col].tolist()
    if verify is None:
        return alarm_times(times, scores, cfg, threshold)
    return verified_alarm_times(times, scores, cfg, threshold, lambda t: verify(video_id, t))


def video_alarm_summary(
    windows: pd.DataFrame,
    videos: pd.DataFrame,
    score_col: str,
    cfg: AlarmConfig,
    threshold: float,
    grace_s: float = DETECTION_GRACE_S,
    verify: Callable[[str, float], bool] | None = None,
) -> pd.DataFrame:
    """Per video: alarms classified by WHEN they fire (alarms are timestamped at window end).
    ``verify(video_id, t)`` (optional, D-027) is a second-stage check of every candidate alarm.

    - accident video with annotated onset: detection = an alarm in [onset, offset + grace_s];
      alarms before onset are false alarms on ROAD footage (``road_hours`` = onset);
      alarms after offset + grace_s are ``late_alarms`` (aftermath: neither hit nor false alarm);
    - normal video (UCF-Crime "Normal", mostly NOT road scenes): every alarm is a false alarm on
      ``nonroad_hours``;
    - accident video without onset: only ``any_alarm`` (no timing can be checked).
    """
    rows = []
    for _, v in videos.iterrows():
        w = windows[windows["video_id"] == v["video_id"]].sort_values("end_s")
        alarms = _alarms(w, score_col, cfg, threshold, v["video_id"], verify)
        has_onset = bool(v.get("has_onset", False))
        row = {
            "video_id": v["video_id"],
            "source": v["source"],
            "label": v["label"],
            "has_onset": has_onset,
            "threshold": threshold,
            "n_alarms": len(alarms),
            "any_alarm": bool(alarms),
            "detected": False,
            "delay_s": None,
            "road_false_alarms": 0,
            "road_hours": 0.0,
            "nonroad_false_alarms": 0,
            "nonroad_hours": 0.0,
            "late_alarms": 0,
        }
        if v["label"] == "normal":
            row["nonroad_false_alarms"] = len(alarms)
            row["nonroad_hours"] = float(v["duration_s"]) / SECONDS_PER_HOUR
        elif has_onset:
            onset = float(v["accident_onset_s"])
            offset = float(v["accident_offset_s"]) if pd.notna(v["accident_offset_s"]) else onset
            hits = [a for a in alarms if onset <= a <= offset + grace_s]
            row["detected"] = bool(hits)
            row["delay_s"] = (hits[0] - onset) if hits else None
            row["road_false_alarms"] = sum(a < onset for a in alarms)
            row["road_hours"] = onset / SECONDS_PER_HOUR
            row["late_alarms"] = sum(a > offset + grace_s for a in alarms)
        rows.append(row)
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


def cluster_bootstrap_rate(counts: np.ndarray, hours: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval of sum(counts) / sum(hours), resampling VIDEOS (false alarms in
    one video are not independent)."""
    counts, hours = np.asarray(counts, float), np.asarray(hours, float)
    if len(counts) == 0 or hours.sum() <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(counts), size=(BOOTSTRAP_REPS, len(counts)))
    tot_h = hours[idx].sum(axis=1)
    ok = tot_h > 0
    rates = counts[idx].sum(axis=1)[ok] / tot_h[ok]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _ci(name: str, interval: tuple[float, float]) -> dict[str, float]:
    return {f"{name}_ci95_lo": interval[0], f"{name}_ci95_hi": interval[1]}


def rate_block(name: str, counts: pd.Series, hours: pd.Series) -> dict[str, float]:
    total_h = float(hours.sum())
    total = int(counts.sum())
    return {
        f"{name}_false_alarms": total,
        f"{name}_hours": total_h,
        f"{name}_false_alarms_per_hour": total / total_h if total_h > 0 else float("nan"),
        **_ci(f"{name}_fa_poisson", poisson_rate_interval(total, total_h)),
        **_ci(f"{name}_fa_cluster", cluster_bootstrap_rate(counts.to_numpy(), hours.to_numpy())),
    }


def accident_alarm_summary(
    windows: pd.DataFrame,
    videos: pd.DataFrame,
    score_col: str,
    cfg: AlarmConfig,
    threshold: float,
    window_s: float,
    stride_s: float,
    grace_s: float = DETECTION_GRACE_S,
    pre_s: float = ACCIDENT_PRE_TOLERANCE_S,
    verify: Callable[[str, float], bool] | None = None,
) -> pd.DataFrame:
    """Evaluation-v5 rule for ACCIDENT clips (``accident_onset_s`` = impact time t*):
    detection = first alarm in [t* - pre_s, t* + grace_s]; false alarm = alarm before t* - pre_s;
    road footage = max(0, t* - pre_s - warm-up), where warm-up = window_s + (k - 1) * stride_s is
    the earliest time an alarm can fire; later alarms are ``late_alarms``. Same columns as
    :func:`video_alarm_summary`, so :func:`summarise_alarms` applies."""
    warm_up = window_s + (cfg.consecutive_windows - 1) * stride_s
    rows = []
    for _, v in videos.iterrows():
        w = windows[windows["video_id"] == v["video_id"]].sort_values("end_s")
        alarms = _alarms(w, score_col, cfg, threshold, v["video_id"], verify)
        start = float(v["accident_onset_s"]) - pre_s
        end = float(v["accident_onset_s"]) + grace_s
        hits = [a for a in alarms if start <= a <= end]
        rows.append(
            {
                "video_id": v["video_id"],
                "source": v["source"],
                "label": "accident",
                "has_onset": True,
                "threshold": threshold,
                "n_alarms": len(alarms),
                "any_alarm": bool(alarms),
                "detected": bool(hits),
                "delay_s": (hits[0] - float(v["accident_onset_s"])) if hits else None,
                "road_false_alarms": sum(a < start for a in alarms),
                "road_hours": max(0.0, start - warm_up) / SECONDS_PER_HOUR,
                "nonroad_false_alarms": 0,
                "nonroad_hours": 0.0,
                "late_alarms": sum(a > end for a in alarms),
            }
        )
    return pd.DataFrame(rows)


def normal_road_summary(
    windows: pd.DataFrame,
    videos: pd.DataFrame,
    score_col: str,
    cfg: AlarmConfig,
    threshold: float,
    window_s: float,
    stride_s: float,
    verify: Callable[[str, float], bool] | None = None,
) -> pd.DataFrame:
    """Normal road CCTV without accidents (UA-DETRAC, amendment 2): every alarm is a false alarm;
    footage counts after the warm-up (window_s + (k - 1) * stride_s)."""
    warm_up = window_s + (cfg.consecutive_windows - 1) * stride_s
    rows = []
    for _, v in videos.iterrows():
        w = windows[windows["video_id"] == v["video_id"]].sort_values("end_s")
        alarms = _alarms(w, score_col, cfg, threshold, v["video_id"], verify)
        rows.append(
            {
                "video_id": v["video_id"],
                "road_false_alarms": len(alarms),
                "road_hours": max(0.0, float(v["duration_s"]) - warm_up) / SECONDS_PER_HOUR,
            }
        )
    return pd.DataFrame(rows)


def summarise_alarms(per_video: pd.DataFrame) -> dict[str, float]:
    """Headline metrics. ``onset_detection_rate`` (timing-checked) is the recall exported to the
    simulator; ``any_alarm_rate`` (accident videos without onsets) does NOT check timing."""
    acc = per_video[per_video["label"] == "accident"]
    onset = acc[acc["has_onset"]]
    weak = acc[~acc["has_onset"]]
    normal = per_video[per_video["label"] == "normal"]
    delays = onset["delay_s"].dropna().to_numpy(float)
    det = int(onset["detected"].sum())
    fp_videos = int((onset["road_false_alarms"] > 0).sum()) + int(
        (normal["nonroad_false_alarms"] > 0).sum()
    )
    neg_videos = len(onset) + len(normal)
    tpr = det / len(onset) if len(onset) else float("nan")
    fpr = fp_videos / neg_videos if neg_videos else float("nan")
    return {
        "onset_videos": len(onset),
        "onset_detected": det,
        "onset_detection_rate": tpr,
        **_ci("onset_detection_rate", wilson_interval(det, len(onset))),
        "n_delays": len(delays),
        "delay_median_s": float(np.median(delays)) if len(delays) else float("nan"),
        "delay_max_s": float(np.max(delays)) if len(delays) else float("nan"),
        "late_alarms": int(onset["late_alarms"].sum()),
        "weak_videos": len(weak),
        "any_alarm_rate": float(weak["any_alarm"].mean()) if len(weak) else float("nan"),
        **_ci("any_alarm_rate", wilson_interval(int(weak["any_alarm"].sum()), len(weak))),
        "fp_video_rate": fpr,
        "youden_j": tpr - fpr if neg_videos and len(onset) else float("nan"),
        **rate_block("road", onset["road_false_alarms"], onset["road_hours"]),
        **rate_block("nonroad", normal["nonroad_false_alarms"], normal["nonroad_hours"]),
    }


def per_video_auroc(
    windows: pd.DataFrame, score_col: str
) -> tuple[float, tuple[float, float], int]:
    """Mean per-video AUROC (pre-onset vs accident windows inside each onset video) with a 95 %
    bootstrap interval over videos; videos without both classes are skipped."""
    values = []
    for _, g in windows.groupby("video_id"):
        a = auroc(g["label"].to_numpy(), g[score_col].to_numpy())
        if not np.isnan(a):
            values.append(a)
    if not values:
        return float("nan"), (float("nan"), float("nan")), 0
    v = np.asarray(values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = v[rng.integers(0, len(v), size=(BOOTSTRAP_REPS, len(v)))].mean(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return float(v.mean()), ci, len(v)
