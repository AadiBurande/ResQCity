"""Metrics for the ward-hour demand forecasters (phase 5, D-030).

Pure functions over a panel slice and a vector of predictions: the time-based split, Poisson
deviance, MAE, the calibration table and the top-k ward hit rate. The driver that fits the models
and writes the results lives in ``sim.forecast_eval``, because it needs the oracle and the panel
and ``tests/test_architecture.py`` forbids ``forecasting -> sim``.

Metric notes: MAE is reported but is nearly useless on a panel where 99.6 % of ward-hour cells
are zero (predicting zero everywhere scores well); Poisson deviance, calibration and the top-k
ward hit rate carry the information, and the last is what pre-positioning actually cares about.
"""

import numpy as np
import pandas as pd

from resqcity.forecasting.config import ForecastingConfig

TOP_K = (3, 5, 10)
DECILES = 10
TIE_SEED = 20260126


def split(panel: pd.DataFrame, cfg: ForecastingConfig) -> dict[str, pd.DataFrame]:
    """Train / validation / test by time, with the lag warm-up dropped from all three."""
    live = panel[~panel["warmup"]]
    hours = np.sort(live["ts"].unique())
    test_start = hours[-cfg.split.test_weeks * 168]
    train_all = live[live["ts"] < test_start]
    valid_start = np.sort(train_all["ts"].unique())[-cfg.split.validation_weeks * 168]
    return {
        "train": train_all[train_all["ts"] < valid_start],
        "valid": train_all[train_all["ts"] >= valid_start],
        "test": live[live["ts"] >= test_start],
    }


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean unit Poisson deviance; the term y*log(y/mu) is 0 where y is 0."""
    mu = np.maximum(mu, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def top_k_hit_rate(rows: pd.DataFrame, pred: np.ndarray, k: int) -> float:
    """Share of real incidents that fall in the k wards with the highest predicted demand.

    Computed per hour over the wards, summing predictions across types -- the question
    pre-positioning asks when it decides where to stand units.
    """
    frame = pd.DataFrame(
        {"ts": rows["ts"].to_numpy(), "ward": rows["ward"].astype(str).to_numpy(),
         "count": rows["count"].to_numpy(), "pred": pred}
    )  # fmt: skip
    by_ward = frame.groupby(["ts", "ward"], observed=True)[["count", "pred"]].sum()
    # ties are broken at random rather than by ward name: a model that predicts the same value
    # everywhere (the uniform floor) would otherwise always "choose" the alphabetically first
    # wards, and score whatever those wards happen to contain
    rng = np.random.default_rng(TIE_SEED)
    jitter = rng.random(len(by_ward)) * 1e-12
    by_ward = by_ward.assign(rank_key=by_ward["pred"].to_numpy() + jitter)
    hits = total = 0
    for _, group in by_ward.groupby(level=0, observed=True):
        if group["count"].sum() == 0:
            continue
        chosen = group["rank_key"].nlargest(k).index
        hits += int(group.loc[chosen, "count"].sum())
        total += int(group["count"].sum())
    return hits / total if total else float("nan")


def calibration(rows: pd.DataFrame, pred: np.ndarray, model: str, scale: str) -> pd.DataFrame:
    """Observed mean against predicted mean over deciles of the prediction."""
    order = np.argsort(pred, kind="stable")
    bins = np.array_split(order, DECILES)
    counts = rows["count"].to_numpy(float)
    return pd.DataFrame(
        [
            {
                "model": model,
                "scale": scale,
                "decile": i + 1,
                "cells": len(b),
                "mean_predicted": float(pred[b].mean()),
                "mean_observed": float(counts[b].mean()),
            }
            for i, b in enumerate(bins)
        ]
    )


def to_daily(rows: pd.DataFrame, pred: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate ward-hour cells to ward-day (the panel is 99.6 % zeros at ward-hour)."""
    frame = pd.DataFrame(
        {"day": rows["ts"].dt.floor("D").to_numpy(), "ward": rows["ward"].astype(str).to_numpy(),
         "type": rows["type"].astype(str).to_numpy(), "count": rows["count"].to_numpy(),
         "pred": pred}
    )  # fmt: skip
    grouped = frame.groupby(["day", "ward", "type"], observed=True)[["count", "pred"]].sum()
    out = grouped.reset_index().rename(columns={"day": "ts"})
    return out, out["pred"].to_numpy(float)


def score(rows: pd.DataFrame, pred: np.ndarray, model: str) -> tuple[dict, list[pd.DataFrame]]:
    y = rows["count"].to_numpy(float)
    daily_rows, daily_pred = to_daily(rows, pred)
    metrics = {
        "model": model,
        "cells": len(rows),
        "observed": float(y.sum()),
        "predicted": float(pred.sum()),
        "poisson_deviance": poisson_deviance(y, pred),
        "mae": float(np.mean(np.abs(y - pred))),
        "poisson_deviance_daily": poisson_deviance(daily_rows["count"].to_numpy(float), daily_pred),
        "mae_daily": float(np.mean(np.abs(daily_rows["count"].to_numpy(float) - daily_pred))),
        **{f"top_{k}_ward_hit_rate": top_k_hit_rate(rows, pred, k) for k in TOP_K},
    }
    tables = [
        calibration(rows, pred, model, "ward_hour"),
        calibration(daily_rows, daily_pred, model, "ward_day"),
    ]
    return metrics, tables
