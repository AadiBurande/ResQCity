"""Combine stage 1 (heuristics) and stage 2 (classifier) into one calibrated confidence.

A logistic regression on [classifier logit, heuristic score, heuristic components] is fitted on the
VALIDATION windows (stacking). Its output is the ``Event.confidence``; the inputs are kept as
``Event.confidence_components`` for the explanation panel. Coefficients are saved so the weight of
each cue is inspectable.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from resqcity.vision.heuristics import COMPONENTS

STACK_FEATURES = ["clf_logit", "heur_score", *[f"c_{k}" for k in COMPONENTS]]
EPS = 1e-6


def window_heuristics(windows: pd.DataFrame, heur: pd.DataFrame) -> pd.DataFrame:
    """Max heuristic score/components over each window's frames (one video at a time)."""
    out = windows.copy()
    t = heur["t"].to_numpy()
    cols = ["score", *[f"c_{k}" for k in COMPONENTS]]
    values = heur[cols].to_numpy()
    agg = np.zeros((len(windows), len(cols)))
    for i, (s, e) in enumerate(zip(windows["start_s"], windows["end_s"], strict=True)):
        mask = (t >= s) & (t <= e)
        if mask.any():
            agg[i] = values[mask].max(axis=0)
    out["heur_score"] = agg[:, 0]
    for j, k in enumerate(COMPONENTS, start=1):
        out[f"c_{k}"] = agg[:, j]
    return out


@dataclass
class Stacker:
    coef: np.ndarray
    intercept: float
    features: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str] | None = None) -> "Stacker":
        from sklearn.linear_model import LogisticRegression

        features = features or STACK_FEATURES
        model = LogisticRegression(C=1.0, max_iter=2000, class_weight=None)
        model.fit(df[features].to_numpy(), df["label"].to_numpy().astype(int))
        return cls(model.coef_[0].astype(float), float(model.intercept_[0]), list(features))

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy() @ self.coef + self.intercept

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(df)))

    def proba_row(self, row: dict[str, float]) -> float:
        """Probability for one window given its features as a dict (live pipeline)."""
        x = np.array([row[f] for f in self.features], dtype=float)
        return float(1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept))))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "features": self.features,
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept,
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: Path) -> "Stacker":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.asarray(d["coef"], float), float(d["intercept"]), list(d["features"]))

    def components(self, row: dict[str, float]) -> dict[str, float]:
        """Per-cue contribution to the logit (for explanations)."""
        return {f: float(c * row[f]) for f, c in zip(self.features, self.coef, strict=True)}


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))
