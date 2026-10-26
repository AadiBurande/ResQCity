"""Ward-hour demand forecasters (phase 5, D-030).

Every model answers the same question -- the expected number of incidents of one type, in one
ward, in one hour -- behind the :class:`Forecaster` protocol, so phase 6 pre-positioning and the
phase-10 sweeps can swap them freely:

``Climatology``      the mean count per (ward, hour-of-week, type) over the training years. The
                     baseline a control room already has.
``UniformRate``      the citywide mean spread evenly over wards and types: the "no information"
                     floor for phase 10.
``LightGbmPoisson``  gradient boosting with a Poisson objective, fitted twice -- once on the
                     ``realistic`` feature set and once on ``informed``. The informed model is
                     handed the generator's own inputs and therefore recovers the intensity
                     formula almost exactly; it is CIRCULAR by construction and labelled as such.
``NoisyForecaster``  wraps any of the above and multiplies by lognormal noise: the phase-10
                     forecast-quality knob. Seeded through ``common.rng.stream`` so a sweep can
                     use common random numbers.

The oracle (the generator's true intensity) is not here: it would need ``sim``, which
``tests/test_architecture.py`` forbids this package from importing. It lives in
``sim.oracle_forecast`` and reads the panel's ``intensity`` column.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from resqcity.common.rng import stream
from resqcity.forecasting.config import ForecastingConfig

NOISE_STREAM = "forecast_noise"
CATEGORICAL = ("ward", "type")
FEATURE_COLUMNS = {
    "hour": ["hour"],
    "weekday": ["weekday"],
    "month": ["month"],
    "ward": ["ward"],
    "type": ["type"],
    "festival": ["festival"],
    "rain_true": ["rain_mm_h"],
    "rain_forecast": ["rain_fc"],
    "hotspot_mass": ["hotspot_mass"],
}


def feature_columns(cfg: ForecastingConfig, name: str) -> list[str]:
    """Panel columns a named feature set may use ("lags" expands to the configured lags)."""
    out: list[str] = []
    for item in cfg.feature_sets[name]:
        if item == "lags":
            out += [f"lag_{k}h" for k in cfg.features.ward_lags_h]
            out.append(f"lag_city_{cfg.features.citywide_lag_h}h")
        else:
            out += FEATURE_COLUMNS[item]
    return out


class Forecaster(Protocol):
    name: str

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Expected incident count for each row of a panel slice."""
        ...


@dataclass
class UniformRate:
    """Citywide mean count per cell: knows nothing about ward, hour or type."""

    rate: float
    name: str = "uniform"

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "UniformRate":
        return cls(rate=float(train["count"].mean()))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.rate)


@dataclass
class Climatology:
    """Historical average, estimated the way a sparse panel demands.

    A per-cell mean over (ward, hour-of-week, type) is degenerate here: with ~0.1 incidents per
    cell in the training years it predicts exactly zero for 91 % of cells, and most test
    incidents then land on a zero. So the hour-of-week shape is POOLED across wards (58x more
    data) and applied multiplicatively to each ward's own long-run rate:

        rate(ward, how, type) = rate(ward, type) x profile(how, type)

    with the profile normalised to mean 1 within each type, which keeps the citywide total equal
    to the training total. This is the baseline a competent analyst would build, not a strawman.
    """

    ward_type: pd.Series
    profile: pd.Series
    fallback: float
    name: str = "climatology"

    @staticmethod
    def _key(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        ward = rows["ward"].astype(str).reset_index(drop=True)
        how = (rows["weekday"].astype(int) * 24 + rows["hour"].astype(int)).reset_index(drop=True)
        kind = rows["type"].astype(str).reset_index(drop=True)
        return ward, how, kind

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "Climatology":
        ward, how, kind = cls._key(train)
        counts = train["count"].reset_index(drop=True)
        ward_type = counts.groupby([ward, kind], observed=True).mean()
        by_how = counts.groupby([how, kind], observed=True).mean()
        by_type = counts.groupby(kind, observed=True).mean()
        profile = by_how / by_type.reindex(by_how.index.get_level_values(1)).to_numpy()
        return cls(ward_type=ward_type, profile=profile, fallback=float(counts.mean()))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        ward, how, kind = self._key(rows)
        base = self.ward_type.reindex(pd.MultiIndex.from_arrays([ward, kind])).to_numpy(float)
        shape = self.profile.reindex(pd.MultiIndex.from_arrays([how, kind])).to_numpy(float)
        out = np.nan_to_num(base, nan=self.fallback) * np.nan_to_num(shape, nan=1.0)
        return np.maximum(out, 0.0)


@dataclass
class LightGbmPoisson:
    """LightGBM with a Poisson objective on one of the two feature sets."""

    booster: object
    columns: list[str]
    feature_set: str
    best_iteration: int
    name: str = "lightgbm"

    @staticmethod
    def _frame(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = rows[columns].copy()
        for col in CATEGORICAL:
            if col in out:
                out[col] = out[col].astype("category")
        return out

    @classmethod
    def fit(
        cls,
        cfg: ForecastingConfig,
        feature_set: str,
        train: pd.DataFrame,
        valid: pd.DataFrame,
    ) -> "LightGbmPoisson":
        import lightgbm as lgb

        columns = feature_columns(cfg, feature_set)
        lgbm = cfg.lightgbm
        sets = [
            lgb.Dataset(
                cls._frame(part, columns),
                label=part["count"].to_numpy(float),
                categorical_feature=[c for c in CATEGORICAL if c in columns],
                free_raw_data=False,
            )
            for part in (train, valid)
        ]
        booster = lgb.train(
            lgbm.params(),
            sets[0],
            num_boost_round=lgbm.num_boost_round,
            valid_sets=[sets[1]],
            callbacks=[lgb.early_stopping(lgbm.early_stopping_rounds, verbose=False)],
        )
        return cls(
            booster=booster,
            columns=columns,
            feature_set=feature_set,
            best_iteration=int(booster.best_iteration),
            name=f"lightgbm_{feature_set}",
        )

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        values = self.booster.predict(
            self._frame(rows, self.columns), num_iteration=self.best_iteration
        )
        return np.maximum(np.asarray(values, float), 0.0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "booster.txt"), num_iteration=self.best_iteration)
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "feature_set": self.feature_set,
                    "columns": self.columns,
                    "best_iteration": self.best_iteration,
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, directory: Path) -> "LightGbmPoisson":
        import lightgbm as lgb

        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(directory / "booster.txt"))
        return cls(
            booster=booster,
            columns=meta["columns"],
            feature_set=meta["feature_set"],
            best_iteration=meta["best_iteration"],
            name=meta["name"],
        )

    def importance(self) -> dict[str, int]:
        gains = self.booster.feature_importance(importance_type="gain")
        return {
            c: int(g) for c, g in sorted(zip(self.columns, gains, strict=True), key=lambda x: -x[1])
        }


@dataclass
class NoisyForecaster:
    """Phase-10 knob: ``inner`` times lognormal(-sigma^2/2, sigma), so the mean is unchanged."""

    inner: Forecaster
    sigma: float
    seed_base: int
    replication: int = 0

    def __post_init__(self) -> None:
        self.name = f"{self.inner.name}+noise{self.sigma:g}"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        base = self.inner.predict(rows)
        if self.sigma <= 0:
            return base
        rng = stream(self.seed_base, self.replication, NOISE_STREAM)
        draw = rng.normal(-0.5 * self.sigma**2, self.sigma, len(base))
        return base * np.exp(draw)
