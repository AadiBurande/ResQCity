"""The oracle forecaster: the generator's own intensity (phase 5, D-030).

A DECLARED ORACLE ABLATION. It reads the ``intensity`` column of the panel, which is the
generator's expected count for that cell computed analytically from its own parameters, so no
forecaster can do better on average. It lives in ``sim`` because that is where ground truth is
allowed (master plan section 1: "a declared oracle ablation, if ever needed, must live in sim/ and
be named as such"), and it exists to bound phase 10's forecast-quality sweep from above -- never
to be reported as an achievable forecast.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OracleForecast:
    """Ground truth. Not a forecast: the perfect-information end of the phase-10 sweep."""

    name: str = "oracle"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if "intensity" not in rows:
            raise KeyError("the oracle needs the panel's ground-truth 'intensity' column")
        return rows["intensity"].to_numpy(float)
