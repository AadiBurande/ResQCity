"""Fit and score the ward-hour demand forecasters (phase 5, D-030).

    uv run python -m resqcity.sim.forecast_eval

Time-based split, never shuffled: the last ``test_weeks`` of the history are held out, and
LightGBM picks its stopping round on the last ``validation_weeks`` of the TRAINING part only.

**Forecast accuracy here is not a research claim.** The training data comes from the incident
generator, whose intensity is a formula this project wrote, so the ``informed`` model -- handed
the generator's own rain and ward weights -- recovers that formula almost exactly. It is reported
as the circular upper bound it is. The research claim is only how dispatch performance depends on
forecast quality, which phase 10 sweeps using these models as its quality levels.

Metrics (per model, at ward-hour and ward-day): Poisson deviance, MAE (reported but nearly
useless on a panel where 99.6 % of cells are zero -- predicting zero everywhere scores well),
a calibration table over predicted deciles, and the top-k ward hit rate, which is what
pre-positioning actually cares about.
Outputs ``results/forecasting/metrics_synthetic.json`` and ``calibration_synthetic.csv``.

It lives in ``sim`` rather than ``forecasting`` because it uses the declared oracle ablation
and the panel builder, and ``tests/test_architecture.py`` forbids ``forecasting -> sim``.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, load_base_config
from resqcity.common.provenance import write_meta
from resqcity.forecasting.config import ForecastingConfig, load_forecasting_config
from resqcity.forecasting.metrics import score, split
from resqcity.forecasting.models import (
    Climatology,
    Forecaster,
    LightGbmPoisson,
    NoisyForecaster,
    UniformRate,
)

OUT = PROJECT_ROOT / "results" / "forecasting"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "forecast"
NOISE_LEVELS = (0.5, 1.0)
log = logging.getLogger(__name__)


def fit_all(cfg: ForecastingConfig, parts: dict[str, pd.DataFrame]) -> list[Forecaster]:
    from resqcity.sim.oracle_forecast import OracleForecast  # declared oracle ablation

    train, valid = parts["train"], parts["valid"]
    fitted: list[Forecaster] = [UniformRate.fit(train), Climatology.fit(train)]
    for name in ("realistic", "informed"):
        log.info("fitting lightgbm on the %s feature set (%d rows)", name, len(train))
        model = LightGbmPoisson.fit(cfg, name, train, valid)
        model.save(MODEL_DIR / model.name)
        log.info("  stopped at iteration %d", model.best_iteration)
        fitted.append(model)
    fitted.append(OracleForecast())
    return fitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from resqcity.sim.panel import panel_path  # path helper only; no ground truth is read here

    cfg = load_forecasting_config()
    base = load_base_config()
    panel = pd.read_parquet(args.panel or panel_path(cfg))
    parts = split(panel, cfg)
    log.info("split: %s", {k: len(v) for k, v in parts.items()})
    models = fit_all(cfg, parts)
    rows = parts["test"]
    metrics, tables = [], []
    for model in models:
        m, t = score(rows, model.predict(rows), model.name)
        metrics.append(m)
        tables += t
        log.info(
            "%-22s deviance %.5f  top-5 %.3f  predicted %.0f vs %.0f observed",
            model.name, m["poisson_deviance"], m["top_5_ward_hit_rate"],
            m["predicted"], m["observed"],
        )  # fmt: skip
    base_model = next(m for m in models if m.name == "lightgbm_realistic")
    for sigma in NOISE_LEVELS:
        noisy = NoisyForecaster(base_model, sigma, base.random.forecast_training_seed_base)
        m, t = score(rows, noisy.predict(rows), noisy.name)
        metrics.append(m)
        tables += t
    importances = {m.name: m.importance() for m in models if isinstance(m, LightGbmPoisson)}
    report = {
        "status": "SYNTHETIC. Forecast accuracy is NOT a research claim: the training data comes "
        "from the project's own incident generator, so the 'informed' model recovers its "
        "intensity formula by construction. The claim is only how dispatch performance depends "
        "on forecast quality (phase 10).",
        "panel": str((args.panel or panel_path(cfg)).name),
        "scenario": cfg.history_scenario,
        "split_rows": {k: len(v) for k, v in parts.items()},
        "test_weeks": cfg.split.test_weeks,
        "feature_sets": cfg.feature_sets,
        "metrics": metrics,
        "lightgbm_gain_importance": importances,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_synthetic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / "calibration_synthetic.csv", index=False, lineterminator="\n"
    )
    configs = [CONFIG_DIR / "forecasting.yaml", CONFIG_DIR / "incidents.yaml"]
    seeds = {"forecast_training_seed_base": base.random.forecast_training_seed_base}
    for path in (OUT / "metrics_synthetic.json", OUT / "calibration_synthetic.csv"):
        write_meta(path, configs=configs, synthetic=True, seeds=seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
