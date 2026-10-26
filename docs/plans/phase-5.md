Status: DONE (2026-10-26). Approved by the user on 2026-10-26 before any Phase 5 code; results
and findings in D-030. NOTE: this file was committed late, after the implementation - the plan
was approved in session but not copied into the repo at the time, contrary to CLAUDE.md.

# Phase 5 plan: ward-hour demand forecasting

## Context
Phases 1–4 are built. Phase 6 pre-positioning (model C3) and the Phase 10 sensitivity sweeps both
need one thing from Phase 5: a callable `predict(ward, t)` for expected incident demand, plus a
way to vary how good that forecast is.

The honest difficulty, which shapes the whole design: **the training data is synthetic and its
intensity is a formula I wrote** (base rate × hour factor × weekday factor × rain multiplier ×
ward hotspot share × festival multiplier, `src/resqcity/sim/generator.py`). A model given those
same inputs recovers the formula almost exactly. So forecast *accuracy* is not a research claim
here, and the master plan already records this (§5, risk row for Phase 5). The research claim is
only how dispatch performance depends on forecast quality.

Measured facts that drove the design (my own runs today, on the committed generator):
- Generation is cheap: **0.5 s per 90 simulated days**, so a multi-year history costs seconds.
- The panel is very sparse: **0.41 % of ward-hours** contain an incident (9.3 % of ward-days),
  ~2,090 incidents per simulated year over 58 wards. MAE is therefore near-useless as a metric
  (predicting zero everywhere scores well); Poisson deviance, calibration and top-k ward hit rate
  carry the information.
- Type mix: accident 42 %, fire_smoke 57 %, crowd + other 2 %.

Your decisions (2026-10-26):
- **Fit both an "informed" and a "realistic" model, side by side.** The gap between them is a
  measured forecast-quality range for Phase 10, and the circular model is labelled as such.
- **Predict per incident type**, so pre-positioning can place ambulances, police and fire tenders
  against different demand surfaces.

## Approach (vertical slices, each with tests; commit at the end of each)

**0. Close Phase 4 first.** Fold the running 30-week check into a new decision entry D-029, mark
`docs/plans/phase-4.md` DONE, and get a `research-auditor` review. Phase 5 code starts after that.

**1. Dependency and history scenario.**
- `uv add lightgbm` (recorded in `docs/decisions.md`; MIT licence, add to `docs/data-sources.md`).
- A small generated config `configs/scenarios/history_3y.yaml` (header "GENERATED", like
  `configs/cameras.yaml`), written by `src/resqcity/sim/make_history.py` from a compact monthly
  rainfall shape and the Ganeshotsav dates. 3 years, rain `day_factors` of length 365 (the
  existing `ScenarioState` cycles it) and festival windows at each year's day offsets — **no
  change to `geo/scenario.py` is needed**. I will try to cite IMD Pune monthly rainfall normals
  for the seasonal shape; if I cannot confirm them at the source, the value is marked UNVERIFIED.

**2. Panel builder (in `sim/`, because it reads ground truth).**
- `src/resqcity/sim/panel.py`: generator → incidents → ward × hour × type counts, plus the
  columns each model is allowed to see. Uses `IncidentGenerator`, `ScenarioState`,
  `hotspot_weights` and `common/rng.stream`; seeded from `random.forecast_training_seed_base`
  (already in `configs/base.yaml`, deliberately disjoint from the evaluation seeds).
- Writes `data/processed/synthetic/panel_<scenario>_<years>y.parquet` with a provenance sidecar.
- Also writes the **true intensity** per ward-hour-type (from `IncidentGenerator.ward_hour_rates`,
  extended per type) as the oracle column — used for the oracle forecaster and as the calibration
  reference, never as a model feature.

**3. Feature sets (`configs/forecasting.yaml`, typed in `src/resqcity/forecasting/config.py`).**

| Feature | Realistic model | Informed model |
|---|---|---|
| hour, weekday, month, incident type | yes | yes |
| ward identity (categorical) | yes | yes |
| festival window flag | yes | yes |
| lagged counts: same ward 1 h / 24 h / 7 d, citywide 24 h | yes | yes |
| rain | **noisy forecast** (true + lognormal error, σ in config) | true `rain_mm_h` |
| ward hotspot mass (the generator's own ward share) | no | yes |

The realistic model must learn each ward's base rate from its own history; the informed model is
handed it. Both are honestly labelled in the results file.

**4. Models (`src/resqcity/forecasting/`), all behind one `Forecaster` protocol with
`predict(ward_id, ts, type) -> expected count`:**
- `Climatology`: mean count per (ward, hour-of-week, type) from the training years — the baseline.
- `LightGBMPoisson`: Poisson objective, fixed seed, early stopping on a validation slice taken
  from the *training* years only. Fitted twice (realistic, informed).
- `UniformRate`: citywide mean spread evenly — the floor, for Phase 10's "no information" case.
- `NoisyForecaster`: wraps any of the above and multiplies by lognormal noise (σ = the Phase 10
  knob, `common/rng.stream` so sweeps use common random numbers).
- The **oracle** (true intensity) lives in `src/resqcity/sim/oracle_forecast.py`, not in
  `forecasting/`, because `tests/test_architecture.py` forbids `forecasting → sim`.

**5. Evaluation (`src/resqcity/forecasting/evaluate.py`).**
- Time-based split: first 2 years train, last year test; no shuffling.
- Metrics per model, at ward-hour **and** ward-day (sparsity): Poisson deviance, MAE (reported
  with the caveat), calibration table (predicted decile vs observed mean), top-k ward hit rate
  (k = 3, 5, 10) and a reliability plot input.
- Output `results/forecasting/metrics_synthetic.json` + `calibration_synthetic.csv`, with
  provenance sidecars and a `status` line saying accuracy is not a research claim.
- `notebooks/05_forecasting.ipynb`: calibration and top-k plots (loads the dataviz skill before
  any chart code, as in Phase 4).

**6. Persistence and determinism.**
- Save to `data/models/forecast/<name>/`: LightGBM booster `.txt`, feature list, config hash.
- A test reloads the saved model and asserts predictions are **bit-identical** on a fixed sample
  (the acceptance criterion in the brief).

**7. Wrap-up.** `make forecast` target in `tools/tasks.py` + `Makefile`; `docs/decisions.md` D-030
(feature split, targets, why accuracy is not a claim); `docs/limitations.md` rows (circularity,
synthetic rain, sparsity, rain assumed known at short horizon); `research-auditor` review; commit.

## Critical files
- **New:** `src/resqcity/forecasting/{__init__,config,models,evaluate}.py`,
  `src/resqcity/sim/{panel,make_history,oracle_forecast}.py`, `configs/forecasting.yaml`,
  `configs/scenarios/history_3y.yaml` (generated), `notebooks/05_forecasting.ipynb`,
  `tests/forecasting/*`, `tests/sim/test_panel.py`.
- **Changed:** `pyproject.toml` (lightgbm), `tools/tasks.py`, `Makefile`, `docs/decisions.md`,
  `docs/limitations.md`, `docs/data-sources.md`, `README.md`.
- **Reused:** `sim.generator.{IncidentGenerator,hotspot_weights,load_incidents_config}`,
  `geo.scenario.{ScenarioState,load_scenario}`, `common.rng.stream`,
  `common.config.{SourcedValue,load_model,find_unverified}`, `common.provenance.write_meta`.

## Verification
- **Unit tests** (`uv run pytest`): panel counts match the generated incidents exactly; panel is
  reproducible from the seed; lagged features never use future rows (a deliberate leakage test);
  climatology equals a hand-computed mean on a toy panel; a saved model reloads to identical
  predictions; the noise knob is reproducible and has the configured median; the oracle predicts
  the generator's own intensity; `find_unverified` flags every assumed value.
- **Lint:** `uv run ruff check`.
- **Runs:** `make forecast` writes `results/forecasting/*.json` with `dirty: false` sidecars; the
  notebook executes without errors; `uv run pytest tests/test_architecture.py` still passes
  (`forecasting` must not import `sim`).
- **Review:** `research-auditor` on the Phase 5 claims before marking the phase done — with
  specific attention to the circularity framing.

## Deviations during implementation
- The evaluation driver is `src/resqcity/sim/forecast_eval.py`, not `forecasting/evaluate.py`:
  it needs the oracle and the panel, and `tests/test_architecture.py` forbids `forecasting ->
  sim`. The pure metrics stayed in `forecasting/metrics.py`. The architecture test caught this.
- Two baselines had to be repaired before they were fair: the per-cell climatology was a
  strawman at this sparsity, and the top-k metric was arbitrary under tied predictions (D-030
  item 6).
- The IMD Pune rainfall normals could not be confirmed at the source, so the seasonal profile is
  UNVERIFIED rather than cited (L28).
