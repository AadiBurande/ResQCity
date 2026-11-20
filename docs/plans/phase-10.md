# Phase 10 — Experiments and statistics

Status: implemented · Date: 2026-11-20 · Decisions: docs/decisions.md

## Goal

A reproducible experiment framework that answers all four research questions with statistical
rigour, generates publication figures, and writes `results/summary.md` from data (never typed).

## Design

### Experiment matrix

The main grid crosses scenarios, policies and replications with **common random numbers**:

| Factor | Levels |
|---|---|
| Scenario | `normal`, `monsoon`, `ganeshotsav` |
| Policy | `B0`, `B1`, `B2`, `B3`, `FULL` |
| Replications | 30 (from `configs/sim.yaml`) |

Each (scenario, policy, rep) calls `sim.runner.run_replication` and writes one parquet file and
a summary JSON. CRN: for a given (scenario, rep), the same `Week` (incidents, alarms, calls) is
produced regardless of policy, so paired differences are exact.

### Ablation studies

FULL minus one mechanism at a time, to attribute any gain:

| Ablation | Config override |
|---|---|
| no joint MILP | B1 already tests this vs B2 |
| no confidence tiers | B2 already tests this vs B3 |
| no green corridor | `routing.green_corridor.rho = 1.0` |
| no pre-positioning | `preposition.every_minutes = 999999` (disabled) |
| no background workload | `background.enabled = false` + all targets 0 |

### Sensitivity sweeps

| Parameter | Values | Why |
|---|---|---|
| `b0.sequential_handoff_s` | 0, 60, 120, 300 | RQ2 depends on this UNVERIFIED parameter |
| `policy.tau_low` | 0.1, 0.2, 0.3, 0.5 | RQ3 threshold grid |
| `policy.tau_high` | 0.5, 0.7, 0.9 | RQ3 threshold grid |
| `policy.t_review_s` | 60, 120, 300 | RQ3 review window |
| `background.target_utilisation` (all) | 0.0, 0.2, 0.412, 0.6 | fleet stress |
| survival `tau_multipliers` | 0.5, 1.0, 2.0 | secondary outcome sensitivity |

### Metrics (per incident, then aggregated)

**Primary:** `time_to_last_required_arrival` (seconds from onset to arrival of the last required
agency — the metric the whole study is built on).

**Secondary:**
- `detection_delay`: first detection (camera or call) minus onset
- `detection_to_dispatch`: decision minus first evidence
- `response_time_{agency}`: per-agency arrival minus onset
- `first_to_last_gap`: last arrival minus first arrival
- `pct_within_8min` / `pct_within_15min`: share of incidents below standard
- `utilisation_{agency}`: fraction of unit-hours busy (after warmup)
- `false_alarm_unit_hours`: total unit-time wasted on false alarms
- `expected_survival`: secondary outcome from configs/survival.yaml
- `ward_gini`: Gini coefficient of ward-level p90 response times
- `ward_max_min_ratio`: max / min ward p90

### Statistical plan

1. **Paired differences per replication** (CRN guarantees pairing).
2. **Normality check**: Shapiro-Wilk on paired differences.
3. **Test**: paired t-test if normal, Wilcoxon signed-rank otherwise.
4. **Multiple comparisons**: Holm-Bonferroni family-wise correction across the 10 pairwise
   comparisons (5 choose 2).
5. **Effect sizes**: Cohen's d (or matched-rank biserial) + 95% bootstrap CIs on the difference.
6. **Equity (RQ4)**: one-sided non-inferiority test on ward-level Gini (δ = 0.02). FULL Gini
   must not exceed B0 Gini + δ at α = 0.05.

## Modules

| File | Responsibility |
|---|---|
| `src/resqcity/experiments/__init__.py` | Package init |
| `src/resqcity/experiments/run.py` | Main experiment runner: grid, ablations, sensitivity |
| `src/resqcity/experiments/metrics.py` | Compute per-incident and aggregate metrics from IncidentRows |
| `src/resqcity/experiments/stats.py` | Paired tests, Holm correction, bootstrap CIs, Gini |
| `src/resqcity/experiments/summarize.py` | Generate `results/summary.md` from stats CSVs |
| `src/resqcity/experiments/figures/` | One script per figure, colorblind-safe, PDF+PNG |

## Acceptance

- `make experiments` runs the grid and writes `results/experiments/`.
- `make figures` regenerates all publication plots.
- `results/summary.md` lists every hypothesis with test, statistic, p, effect size and a
  one-line plain-English reading.
- All numbers come from files, never typed.
