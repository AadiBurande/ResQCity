---
name: run-experiment
description: Design, run and analyse simulation experiments for the research questions (policies B0–FULL, ablations, threshold sweeps, sensitivity, price of fairness). Use when adding an experiment, running experiments, computing statistics or producing result tables and figures.
---

# Running a ResQCity experiment

## Design (write before running)
Create `experiments/configs/<exp_name>.yaml` containing:
- `research_question`, `hypothesis` (H0 and H1 in words), `primary_metric`, `secondary_metrics`.
- `scenarios`, `policies`, factor levels, `replications` (>= 30), `seed_base`.
- `horizon` (simulated days) and `warmup` (discarded at start).
- `analysis`: test family (paired), multiple-comparison correction (Holm), alpha (0.05).
Ask the `research-auditor` subagent to review the design file before running.

## Run
- Use common random numbers: replication r uses the same seeds for every policy.
- Run with `python -m experiments.run --config experiments/configs/<exp_name>.yaml`; it must be
  resumable and write one parquet per (scenario, policy, replication) to `results/raw/<exp_name>/`.
- Record git commit hash, config hash, package versions and machine spec in `results/raw/<exp_name>/meta.json`.

## Analyse
- Aggregate per replication first, then compare policies on replication-level values (paired).
- Check normality of paired differences (Shapiro–Wilk). Normal -> paired t-test; else Wilcoxon
  signed-rank. Apply Holm correction across the comparisons in the experiment.
- Report effect size (mean difference with 95% CI; Cohen's d_z or rank-biserial r) and, for
  percentile metrics (p90), a bootstrap 95% CI.
- Write `results/stats/<exp_name>.csv` and regenerate `results/summary.md` via
  `python -m experiments.summarize` (never edit summary.md by hand).

## Figures
- One script per figure in `experiments/figures/`, reading only from `results/`.
- Colour-blind-safe palette, labelled axes with units, CI bands or error bars, saved as PNG and PDF.

## Report honestly
- If a hypothesis is not supported, say so. Report all pre-registered metrics, not only good ones.
- Note when results depend on synthetic or UNVERIFIED parameters.
