# Phase 11 — Documentation, Reproducibility and Paper Support

Status: implementing · Date: 2026-11-20 · Decisions: docs/decisions.md

## Goal

Provide rigorous academic documentation, reproducible experiment execution via `make reproduce`,
comprehensive methodology and ethics treatises, and automated IMRaD paper compilation with zero
hand-typed numbers (pulled directly from `results/experiments/` and `results/figures/`).

## Deliverables

### 1. Paper Generation Module (`src/resqcity/paper/`)
- `src/resqcity/paper/__init__.py`: Package init.
- `src/resqcity/paper/build.py`: Dynamic compiler that reads:
  - `results/experiments/summary_*.json` (per-policy aggregates, primary mean, p90, <=8m, survival)
  - `results/experiments/stats/paired_tests_*.csv` (Wilcoxon/t-test, p-values, Holm-adjusted, effect sizes)
  - `results/experiments/stats/equity_*.csv` (Gini non-inferiority test, t-statistic, p-value)
  - `results/figures/` (metadata sidecars and generated plot artifacts)
  Injects values into structured IMRaD sections and outputs:
  - `paper/01_introduction.md`
  - `paper/02_methods.md`
  - `paper/03_results.md`
  - `paper/04_discussion.md`
  - `paper/paper.md` (full assembled paper)
  - `paper/figures_list.md` (figure catalog with captions and provenance)
  Uses `[CITE: topic]` placeholders wherever external literature citations belong, preserving research integrity.

### 2. Research Methodology (`docs/methodology.md`)
- Detailed research questions (RQ1–RQ4) and formal hypotheses (H1–H4).
- Experimental variables: Independent (dispatch policy, weather/traffic scenario), Dependent (time-to-last-arrival, first-to-last arrival gap, detection delay, ward-level Gini, false alarm unit-hours, expected survival), Controlled (road network, fleet roster, facility locations).
- Discrete-event simulation design on Pune road network (SimPy engine, shortest path routing, queueing dynamics, background workload calibration against real Dial-112 data).
- Variance reduction: Common Random Numbers (CRN) across policies per replication.
- Calibration & model validation: Erlang-C consistency, MoRTH incident rate calibration.
- Statistical testing framework: Shapiro-Wilk normality tests, paired Wilcoxon signed-rank / paired t-test, Holm-Bonferroni FWER control, bootstrap 95% confidence intervals, non-inferiority test on ward Gini equity ($H_0: \text{Gini}_{FULL} - \text{Gini}_{B0} \ge 0.02$).
- Threats to validity:
  - Internal validity (confounding factors, warm-up exclusion, self-masking in detection)
  - External validity (generalisability of Pune findings to other urban topographies)
  - Construct validity (proxy metrics, survival curve limitations)
- Limitations: synthetic incident distribution, lack of direct production CAD/RMS dispatch API integration.

### 3. Ethics, Privacy & Governance (`docs/ethics.md`)
- Privacy by design: edge blurring of human faces and vehicle license plates (D-026).
- Compliance with India's Digital Personal Data Protection (DPDP) Act 2023 and DPDP Rules 2025.
- Ephemeral video processing: no centralized long-term recording or surveillance archival.
- Algorithmic fairness & spatial equity: fairness-constrained pre-positioning (C3) protecting peripheral wards from dispatch starvation.
- Human-in-the-loop governance: operator review timeouts, confirmation protocols, preventing unverified emergency denials.
- Dual-use hazards: preventing traffic surveillance repurposing for unauthorized tracking.

### 4. Reproducibility & Task Runner (`src/resqcity/tools/tasks.py`)
- Replace the `reproduce` stub in `tasks.py` with an automated orchestration pipeline:
  - Verifies processed data and configs.
  - Runs experiment suite (`experiments`).
  - Generates figures (`figures`).
  - Builds paper and outputs summary (`paper.build`).
  - Records execution metadata, environment, machine spec, and git commit.

### 5. Updated Project README (`README.md`)
- Architecture diagram (Mermaid flow of vision -> fusion -> optimization -> sim / API).
- Complete setup and quickstart steps via `uv`.
- Reproducibility guide (`make reproduce`).
- Research questions and findings summary.
- Ethics, privacy and synthetic data declarations.

### 6. Tests & Architectural Verification
- `tests/paper/test_paper.py`: Tests that `build.py` runs cleanly, injects numbers, verifies placeholders, and matches results data.
- Ensure `test_architecture.py` passes all module boundary constraints.
