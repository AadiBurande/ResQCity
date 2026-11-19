# ResQCity — Experiment Summary

> Generated 2026-11-19 21:26 UTC by `python -m resqcity.experiments.summarize`.
> Every number is read from `results/experiments/`; none is typed.

---

## Primary outcome: time to last required arrival

### Normal scenario

| Policy | Incidents | Primary mean (s) | Primary p90 (s) | ≤8 min (%) | ≤15 min (%) | Gini | Survival |
|--------|-----------|------------------|-----------------|----------|-------------|------|----------|
| B0 | 68875 | 1838.1 | 3128.6 | 0.0 | 0.0 | 0.2890 | 0.6805 |
| B1 | 68875 | 1786.3 | 3077.0 | 0.0 | 0.0 | 0.2963 | 0.6855 |
| B2 | 68875 | 1802.6 | 3143.1 | 0.0 | 0.0 | 0.2996 | 0.6851 |
| B3 | 105 | 1778.1 | 3148.0 | 3.8 | 21.9 | 0.3079 | 0.6888 |
| FULL | 105 | 1779.0 | 3148.0 | 3.8 | 21.9 | 0.3097 | 0.6892 |

### Monsoon scenario

| Policy | Incidents | Primary mean (s) | Primary p90 (s) | ≤8 min (%) | ≤15 min (%) | Gini | Survival |
|--------|-----------|------------------|-----------------|----------|-------------|------|----------|
| B0 | — | — | — | — | — | — | — |
| B1 | — | — | — | — | — | — | — |
| B2 | — | — | — | — | — | — | — |
| B3 | — | — | — | — | — | — | — |
| FULL | — | — | — | — | — | — | — |

### Ganeshotsav scenario

| Policy | Incidents | Primary mean (s) | Primary p90 (s) | ≤8 min (%) | ≤15 min (%) | Gini | Survival |
|--------|-----------|------------------|-----------------|----------|-------------|------|----------|
| B0 | — | — | — | — | — | — | — |
| B1 | — | — | — | — | — | — | — |
| B2 | — | — | — | — | — | — | — |
| B3 | — | — | — | — | — | — | — |
| FULL | — | — | — | — | — | — | — |

---

## Paired comparisons (Holm-Bonferroni corrected)

### Paired comparisons (normal)

| Comparison | Test | Statistic | p | p (adj) | Effect | 95% CI | Sig? |
|------------|------|-----------|---|---------|--------|--------|------|
| B0 vs B1 | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [32.7, 74.3] | — |
| B0 vs B2 | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [15.9, 74.2] | — |
| B0 vs B3 | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [34.3, 95.4] | — |
| B0 vs FULL | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [28.0, 95.4] | — |
| B1 vs B2 | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [-30.1, -0.1] | — |
| B2 vs B3 | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [16.2, 36.2] | — |
| B2 vs FULL | wilcoxon | 0.000 | 0.2500 | 1.0000 | rank_biserial = 1.000 | [10.0, 40.4] | — |
| B1 vs B3 | wilcoxon | 1.000 | 0.5000 | 1.0000 | rank_biserial = 0.667 | [-13.9, 21.0] | — |
| B1 vs FULL | wilcoxon | 1.000 | 0.5000 | 1.0000 | rank_biserial = 0.667 | [-20.1, 23.7] | — |
| B3 vs FULL | wilcoxon | 1.000 | 1.0000 | 1.0000 | rank_biserial = 0.333 | [-6.2, 4.3] | — |

_No paired tests found for monsoon._

_No paired tests found for ganeshotsav._

---

## Equity (RQ4): ward-level Gini non-inferiority

### Equity non-inferiority (normal)

- **H0**: Gini(FULL) - Gini(B0) >= delta = 0.02
- **H1**: Gini(FULL) - Gini(B0) < delta (non-inferior)
- Mean Gini(FULL) = 0.2927, Mean Gini(B0) = 0.2672
- Diff = 0.0255, t = 1.205, p = 0.8242
- **Verdict: ✗ NOT ESTABLISHED**

_No equity test found for monsoon._

_No equity test found for ganeshotsav._

---

## Research question verdicts

_The verdicts below are populated by the statistical tests above. Each refers to the_
_specific table rows and test results that support it._

### RQ1: Does CCTV detect incidents earlier than 112 calls?

See `detection_delay_mean_s` across policies. The camera-only detection delay is bounded by the measured detector (D-029).

### RQ2: Does joint dispatch reduce time-to-full-response?

Compare B0 vs FULL on the primary outcome (paired tests above). The B0 hand-off delay is UNVERIFIED; check whether the result holds at hand-off = 0.

### RQ3: What are the optimal confidence thresholds?

See the sensitivity sweep results in `results/experiments/sensitivity/`.

### RQ4: Does FULL preserve equity across wards?

See the equity non-inferiority tests above (δ = 0.02 on ward-level Gini).
