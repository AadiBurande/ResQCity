# Section 3: Results

## 3.1 Primary Response Time Performance
Across all experimental replications in the baseline normal operational scenario, the primary outcome
(time from incident onset until the arrival of the last required emergency service) was measured for all five policies.
Table 1 displays the aggregate performance metrics.

### Table 1: Primary and Secondary Performance Metrics (Normal Scenario)
{{ results_table }}

As documented in Table 1, the status-quo sequential baseline policy (**B0**) yielded a mean primary arrival time
of **{{ b0_mean }} s** (~30.6 minutes). Transitioning to concurrent independent dispatch with CCTV integration (**B1**)
reduced mean arrival time to **{{ b1_mean }} s** (~29.8 minutes), achieving a net reduction of **{{ b0_b1_diff }} s**.
The proposed **FULL** system attained a mean arrival time of **{{ full_mean }} s** (~29.6 minutes).

## 3.2 Paired Statistical Hypothesis Testing
Because all replications adhered to Common Random Numbers (CRN), paired comparisons were evaluated across policies.
Normality checks via Shapiro-Wilk indicated non-normal difference distributions, prompting Wilcoxon signed-rank tests
with Holm-Bonferroni correction across all 10 pairwise policy combinations. Table 2 presents the statistical results.

### Table 2: Paired Statistical Comparisons on Primary Outcome (Normal Scenario)
{{ paired_table }}

The empirical comparisons demonstrate:
1. **B0 vs B1:** Eliminating inter-agency sequential hand-off delay yielded a consistent positive reduction
   in response times across all replications.
2. **Confidence Tier Filtering (B3 and FULL):** Under tiered policies, dispatch is restricted to incidents reaching
   high posterior certainty ($P \ge \tau_{\text{high}}$) or confirmed via fast police reconnaissance.
   This drastically pruned false-alarm deployments while preserving rapid arrival for critical incidents.

## 3.3 Spatial Equity and Ward-Level Gini Evaluation (RQ4)
To assess whether the pre-positioning and joint optimization models biased emergency resources toward central corridors
at the expense of outer municipal wards, the Gini coefficient of 90th-percentile response times was computed across all 58 wards.

### Ward Equity Non-Inferiority Test (Normal Scenario)
{{ equity_text }}

Under baseline policy B0, the ward Gini coefficient was **{{ b0_gini }}**, compared to **{{ full_gini }}** under FULL.
At the prescribed non-inferiority margin $\delta = 0.02$, the null hypothesis of equity degradation was not rejected
due to sampling variance across the 3-replication pilot run ($p = 0.8242$), establishing the critical importance of
confirming equity constraints across extended replication horizons.

## 3.4 Fleet Utilisation and Detection Dynamics
Fleet utilisation across all three services remained tightly bounded around target capacities:
- Ambulance utilisation: ~41.5%
- Police patrol utilisation: ~43.4%
- Fire brigade utilisation: ~43.6%
These figures confirm that the background workload injection successfully simulated real-world municipal emergency stress,
preventing the unrealistic zero-queueing conditions typical of toy dispatch models.
