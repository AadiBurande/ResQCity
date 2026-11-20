# ResQCity — Research Methodology

Status: final · Date: 2026-11-20 · Phase 11 · Decisions: docs/decisions.md

---

## 1. Overview and Research Questions

ResQCity is an operational research study and decision-support prototype evaluating whether
**automated CCTV accident detection combined with confidence-aware Bayesian fusion and joint multi-agency dispatch**
reduces emergency response time across Pune's road network, without inducing spatial inequity across municipal wards.

The investigation is governed by four formal research questions:

- **RQ1 (Detection Lead Time):** How much earlier is a road incident detected through automated computer vision
  monitoring of fixed CCTV streams compared to civilian 112 emergency calls?
- **RQ2 (Joint Dispatch Efficiency):** Does joint, synchronized multi-agency dispatch (coordinating ambulance,
  police, and fire brigades simultaneously via integer programming) reduce the time until the *last* required service
  arrives on scene, compared with sequential, agency-by-agency dispatch?
- **RQ3 (Confidence-Aware Tiering):** Which posterior probability thresholds ($\tau_{\text{low}}, \tau_{\text{high}}$)
  and operator review timeouts ($T_{\text{review}}$) optimize the trade-off between false-alarm unit waste and delayed dispatch?
- **RQ4 (Spatial Equity):** What is the price of fairness—measured as the change in system-wide mean response time—when
  enforcing minimum coverage constraints across all 58 municipal wards, and does joint dispatch preserve equity
  under non-inferiority testing on ward-level Gini coefficients?

---

## 2. Hypotheses and Variable Operationalisation

### 2.1 Hypotheses

| Hypothesis | Null Hypothesis ($H_0$) | Alternative Hypothesis ($H_1$) | Evaluation Metric |
|---|---|---|---|
| **H1 (Detection)** | $\mu_{\text{CCTV}} \ge \mu_{\text{call}}$ | $\mu_{\text{CCTV}} < \mu_{\text{call}}$ | Detection delay from incident onset (seconds) |
| **H2 (Synchronization)** | $\mu_{\text{FULL}} \ge \mu_{\text{B0}}$ | $\mu_{\text{FULL}} < \mu_{\text{B0}}$ | Primary outcome: Time to last required arrival |
| **H3 (Threshold Trade-off)** | Tiered policy does not lower false-alarm unit-hours vs. immediate dispatch | Tiered policy reduces false-alarm unit-hours while bounding delay | False-alarm unit-hours vs. secondary dispatch delay |
| **H4 (Fairness & Equity)** | $\text{Gini}(\text{FULL}) - \text{Gini}(\text{B0}) \ge \delta$ | $\text{Gini}(\text{FULL}) - \text{Gini}(\text{B0}) < \delta$ ($\delta = 0.02$) | Ward-level p90 response time Gini coefficient |

### 2.2 Variables

- **Independent Variables:**
  - **Dispatch Policy:** Baseline 0 (B0: status quo call-only, sequential dispatch), Baseline 1 (B1: CCTV + calls, independent dispatch),
    Baseline 2 (B2: B1 + joint MILP without confidence tiers), Baseline 3 (B3: B2 + confidence-aware dispatch tiers),
    and FULL (B3 + fairness-constrained pre-positioning + dynamic green corridor routing).
  - **Operational Scenario:** `normal` (baseline dry traffic), `monsoon` (impassable river-adjacent roads, reduced travel speeds),
    and `ganeshotsav` (processional choke points, festival crowd density).
- **Dependent Variables:**
  - *Primary Outcome:* Time to last required arrival ($T_{\text{primary}} = \max_{s \in \mathcal{S}_i} t_{i,s}^{\text{arrive}} - t_i^{\text{onset}}$).
  - *Secondary Outcomes:* Detection delay, detection-to-dispatch decision latency, first-to-last arrival gap,
    fraction of incidents served within 8 minutes ($\le 8$ min) and 15 minutes ($\le 15$ min),
    per-agency fleet utilisation ($\rho_{\text{amb}}, \rho_{\text{pol}}, \rho_{\text{fire}}$),
    false-alarm unit-hours, expected survival probability (illustrative secondary transform).
  - *Equity Outcome:* Ward-level Gini coefficient of 90th-percentile response times across the 58 electoral wards.
- **Controlled / Nuisance Variables:**
  - Pune drivable road network topology (OSMnx extract, 58 wards, facility locations).
  - Background civilian demand workload (Poisson arrival rate calibrated to 23,887 monthly Dial-112 calls).
  - Fleet sizing (ambulances, police cruisers, fire engines sized via Erlang-C model).

---

## 3. Simulation Testbed and Experimental Design

### 3.1 Digital Twin Architecture (SimPy)

Experiments are executed within a discrete-event simulation model (`src/resqcity/sim/`) built on `simpy.Environment`:

1. **Incident Generation:** Synthetic spatio-temporal Poisson point process across Pune's road nodes, weighted by ward risk scores
   and calibrated to Ministry of Road Transport and Highways (MoRTH) accident statistics.
2. **Detection & Masking:** Camera detection is simulated from the calibrated statistical performance of YOLO+ByteTrack models
   evaluated on public accident benchmarks (TAD, CADP, MADP), incorporating spatial coverage and line-of-sight self-masking (effective detection ~0.31 per D-029).
3. **Call Simulation & Bayesian Fusion:** Dial-112 calls generated with log-normal reporting delays and positional noise.
   Calls and camera events are associated spatio-temporally and fused via recursive log-odds Bayesian updates into an incident posterior $P(\text{incident} \mid E)$.
4. **Operations Research Dispatch Solvers:** OR-Tools CP-SAT integer programming model (`src/resqcity/optimization/joint_dispatch.py`)
   optimizes unit-to-incident assignments minimizing the latest critical arrival time, incorporating police escort speedups ($\rho_{\text{green}} < 1.0$).
5. **Fairness Pre-positioning:** Rolling multi-period assignment problem shifting idle emergency units toward forecasted high-risk peripheral wards
   subject to a maximum allowable travel threshold and ward coverage guarantees.
6. **Background Workload Process:** Out-of-scope civilian calls drawn as an independent Poisson stream occupying emergency units for exponentially
   distributed service durations, maintaining realistic background fleet saturation (~41% utilisation per D-032).

### 3.2 Common Random Numbers (CRN)

To eliminate extraneous Monte Carlo variance between policy comparisons, the experiment grid enforces strict **Common Random Numbers (CRN)**.
For any given replication index $r \in \{0, \dots, R-1\}$ and scenario $s$:
$$\text{Seed}(s, r) = \text{seed\_base} + 1000 \cdot s_{\text{idx}} + r$$
Under this formulation, the exact same synthetic incident schedule (onset times, locations, vehicle classes, injury severities,
and camera field-of-view intersections) and civilian call arrival stream are presented identically to B0, B1, B2, B3, and FULL.
Differences in outcomes are strictly attributable to policy interventions.

---

## 4. Calibration and Model Validation

### 4.1 Queueing and Fleet Calibration (Erlang-C)

Before experimental runs, queueing dynamics were cross-validated against analytical Erlang-C formulations (`sim/validate_queue.py`).
In isolated test environments with $c$ homogeneous servers and Poisson arrivals at rate $\lambda$ with mean service time $1/\mu$:
- Measured mean waiting time in queue matched analytical $W_q = \frac{C(c, \lambda/\mu)}{c\mu - \lambda}$ within 2.4% relative error.
- Measured server utilisation matched nominal traffic intensity $\rho = \lambda / (c\mu)$ with $p > 0.95$ under Kolmogorov-Smirnov testing.

### 4.2 Pune Road Network and Empirical Calibration

- **Road Graph:** Extracted from OpenStreetMap for Pune Municipal Corporation (PMC) boundaries, comprising drivable vehicular ways
  simplified to major intersections. Edge speeds calibrated against peak and off-peak empirical probe estimates.
- **Incident Frequency:** Ground truth accident frequency scaled to match Pune's empirical proportion of Maharashtra state road casualties
  reported in MoRTH Road Accidents in India (2022).
- **Background Demand:** Calibrated to the verified July 2026 Pune Dial-112 volume (771 calls/day) to maintain realistic fleet busy probabilities.

---

## 5. Statistical Analysis Plan

### 5.1 Paired Hypothesis Testing

Given the paired structure guaranteed by CRN:
1. For each replication $r$, calculate the paired difference:
   $$\Delta_r = T_{\text{primary}, r}(\text{Policy A}) - T_{\text{primary}, r}(\text{Policy B})$$
2. **Normality Assessment:** Assess $\Delta_r$ via the Shapiro-Wilk test at $\alpha = 0.05$.
3. **Hypothesis Test:**
   - If $\Delta_r$ is normally distributed, apply the two-sided **paired Student's t-test**.
   - If non-normal, apply the **Wilcoxon signed-rank test**.
4. **Multiple Comparison Adjustment:** Control the Family-Wise Error Rate (FWER) across the 10 pairwise comparisons
   among the 5 policies using the step-down **Holm-Bonferroni correction**.

### 5.2 Effect Size and Confidence Intervals

- For normally distributed differences: Cohen's $d = \frac{\bar{\Delta}}{s_{\Delta}}$.
- For Wilcoxon signed-rank tests: Matched-pairs rank-biserial correlation $r = \frac{W^+ - W^-}{W^+ + W^-}$.
- Confidence intervals: 95% non-parametric bootstrap confidence intervals (1,000 resamples) computed for all median and p90 metrics.

### 5.3 Spatial Equity Non-Inferiority Testing (RQ4)

To ensure that optimizing global response times does not sacrifice peripheral or vulnerable wards:
- Let $G_{\text{FULL}, r}$ and $G_{\text{B0}, r}$ denote the Gini coefficient of 90th-percentile response times across the 58 wards in replication $r$.
- Non-inferiority margin: $\delta = 0.02$.
- Hypotheses:
  $$H_0: \mathbb{E}[G_{\text{FULL}} - G_{\text{B0}}] \ge \delta \quad \text{vs.} \quad H_1: \mathbb{E}[G_{\text{FULL}} - G_{\text{B0}}] < \delta$$
- Tested via one-sided paired Student's t-test on $(G_{\text{FULL}, r} - G_{\text{B0}, r} - \delta)$ at significance level $\alpha = 0.05$.

---

## 6. Threats to Validity and Mitigation

### 6.1 Internal Validity
- *Warm-Up Bias:* Simulations start with idle fleets, which would artificially deflate initial response times.
  *Mitigation:* An initial 12-hour simulation warm-up period is executed before every 7-day run, and all incidents during warm-up are excluded from metric accumulation.
- *Detector Over-Fitting / Leakage:* Vision evaluation could leak temporal frames.
  *Mitigation:* Video-level (never frame-level) splits applied on benchmark datasets. Simulation uses statistical confusion profiles rather than live inference.
- *Self-Masking:* Severe accidents cause camera obstruction or congestion.
  *Mitigation:* Modelled with an empirical detection probability ceiling (0.31) reflecting true roadside occlusion.

### 6.2 External Validity
- *Topographical Specificity:* Network topology, river bridges (Mula-Mutha), and ward geometry are specific to Pune.
  *Mitigation:* The optimization formulation and SimPy engine are parameterized by graph inputs and apply directly to any OSMnx-compatible city.
- *Monsoon & Event Generalisability:* Flooding and festival disruptions are modelled based on historical PMC flood choke points.

### 6.3 Construct Validity
- *Survival Probability as Outcome:* As noted in D-032, published trauma literature shows mixed, non-linear relationships between response time and mortality.
  *Mitigation:* All survival metrics are designated as secondary illustrative transforms (`configs/survival.yaml` marked `UNVERIFIED`).
  The primary outcome remains objective physical duration ($T_{\text{primary}}$).

---

## 7. Limitations

1. **Synthetic Demand Schedules:** Due to police data sensitivity, individual incident locations are synthesized via Poisson models rather than real emergency logs.
2. **Simulated RTSP Ingestion:** Fixed CCTV feeds are emulated via re-streamed public benchmark clips pinned to real Pune intersections due to lack of municipal API access.
3. **No Dynamic Traffic Light Preemption Hardware:** Green corridor speedups assume passive clearance and police escort rather than connected vehicle-to-infrastructure (V2I) signal actuation.
4. **No Real-Time CAD Integration:** Dispatch recommendations are evaluated in simulation and a local demonstration testbed rather than deployed inside an operational 112 emergency operations center.
