# Section 2: Methods

## 2.1 Road Network and Spatial Environment
The simulation environment is constructed on the drivable road graph of Pune Municipal Corporation,
extracted from OpenStreetMap via OSMnx [CITE: OSMnx]. The network comprises verified intersections,
arterial thoroughfares, river crossings across the Mula and Mutha rivers, and 58 official PMC municipal wards.
Emergency facilities are anchored to real geographic facilities:
- 15 Fire Stations (OSM amenity=fire_station)
- 40 Police Stations (OSM amenity=police)
- 38 Major Hospitals with Trauma / Emergency facilities (OSM amenity=hospital)
Fleet sizes were established via analytical Erlang-C dimensioning to achieve target queueing delay bounds
under baseline emergency workloads.

## 2.2 Accident Generation and Detection Process
Incident generation follows an inhomogeneous spatio-temporal Poisson point process across graph nodes,
weighted by historical ward risk and time-of-day traffic distributions calibrated against official
Ministry of Road Transport and Highways (MoRTH 2022) statistics [CITE: MoRTH report].
A Maximal Covering Location Problem (MCLP) places 40 CCTV camera nodes at top risk intersections.
The automated vision pipeline models calibrated performance metrics from YOLO+ByteTrack benchmarks
(TAD, CADP, MADP) including line-of-sight self-masking and weather attenuation (effective detection probability = 0.31 per D-029).

## 2.3 Dial-112 Call Simulation and Bayesian Evidence Fusion
Civilian 112 calls are simulated with log-normal reporting latencies and spatial triangulation noise.
Incoming camera alarms and calls are associated spatio-temporally and ingested by a recursive Bayesian
fusion engine. The engine computes log-odds updates yielding a continuous posterior probability:
$$P(\\text{incident} \\mid E_1, \\dots, E_k) = \\frac{P_0 \\prod \\lambda_i}{1 - P_0 + P_0 \\prod \\lambda_i}$$
where $\\lambda_i$ represents the likelihood ratio of each evidence piece.

## 2.4 Operations Research Models and Dispatch Policies
Five distinct dispatch policies are evaluated:
- **B0 (Status Quo Baseline):** Call-only reporting. Independent nearest-unit dispatch per agency.
  Sequential inter-agency hand-off delay ($T_{\\text{handoff}} = 120\\text{s}$).
- **B1 (CCTV + Calls, Independent):** Both camera and call inputs available. Nearest available unit assigned
  immediately without coordination or hand-off latency.
- **B2 (Joint MILP):** Solves the joint multi-agency dispatch formulation via OR-Tools CP-SAT,
  minimizing $\\max_{s} t_{i,s}^{\\text{arrive}}$ across all required services.
- **B3 (Confidence-Aware Tiering):** Incorporates three posterior thresholds:
  - $P \\ge \\tau_{\\text{high}}$: Immediate multi-agency dispatch.
  - $\\tau_{\\text{low}} \\le P < \\tau_{\\text{high}}$: Dispatch police verification unit; pre-alert ambulance and fire.
  - $P < \\tau_{\\text{low}}$: Operator review queue with timeout $T_{\\text{review}} = 120\\text{s}$.
- **FULL (Proposed System):** B3 combined with dynamic police-assisted green corridor routing
  ($\\rho_{\\text{green}} = 0.8$) and fairness-constrained pre-positioning of idle units based on LightGBM ward risk forecasts.

## 2.5 Experimental Protocol and Statistical Rigour
- **Replications:** 7-day simulated horizon preceded by a 12-hour warm-up period (excluded from metrics).
- **Common Random Numbers (CRN):** All policies within each replication witness the identical sequence of incident
  locations, onset times, severity classes, and civilian calls.
- **Background Workload:** Independent Poisson process representing non-traffic emergency calls, calibrated to
  Pune Dial-112 empirical volume (23,887 calls/month), maintaining ~41% baseline fleet utilisation.
- **Statistical Testing:** Shapiro-Wilk normality tests on paired replication differences, followed by paired
  Student's t-test (if normal) or Wilcoxon signed-rank test. Family-Wise Error Rate controlled via Holm-Bonferroni correction.
- **Spatial Equity Non-Inferiority:** One-sided non-inferiority test on ward-level p90 Gini coefficient
  with non-inferiority margin $\\delta = 0.02$.
