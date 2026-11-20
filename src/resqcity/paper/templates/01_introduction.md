# Section 1: Introduction

## 1.1 Problem Statement and Context
Emergency medical, police, and firefighting services are essential public goods whose operational
efficacy directly determines trauma survival and societal resilience [CITE: emergency medical response systems].
In high-density South Asian metropolitan environments such as Pune, India (population > 3.5 million;
area 516 sq km across 58 municipal electoral wards), emergency services face acute systemic challenges:
congested arterial roadways, fragmented communication channels, and a historical reliance on reactive
civilian distress calls via the national emergency Dial-112 infrastructure [CITE: Pune traffic and urban mobility].

Currently, emergency responses operate under an uncoordinated, agency-by-agency paradigm. When a severe road accident
occurs, a civilian witness dials 112, initiating call intake, location triangulation, and triage. If multiple services
are required (e.g. an ambulance for trauma victims and police for traffic clearance), the request is relayed
sequentially between disparate dispatch centres. This introduces a substantial hand-off delay, compounding arrival
latencies and leaving trauma casualties unattended during the critical "Golden Hour" of trauma care [CITE: trauma golden hour].

## 1.2 Automated Surveillance and Smart-City Sensing
Concurrently, modern municipal smart-city projects have deployed extensive closed-circuit television (CCTV)
surveillance networks at critical road junctions. In Pune, hundreds of traffic surveillance cameras operate under
the Pune Municipal Corporation (PMC) and Smart City Development Corporation. Automated computer vision models
(e.g. YOLO, ByteTrack) offer the theoretical potential to detect vehicular collisions within seconds of occurrence
[CITE: video anomaly detection].

However, translating raw CCTV detections into actionable emergency dispatch presents significant operational hurdles:
1. **False Alarm Saturation:** Edge vision models suffer non-trivial false-alarm rates under variable lighting,
   monsoon rains, and shadow occlusion. Naively dispatching emergency fleets to unverified computer vision detections
   would rapidly deplete emergency capacity [CITE: false alarms in automated surveillance].
2. **Partial Camera Coverage:** Municipal CCTV covers only a fraction of road junctions (~4% of high-risk nodes per
   Maximal Covering Location models), meaning the majority of city accidents remain camera-blind.
3. **Multi-Agency Synchronization:** Road trauma survival depends on the arrival of the *last* essential service
   (e.g., extrication or advanced life support). Optimizing individual agency arrivals independently fails to
   synchronize joint operations.

## 1.3 Research Questions
This study designs, implements, and evaluates **ResQCity**, a confidence-aware, joint multi-agency emergency
decision-support digital twin on the Pune road network, addressing four core research questions:

- **RQ1:** How much earlier is a road incident detected from automated CCTV monitoring than from civilian 112 emergency calls?
- **RQ2:** Does joint, synchronized multi-agency dispatch reduce the time until the *last* required service arrives,
  compared to independent per-agency dispatch?
- **RQ3:** Which confidence thresholds best trade off false-alarm unit costs against response delay?
- **RQ4:** What is the price of fairness when enforcing spatial equity across all 58 municipal wards, and does
  joint dispatch preserve equity under non-inferiority testing on ward-level Gini coefficients?

## 1.4 Novel Contributions
- **C1: Joint Multi-Agency Dispatch MILP:** An integer programming formulation with severity-dependent unit requirements,
  minimizing the arrival of the last critical service and incorporating police green corridor speedups.
- **C2: Confidence-Aware Tiered Policy:** A three-tier dispatch policy operating on fused Bayesian posteriors
  (Immediate Dispatch / Police Verification with Pre-Alert / Operator Review with Timeout).
- **C3: Fairness-Constrained Pre-positioning:** Rolling multi-period idle vehicle redeployment driven by forecasted ward-hour
  accident risk, bounded by an explicit spatial equity constraint.
- **C4: Open Discrete-Event Digital Twin:** A high-fidelity SimPy simulation grounded on OpenStreetMap Pune graph topology,
  calibrated against empirical MoRTH accident rates and Dial-112 volume.
