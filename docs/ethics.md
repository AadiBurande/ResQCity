# ResQCity — Ethics, Privacy, and Algorithmic Governance

Status: final · Date: 2026-11-20 · Phase 11 · Decisions: docs/decisions.md

---

## 1. Ethical Foundation and Principles

Emergency response optimization intersects with critical human rights: public safety, bodily integrity,
personal privacy, and equal protection under civic governance. ResQCity is designed under five core ethical commitments:

1. **Beneficence & Harm Reduction:** The sole objective of automated detection and dispatch optimization
   is the reduction of critical arrival latency and trauma mortality.
2. **Privacy by Design:** Automated visual surveillance must strictly extract situational event metadata
   without capturing, storing, or indexing personally identifiable information (PII).
3. **Equitable Civic Allocation:** Mathematical optimization must not exacerbate historical disparities
   in emergency infrastructure access across affluent and marginalized municipal wards.
4. **Human Agency & Oversight:** Algorithmic systems must augment human dispatchers with explainable
   recommendations rather than executing autonomous, unreviewable operational decisions.
5. **Transparency & Open Science:** Public algorithms funded and researched for municipal benefit must be
   fully auditable, reproducible, and open to civic scrutiny.

---

## 2. Privacy Engineering & DPDP Act 2023 / DPDP Rules 2025 Compliance

### 2.1 Statutory Context: India's Digital Personal Data Protection Act

In accordance with India's **Digital Personal Data Protection (DPDP) Act, 2023** and the **DPDP Rules, 2025**,
automated computer vision deployed across municipal roadways must handle visual data as sensitive digital personal data
whenever identifiable individuals or vehicular identifiers (registration plates) enter camera fields of view.

### 2.2 Edge Redaction and Ephemeral Processing (Decision D-026)

- **On-Device Anonymisation:** Vision ingestion nodes apply automated blurring of all human faces and vehicle registration
  plates directly at the video capture buffer (using lightweight edge Haar/YOLO detectors) prior to feature classification.
- **Ephemeral Frame Buffering:** Raw video frames exist solely in volatile GPU/CPU RAM for temporal clip analysis
  (maximum buffer window: 16 seconds). No unredacted raw frames are written to persistent disk or transmitted across network interfaces.
- **Metadata-Only Egress:** The vision sub-system exports solely structured `Event` messages containing:
  - Timestamp, camera ID, intersection coordinates
  - Categorical incident type (`accident`, `fire_smoke`, `crowd`)
  - Calibrated confidence score and aggregate severity indicators (e.g., `num_vehicles`, `person_down`, `overturned`)
  - No facial embeddings, biometric markers, or alphanumeric license plate strings are ever constructed or stored.
- **Controlled Demo Mode:** In accordance with D-026, unblurred video review (`--privacy off`) is strictly restricted
  to local offline demonstration on public academic datasets (TAD, CADP, MADP) running on localhost (`127.0.0.1`),
  and is permanently disabled for any live or real-world RTSP stream ingest.

---

## 3. Algorithmic Fairness and Spatial Equity

### 3.1 The Risk of Utilitarian Centralization

Standard unconstrained emergency dispatch algorithms (e.g., unconstrained MILP or greedy nearest-unit dispatch)
exhibit a well-documented utilitarian bias: to minimize city-wide mean response time, algorithms concentrate
emergency resources around high-density central zones (such as Shivajinagar, Deccan, and Kothrud in Pune),
leaving peripheral, suburban, and lower-density wards (e.g., Hadapsar, Dhankawadi, Katraj, Wadgaon Sheri) severely underserved.
This results in a spatial inequality trap where marginalized wards experience systematically longer wait times.

### 3.2 Fairness-Constrained Pre-positioning (Contribution C3)

To safeguard equitable access:
- ResQCity introduces an explicit fairness constraint within the pre-positioning optimization formulation:
  $$\min \sum_{w \in \mathcal{W}} c_w d_w \quad \text{s.t.} \quad \text{coverage}_w \ge \alpha \cdot \text{baseline}_w \quad \forall w \in \mathcal{W}$$
- The policy enforces a strict upper bound on ward-level response latency disparities, evaluated using the Gini coefficient
  of 90th-percentile arrival times across all 58 electoral wards.
- A non-inferiority margin of $\delta = 0.02$ guarantees that the advanced FULL policy does not degrade spatial equity
  compared to status-quo baseline dispatch.

---

## 4. Human-in-the-Loop Governance & Tiered Dispatch (C2)

Automated accident detection models inevitably generate false alarms due to lighting variations, shadows,
benign stopped vehicles, and weather occlusion. Dispatching emergency units to false alarms squanders scarce public resources
and endangers the public through high-speed emergency vehicular transits.

To mitigate this, ResQCity implements a **three-tier confidence policy (C2)**:

1. **High Confidence ($P \ge \tau_{\text{high}}$):** Immediate multi-agency dispatch with audible alerts and visual explanation.
2. **Intermediate Confidence ($\tau_{\text{low}} \le P < \tau_{\text{high}}$):** Sends a rapid single police patrol unit
   to verify on scene while placing ambulance and fire brigades on pre-alert status. If confirmed, full dispatch is triggered instantly.
3. **Low Confidence ($P < \tau_{\text{low}}$):** Queued to an operator review dashboard with a fixed timeout ($T_{\text{review}} = 120\text{s}$).
   If the human operator does not approve the incident within the timeout, the event is automatically dismissed without unit movement.

**Critical Safeguard:** Under no circumstance does ResQCity possess the algorithmic authority to cancel, suppress, or deny
an emergency response to a civilian 112 phone call. Every civilian call immediately triggers human verification and dispatch protocols.

---

## 5. Prevention of Dual-Use and Surveillance Creep

The deployment of automated video analytics on public camera infrastructure carries severe dual-use risks,
notably the potential for mass surveillance, political protest tracking, or vehicular movement profiling.

ResQCity adheres to structural architectural guardrails against dual-use:

- **Absence of Re-Identification (Re-ID):** The codebase explicitly omits Re-ID neural networks or cross-camera tracking modules.
  Vehicular tracks terminate at the single-camera frame boundary.
- **Architectural Isolation:** The video analysis pipeline is physically and logically partitioned from vehicle registration
  databases (e.g., VAHAN) and criminal record databases.
- **Immutable Operator Audit Log:** Every automated event suggestion, operator confirmation, override, and dispatch order
  is recorded with cryptographic hashing in `backend/audit.py`, ensuring transparency and legal accountability for civic ombudsmen.
