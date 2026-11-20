# Section 4: Discussion

## 4.1 Interpretation of Findings

### RQ1: CCTV vs. Call Detection Lead Time
The experimental results demonstrate that automated CCTV detection provides a consistent lead time over civilian 112 calls
for incidents occurring within camera fields of view. However, because camera placement is restricted to ~4% of municipal
intersections under realistic budget constraints, city-wide detection delay is heavily dominated by civilian call latency.
Automated video detection is thus not a substitute for the public 112 system, but a high-speed accelerator for covered arterial nodes.

### RQ2: Impact of Joint Multi-Agency Synchronization
The comparison between B0 and B1 confirms that the primary operational bottleneck in current municipal emergency systems
is the sequential inter-agency hand-off delay (~120 seconds per agency). By sharing a unified Bayesian operational picture,
joint dispatch eliminates relay latency, enabling ambulances and police units to mobilize simultaneously.
The joint MILP (B2) and FULL policies further reduce the first-to-last arrival gap, ensuring medical personnel do not arrive
at hostile crash scenes without police traffic clearance or extrication equipment.

### RQ3: Confidence Tiering and Resource Preservation
The tiered confidence policy (C2) proved critical in mitigating the false-alarm hazard. Under unmitigated B1/B2 dispatch,
low-confidence camera artifacts trigger premature full-fleet deployments, causing unit depletion. By routing intermediate
events ($\tau_{\text{low}} \le P < \tau_{\text{high}}$) to rapid single-unit police verification and low-confidence events
to operator review, the system preserved over 90% of unit-hours that would otherwise have been squandered on false alarms.

### RQ4: Price of Fairness and Spatial Equity
The analysis of ward-level response times reveals the subtle tension between global efficiency and spatial justice.
Unconstrained mathematical optimization naturally gravitates toward high-density central wards (Shivajinagar, Kasba Peth)
where unit density is high and travel distances are short. The fairness-constrained pre-positioning formulation (C3)
actively counters this by maintaining baseline coverage guarantees in peripheral wards (Hadapsar, Katraj, Dhankawadi).

## 4.2 Practical Implications for Municipal Emergency Operations
For municipal authorities such as the Pune Municipal Corporation (PMC) and Pune Police, this research offers
three direct policy insights:
1. **Integrated Multi-Agency CAD:** Integrating ambulance, police, and fire dispatch into a single unified
   computer-aided dispatch interface yields immediate, substantial arrival reductions without purchasing additional vehicles.
2. **Selective Surveillance Augmentation:** Expanding CCTV networks specifically at identified high-risk accident hotspots
   (via MCLP optimization) maximizes the return on camera infrastructure investments.
3. **Dispatcher-in-the-Loop Safeguards:** An automated decision-support engine must never autonomously deny emergency
   responses to civilian callers. Human dispatcher review with automated timeouts ensures constitutional and operational accountability.

## 4.3 Limitations and Threats to Validity
1. **Synthetic Demand Modelling:** Due to strict privacy restrictions on historical 112 emergency calls,
   incident schedules were generated from calibrated Poisson processes rather than raw municipal dispatch archives.
2. **Simulated Video Ingest:** CCTV cameras were simulated by looping public benchmark crash videos rather than live municipal feeds.
3. **Passive Green Corridors:** Travel time reductions ($\rho_{\text{green}} = 0.8$) model police escort clearing,
   not physical hardware-level traffic signal preemption.
4. **Trauma Survival Modelling:** Trauma survival curves are acknowledged as secondary illustrative transforms
   (marked `UNVERIFIED` in `configs/survival.yaml`), as clinical trauma literature lacks consensus on universal per-minute coefficients.

## 4.4 Conclusion and Future Work
ResQCity demonstrates that combining automated CCTV computer vision with Bayesian evidence fusion and joint integer programming
dispatch significantly reduces emergency response times across the Pune metropolitan network. Future work will explore
real-time integration with municipal smart-city command centres, decentralized reinforcement learning for dynamic fleet routing,
and connected vehicle (V2X) traffic signal priority.
