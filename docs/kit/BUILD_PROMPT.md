# ResQCity — Claude Code build prompts

How to use this file:
1. Paste **Prompt 0 (Kickoff)** into Claude Code first, in plan mode (press Shift+Tab until "plan mode" shows).
2. Then paste one phase prompt at a time (Phase 1, Phase 2, ...). Don't paste them all at once.
   Each phase ends with acceptance checks; only move on when they pass.
3. `CLAUDE.md` and `.claude/` from this kit must already be in the repo root (see SETUP.md).

---

## Prompt 0 — Kickoff (paste first)

```
You are building "ResQCity", a research prototype for my university Research Methodology project.
Read CLAUDE.md fully before doing anything. It contains the architecture, conventions and
research-integrity rules you must follow.

PROJECT SUMMARY
A smart-city decision-support system that (1) detects road accidents from CCTV feeds with a
confidence score and severity estimate, (2) fuses them with simulated 112 emergency calls,
(3) decides which services (ambulance, police, fire brigade) and how many units to send using
Operations Research models, and (4) evaluates this against today's call-based, agency-by-agency
dispatch in a discrete-event simulation of the real Pune road network.

RESEARCH QUESTIONS (the code must be able to answer these)
- RQ1: How much earlier is an incident detected from CCTV than from a 112 call?
- RQ2: Does joint, synchronized multi-agency dispatch reduce the time until the LAST required
       service arrives, compared with independent per-agency dispatch?
- RQ3: Which confidence thresholds best trade off false-alarm cost against delay cost, and how do
       they shift with fleet size?
- RQ4: What is the "price of fairness" — how much does average performance worsen when we enforce
       minimum coverage in every ward?

NOVEL CONTRIBUTIONS (protect these; they are the research)
C1. Joint multi-agency dispatch MILP with severity-dependent unit requirements, synchronized
    arrival (minimize the last critical arrival) and police-assisted travel-time reduction
    (police clearing traffic / green corridor speeds up the ambulance).
C2. Confidence-aware dispatch policy: act on uncertain CCTV detections in tiers
    (dispatch all / send police to verify + pre-alert others / operator review with timeout),
    with thresholds optimized by simulation.
C3. Fairness-constrained pre-positioning of idle units using forecasted ward-hour demand.
C4. Open simulation testbed on Pune's real road network with normal, monsoon (flooded roads) and
    Ganeshotsav scenarios.

CONSTRAINTS
- Live CCTV feeds are NOT available. We simulate cameras by streaming public accident datasets
  (TAD, CADP, MADP, IEEE DataPort accident videos) as RTSP streams pinned to real Pune junctions.
- Incident and call data are synthetic but configurable; label them synthetic everywhere.
- Must run on a laptop (CPU) for the demo; GPU only for training, with a Colab notebook fallback.

YOUR FIRST TASK (this turn)
Do not write application code yet. Produce:
1. docs/plans/master-plan.md: repo layout, module boundaries, data flow, the shared schemas
   (field-level), the list of phases below with your estimate of risk per phase, and anything
   in my plan you think is wrong or missing.
2. A list of questions for me, only where the answer changes the design.
Then wait for my approval.

The phases I will give you next are:
 1 Scaffolding + schemas   2 Pune road network + scenario data   3 Simulated CCTV streams + vision
 4 Fusion + call simulator  5 Forecasting   6 OR models   7 Simulator   8 Backend API
 9 Dashboard   10 Experiments + statistics   11 Docs, reproducibility and paper support
```

---

## Phase 1 — Scaffolding and shared schemas

```
Phase 1: scaffolding. Use plan mode, save the plan to docs/plans/phase-1.md, then implement.

Create the monorepo described in CLAUDE.md:
- pyproject.toml managed by uv (Python 3.11), ruff, pytest, pre-commit; Makefile with every target
  listed in CLAUDE.md (stub targets that print "not implemented yet" are fine for now).
- common/schemas.py (pydantic v2) with: Camera(id, lat, lon, junction_node, ward_id, fov_note),
  Event(id, camera_id, ts, type[accident|fire_smoke|crowd|other], confidence 0..1,
  severity_features{vehicle_classes, overturned, person_down, fire_visible, num_vehicles},
  severity_score 0..1, required_services{ambulance,police,fire: int}, clip_ref),
  Call(id, ts, reported_location, location_error_m, caller_text, triaged_type, triaged_severity),
  Incident(id, location, node, ward_id, created_ts, sources[], posterior, severity, status,
  requirements{ambulance,police,fire}), Unit(id, agency, home_station, node, state, eta_fn_ref),
  Station, Hospital(id, node, has_trauma, has_burns, icu_beds, beds_available),
  DispatchDecision(incident_id, assignments[{unit_id, eta_s, role}], tier, expected_last_arrival_s,
  objective_breakdown), Explanation(decision_id, text, factors[{name, value, weight}]).
- common/config.py: load YAML from configs/ into typed pydantic settings; configs/base.yaml.
- Generate TypeScript types for the dashboard from the pydantic schemas (script in tools/).
- GitHub Actions CI: lint + tests.
- docs/decisions.md and docs/data-sources.md (empty templates).

ACCEPTANCE: `make setup && make lint && make test` pass; schema round-trip tests pass;
TS types generate without errors. Commit as "phase 1: scaffolding and schemas".
```

---

## Phase 2 — Pune road network and scenario data

```
Phase 2: road network, facilities and scenarios. Plan first (docs/plans/phase-2.md).

1. data/: download the Pune drivable network with OSMnx (place: "Pune, Maharashtra, India";
   also support a bounding-box option so we can include Pimpri-Chinchwad later). Add speed and
   travel-time attributes, simplify, keep the largest strongly connected component, and cache to
   data/processed/pune_drive.graphml. Record download date and OSM attribution in docs/data-sources.md.
2. Facilities from OSM tags: police stations (amenity=police), fire stations (amenity=fire_station),
   hospitals (amenity=hospital). Snap each to the nearest graph node. Ambulance bases = hospitals
   plus configurable extra bases. Hospital capabilities (trauma, burns, ICU beds) are synthetic
   and configurable; mark them synthetic.
3. Wards: load PMC ward boundaries if a public GeoJSON is available; otherwise build a hex grid
   (H3, resolution configurable) and note the substitution in docs/decisions.md.
4. Camera sites: choose N (configurable, default 40) junctions — top intersections by degree and
   betweenness on arterial roads — and write configs/cameras.yaml.
5. Time-dependent travel times: speed multipliers per road class per hour (peak/off-peak) in
   config; a function travel_time(u, v, t, scenario).
6. Scenarios in configs/scenarios/:
   - normal.yaml
   - monsoon.yaml: rainfall intensity profile; flood-prone edges = edges within X m of the
     Mula/Mutha rivers or in low-elevation cells (use SRTM elevation if easy, else river buffer);
     a fraction of them become impassable or slowed depending on rainfall.
   - ganeshotsav.yaml: procession routes (configurable polylines) closed or slowed during set
     hours, and higher incident rates nearby.
7. Synthetic incident generator: non-homogeneous Poisson process per ward-hour with rates
   driven by a hotspot score (road class, junction density), hour-of-day, rain and festival
   multipliers. Incident types and severity mix from config. Each incident gets ground-truth
   required services by rule table (e.g. heavy vehicle + overturned -> ambulance 2, police 1,
   fire 1 for extrication). Write the rule table to configs/requirements.yaml.

ACCEPTANCE: graph loads offline in <10 s; shortest-path travel times sane (test a few known
pairs: report them, I will sanity-check); a notebook notebooks/02_network.ipynb plots wards,
facilities, cameras and flooded edges; generator reproducible for a given seed. Commit.
```

---

## Phase 3 — Simulated CCTV streams and the vision pipeline

```
Phase 3: cameras and vision. Use the vision-eval skill. Plan first (docs/plans/phase-3.md).

A. streams/
- docker-compose with MediaMTX. A script that loops video files from data/videos/<camera_id>/
  into rtsp://localhost:8554/<camera_id> using ffmpeg (-re, loop). A "scenario player" that, at
  configured times, switches a camera from a normal-traffic clip to an accident clip so the demo
  can trigger incidents on cue.
- Support a phone IP-camera URL as a real live input for the demo.

B. vision/
- Ingest any RTSP/HTTP/file source with OpenCV; frame sampling configurable (default 5 fps).
- Detection + tracking: Ultralytics YOLO (small model for CPU) + ByteTrack.
- Accident detector, two stages so it is explainable:
  1) Track-based heuristics: sudden deceleration, trajectory intersection / bounding-box overlap
     followed by stopping, vehicle orientation change (overturn proxy), person class appearing
     on carriageway and not moving.
  2) A learned clip classifier on short windows (e.g. 2–4 s) — start with a lightweight model
     (frame-feature extractor + temporal pooling or a small 3D-CNN / VideoMAE-small if GPU
     allows). Output calibrated probability (temperature scaling or isotonic regression).
  Combine both into one confidence score; keep the components for the explanation panel.
- Fire/smoke detector (fine-tune YOLO on a public fire/smoke dataset; document which one).
- Severity features -> severity_score and required_services via configs/requirements.yaml.
- Privacy: blur faces and number plates before any frame is stored or displayed; only Event
  metadata leaves the module. Annotated preview stream (MJPEG) for the dashboard uses blurred frames.
- Training/eval: datasets TAD, CADP, MADP and IEEE DataPort accident videos — write download
  instructions (I will download manually if licences require), a unified annotation format,
  video-level train/val/test splits, and a Colab notebook vision/notebooks/train.ipynb.
- Metrics script: clip-level precision/recall/F1/AUROC, detection delay (seconds from accident
  onset to first alarm above threshold), false alarms per camera-hour on normal-traffic footage,
  latency per frame on CPU. Save to results/vision/ with the model hash.

ACCEPTANCE: `make streams` serves at least 4 RTSP cameras; `python -m vision.run --camera C01`
emits Events on the event bus when an accident clip plays; metrics report generated on the test
split; blur verified on a sample frame saved to results/vision/privacy_check.png. Commit.
```

---

## Phase 4 — 112 call simulator and Bayesian fusion

```
Phase 4: calls and fusion. Plan first (docs/plans/phase-4.md).

- Call simulator: for each ground-truth incident, generate 0..k calls with a reporting-delay
  distribution (configurable; default lognormal, parameters in config with a source field or
  UNVERIFIED), location error, and caller text from templates in English, Hindi and Marathi
  (synthetic). Also generate prank/duplicate calls at a configurable rate.
- Call triage: interface CallTriager with (a) a rule/keyword baseline and (b) an optional
  LLM-based triager using the Anthropic Python SDK (model and key from env; off by default,
  never required for tests or experiments). Output type + severity + confidence.
- Fusion: maintain candidate incidents; associate events/calls by space-time gating
  (distance on graph + time window). Posterior P(real incident) via Bayesian update using the
  detector's calibrated likelihood ratios and the call reliability. Combine severity estimates.
  Emit Incident updates when posterior or severity changes.
- Unit tests for association (two cameras + one call on the same crash -> one incident) and for
  posterior monotonicity.

ACCEPTANCE: replaying a scripted scenario yields correct incident merges; posterior trace
plotted in notebooks/04_fusion.ipynb. Commit.
```

---

## Phase 5 — Demand forecasting

```
Phase 5: forecasting. Plan first (docs/plans/phase-5.md).

- Build a ward-hour panel from the synthetic generator (and from real data later if I get it).
- Features: hour, weekday, holiday/festival flag, rainfall (lagged), hotspot score, recent counts.
- Models: historical-average baseline vs LightGBM (Poisson objective). Time-based split.
- Metrics: MAE, Poisson deviance, calibration of predicted vs observed counts, top-k ward hit rate.
- Expose predict(ward, t) and a "forecast error" knob (inject multiplicative noise) used later
  for sensitivity analysis.

IMPORTANT: since training data is synthetic, state clearly in docs that forecasting accuracy is
not a research claim; the claim is how dispatch performance depends on forecast quality.
ACCEPTANCE: metrics saved to results/forecasting/; model reloads deterministically. Commit.
```

---

## Phase 6 — Operations Research models (core contribution)

```
Phase 6: OR models. Use the or-model-builder skill for EVERY model. Plan first
(docs/plans/phase-6.md). For each model: write the mathematical formulation in
docs/models/<name>.md (LaTeX), implement, test on tiny instances against brute force, then
have the or-model-reviewer subagent review formulation vs code.

6.1 Routing: time-dependent shortest paths (A* with travel_time(u,v,t,scenario)); flooded edges
    removed or slowed; green-corridor mode applies a speed multiplier on the ambulance route
    when a police unit (or signal pre-emption) is active on it. Cache ETA matrices
    (stations x camera/incident nodes) per hour bucket.

6.2 Joint multi-agency dispatch MILP (C1). For a batch of open incidents I and available
    units U (agencies a in {amb, pol, fire}):
    - x[u,i] in {0,1} assignment; requirement constraints sum_{u in U_a} x[u,i] >= r[i,a]
      (allow shortfall variables with big penalty if not enough units).
    - Arrival of agency a at i: A[i,a] >= eta[u,i]*x[u,i] (linearised).
    - Synchronization: L[i] >= A[i,a] for every required a (time of last critical arrival).
    - Police-assisted travel: if police arrives first, ambulance eta reduced by factor rho
      (model with an indicator; keep rho in config and also test rho=0 as ablation).
    - Coverage-loss term: penalty for leaving any ward below its coverage level after assignment
      (use forecasted demand).
    - Objective: minimise sum_i w(severity_i) * L[i] + lambda * coverage_loss + mu * shortfall.
    - Solve with OR-Tools; time limit 1 s for live use; fall back to a greedy heuristic if not
      solved. Return DispatchDecision + Explanation (top factors in plain English).

6.3 Confidence-aware policy (C2): tiers by fused posterior p:
    p >= tau_high: full dispatch; tau_low <= p < tau_high: nearest police to verify + pre-alert
    (units move to "ready"), escalate on confirmation; p < tau_low: operator review with timeout
    T_review, auto-escalate on timeout. Expose (tau_low, tau_high, T_review) as parameters to be
    optimised in Phase 10 by simulation (grid search + optional Bayesian optimisation).

6.4 Pre-positioning with fairness (C3): every 30 min (configurable) solve a maximal expected
    coverage model (MEXCLP or double-standard model) over idle units using forecasted demand,
    with constraint: each ward covered within T_ward minutes with prob >= alpha, and a
    relocation budget (max moves, max total relocation time). Report the "price of fairness"
    by solving with and without the fairness constraint.

6.5 Camera placement (MCLP): choose K new camera junctions to maximise risk-weighted coverage
    of incidents; output map + marginal gain curve for K = 0..50.

6.6 Erlang-C / M/M/c fleet sizing per agency and zone: utilisation, P(wait), expected wait;
    used as an analytical sanity check against the simulator.

ACCEPTANCE: all tiny-instance tests pass; solver time for a realistic batch (10 incidents,
60 units) under 1 s on my laptop or a documented fallback; reviewer subagent finds no
formulation/code mismatch. Commit.
```

---

## Phase 7 — Discrete-event simulator (digital twin)

```
Phase 7: SimPy simulator. Plan first (docs/plans/phase-7.md).

- Processes: incident occurrence (ground truth), CCTV detection (camera coverage, detector
  recall/precision and delay drawn from Phase 3 measured distributions, NOT assumed), calls
  (Phase 4), fusion, policy decision, unit travel on the graph, on-scene time, hospital transport
  and handover (hospital capability + capacity), return/relocation.
- Pluggable policies:
  B0 calls only + independent nearest-unit per agency (status quo)
  B1 CCTV + calls + independent nearest-unit
  B2 CCTV + calls + joint MILP (C1)
  B3 B2 + confidence tiers (C2)
  FULL B3 + fairness pre-positioning (C3) + green corridor
- False alarms consume units (they travel and return) so their cost is real in the metrics.
- Outcome metric: expected survival from a configurable survival-vs-time curve per incident
  type (configs/survival.yaml with `source:` citations; UNVERIFIED until I confirm).
- Per-incident log (parquet) with every timestamp: onset, first detection, first call,
  dispatch, each agency arrival, hospital arrival; plus unit utilisation log.
- Common random numbers: one RNG stream per process family, seeded from replication id.
- Validation: compare simulated utilisation/waits with Erlang-C (Phase 6.6) under simplified
  settings; document the comparison.

ACCEPTANCE: `make sim SCENARIO=normal POLICY=B0` and `POLICY=FULL` both run a 7-day simulated
week in reasonable time; logs pass schema checks; validation notebook saved. Commit.
```

---

## Phase 8 — Backend API (live mode)

```
Phase 8: FastAPI backend. Plan first (docs/plans/phase-8.md).

- Event bus (Redis pub/sub if available, in-process fallback). Services: vision workers,
  fusion, dispatcher, unit simulator (moves demo units along routes in real time).
- REST: /cameras, /incidents, /units, /stations, /hospitals, /decisions/{id},
  POST /decisions/{id}/approve, POST /decisions/{id}/override (with reason), /scenario/start,
  /scenario/stop, /kpis. WebSocket /ws for live updates.
- Operator-in-the-loop: decisions wait for approval in "assist" mode; auto-execute in "auto" mode.
  Every approval/override is written to an audit log (who, when, what changed, reason).
- Hospital pre-alert and green-corridor are emitted as events (simulated integrations).
- Tests with httpx + pytest-asyncio.

ACCEPTANCE: `make api` + a scripted scenario produce an incident, a decision with explanation,
and moving units over WebSocket. Commit.
```

---

## Phase 9 — Dashboard

```
Phase 9: dashboard (React + Vite + TypeScript + Leaflet). Use the frontend-design skill for
visual direction; it is a control-room tool, so prioritise legibility, dark theme, clear status
colours (with icons/shape, not colour alone), and keyboard shortcuts for approve/override.
Plan first (docs/plans/phase-9.md).

Screens:
1. Live operations: map with wards (risk heat layer), cameras, stations, hospitals, units
   (icons by agency, moving), incidents; side panel with camera tiles (annotated, blurred
   preview streams), incident queue sorted by severity x posterior, recommendation card showing
   assignments, ETAs, tier, and the plain-English explanation + factor bars; Approve / Override.
2. Scenario control: start normal / monsoon / Ganeshotsav, trigger accident on a camera,
   knock out units or cameras ("what-if" stress test), toggle policies.
3. KPIs: detection-to-dispatch, response time p50/p90, last-arrival gap, coverage by ward,
   utilisation, override rate.
4. Results: load results/ tables and figures from Phase 10 for the viva demo.

Use the webapp-testing skill (Playwright) to write end-to-end tests for: trigger accident ->
decision appears -> approve -> units move. Run /verify at the end.
ACCEPTANCE: `make demo` runs the full loop in the browser; Playwright tests pass. Commit.
```

---

## Phase 10 — Experiments and statistics

```
Phase 10: experiments. Use the run-experiment skill and have the research-auditor subagent
review the design BEFORE running and the results AFTER. Plan first (docs/plans/phase-10.md).

Design:
- Scenarios: normal, monsoon, ganeshotsav. Policies: B0, B1, B2, B3, FULL.
- Ablations: FULL minus each of {joint MILP, confidence tiers, pre-positioning, green corridor,
  police-assist rho}.
- Replications: 30 per cell (more if CIs too wide), common random numbers across policies.
- Metrics per incident and aggregated: detection delay (CCTV vs call), detection-to-dispatch,
  response time p50/p90 per agency, time to last required arrival, first-to-last arrival gap,
  % within 8/15 min, unit utilisation, false-alarm unit-hours, ward coverage gap (max–min and
  Gini), expected survival.
- RQ3: grid over (tau_low, tau_high, T_review) x fleet size; plot the false-alarm cost vs
  delay trade-off curve and the optimal thresholds per fleet size.
- RQ4: price of fairness = % change in mean response time with vs without the fairness
  constraint, across alpha levels.
- Sensitivity: fleet size, detector recall/precision, forecast error, flooded-edge fraction,
  camera count (link to MCLP placement).
Statistics:
- Paired comparisons per replication (CRN): paired t-test if differences are ~normal
  (Shapiro-Wilk), else Wilcoxon signed-rank; Holm correction across comparisons; report effect
  sizes and 95% CIs (bootstrap for p90 metrics). Save every test to results/stats/*.csv.
Figures: publication-quality, one script per figure in experiments/figures/, colour-blind-safe.

ACCEPTANCE: `make experiments` and `make figures` regenerate everything; results/summary.md
is generated by a script (not typed) and lists every hypothesis with test, statistic, p, effect
size and a one-line plain-English reading. Commit.
```

---

## Phase 11 — Documentation, reproducibility and paper support

```
Phase 11: documentation. Use the research-writeup skill.

- README with architecture diagram, setup, demo instructions, screenshots/GIF.
- docs/methodology.md: research design, variables, hypotheses, simulation design, validation,
  statistical analysis plan, threats to validity (internal, external, construct), limitations
  (synthetic data, simulated feeds, no live integration), ethics (privacy, DPDP Act 2023 and
  DPDP Rules 2025 considerations, bias, human oversight).
- docs/models/*.md complete with LaTeX formulations; docs/data-sources.md with licences.
- `make reproduce` from a clean clone; record runtime and machine spec.
- Draft paper sections (IMRaD) in paper/ as Markdown with every number pulled from results/
  via a script, plus a figures list. Do not invent citations; leave [CITE: topic] placeholders
  where I must add references.

ACCEPTANCE: a fresh clone + `make setup && make reproduce` succeeds; research-auditor reports
no unsupported claims. Commit and tag v1.0.
```
