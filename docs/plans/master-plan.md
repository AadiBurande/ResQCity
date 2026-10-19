# ResQCity — Master plan

Status: draft for approval · Written: 2026-10-20 · Covers: Prompt 0 (kickoff) of `BUILD_PROMPT.md`

Research question: does CCTV-triggered, confidence-aware joint dispatch reduce time-to-full-response
compared with today's call-based, agency-by-agency dispatch, without increasing inequity across wards?

---

## 1. Design principles

1. **One codebase, two modes.** The same fusion, policy and optimization code runs in **live mode**
   (RTSP cameras → FastAPI → dashboard) and in **experiment mode** (SimPy digital twin). The
   simulator never re-implements a policy. It calls the same `DispatchPolicy` objects the backend calls.
2. **Experiments do not run neural networks.** In the simulator, the detector is a *statistical
   model* (recall, false-alarm rate, detection-delay distribution) read from
   `results/vision/sim_inputs.yaml`, which the Phase 3 evaluation produces. Running YOLO for
   thousands of simulated incidents would be slow, and it would add nothing to the research.
3. **Provenance on every number.** Every parameter that comes from the literature is a
   `SourcedValue {value, source, verified}`. Every result file has a sidecar `meta.json`
   (git hash, config hash, seed, package versions, machine).
4. **Synthetic is labelled.** Synthetic files are named `*_synthetic.*`, carry `synthetic: true`
   in their metadata, and the docs say so.
5. **Tiny tests, no network.** Unit tests use toy graphs and toy instances. Only `make data` touches
   the internet (OSM, datasets).

## 2. Repository layout

```
ResQCity/                         (repo root; layout revised in D-017)
├── src/resqcity/                 the single Python package: import resqcity.<module>
│   ├── common/        schemas.py (pydantic v2), config.py (YAML → typed), rng.py, provenance.py, requirements.py
│   ├── geo/           road graph (OSMnx), facilities, wards, camera-site selection, scenarios, travel times
│   ├── streams/       loop_cameras.py, scenario_player.py (docker-compose.yml for MediaMTX at streams/)
│   ├── vision/        ingest, detect_track (YOLO+ByteTrack), accident_heuristics, clip_classifier,
│   │                  calibration, fire_smoke, severity, privacy (blur), run.py, eval/
│   ├── fusion/        call_sim (synthetic 112 calls), triage (rules | optional LLM), associate, posterior
│   ├── forecasting/   panel builder, baseline (historical average), LightGBM-Poisson, predict(ward, t)
│   ├── optimization/  routing, joint_dispatch (C1), confidence_policy (C2), preposition (C3),
│   │                  camera_placement (MCLP), erlang, heuristics, explain, bench_*.py
│   ├── sim/           generator, ground truth, SimPy engine, policies registry (B0…FULL), logging
│   ├── backend/       FastAPI app, bus (Redis | in-process), services, audit log, ws
│   ├── experiments/   run.py (resumable), stats.py, summarize.py, figures/*.py
│   ├── paper/         build.py (fills {{placeholders}} in paper/*.md from results/)
│   └── tools/         gen_ts_types.py, tasks.py (cross-platform task runner)
├── configs/           base, geo, incidents, requirements, cameras (generated), scenarios/, experiments/,
│                      archive/ (superseded settings kept for reproducibility)
├── data/              raw/ (gitignored), processed/ (gitignored, regenerable), videos/<camera_id>/
├── results/           raw/ (gitignored parquet), geo/, stats/, vision/, forecasting/, bench/, figures/, summary.md
├── notebooks/         02_network.ipynb, vision/train.ipynb (Colab), ...
├── dashboard/         React + Vite + TS + Leaflet; src/types/schemas.ts is GENERATED
├── tests/             mirrors the package: tests/common, tests/geo, tests/sim, ...
├── docs/              plans/, models/*.md (LaTeX), decisions.md, data-sources.md, limitations.md,
│                      methodology.md, ethics.md, kit/ (original build prompts + setup guide)
└── CLAUDE.md, README.md, Makefile, pyproject.toml, uv.lock
```

Module folders are created in the phase that implements them (no empty placeholders).

Module boundaries (who may import whom):

```
common  ←  everyone
geo     ←  optimization, sim, backend, fusion, forecasting
vision  →  only common (emits Event; never imports backend). Frames never leave vision/.
fusion  →  common, geo
optimization → common, geo, forecasting (for demand inputs only)
sim     →  common, geo, fusion, optimization, forecasting   (NOT vision, NOT backend)
backend →  common, geo, fusion, optimization, forecasting, (vision via bus messages only)
experiments → sim, common
```

## 3. Data flow

### Live mode (demo)
```
data/videos/C01/*.mp4 ─ffmpeg -re -stream_loop─► MediaMTX rtsp://localhost:8554/C01
                                                    │
                    vision.run --camera C01 ◄───────┘   (blur faces/plates at ingest)
                    YOLO+ByteTrack → heuristics + clip classifier → calibrated p
                    → Event{confidence, severity_features, required_services}
                                                    │ bus: events.vision
call_sim (synthetic 112) → triage → Call ──────────►│ bus: events.call
                                                    ▼
                         fusion: associate (graph distance × time gate) → Bayesian posterior
                         → Incident{posterior, severity, requirements}      bus: incidents
                                                    ▼
             confidence policy (C2): tier by posterior → joint MILP (C1) → DispatchDecision + Explanation
                                                    ▼
                    operator (assist: approve/override | auto) → audit log
                                                    ▼
                    unit simulator moves units along routes ─► WebSocket /ws ─► dashboard
   every 30 min: forecasting.predict → pre-positioning (C3) → relocation decisions
```

### Experiment mode
```
scenario yaml + seed ─► incident generator (ground truth, synthetic NHPP per ward-hour)
   ─► SimPy: detection process (sim_inputs.yaml) + call process (calls.yaml)
   ─► fusion (same code) ─► policy (same code: B0…FULL) ─► unit travel (resqcity.geo.travel_time)
   ─► on-scene, transport, hospital handover, return / relocation
   ─► results/raw/<exp>/<scenario>/<policy>/rep_<r>.parquet  + meta.json
   ─► experiments.stats ─► results/stats/*.csv ─► experiments.summarize ─► results/summary.md
   ─► experiments/figures/*.py ─► results/figures/*.{png,pdf} ─► paper.build
```

## 4. Shared schemas (field level, `common/schemas.py`)

All models are pydantic v2 with `extra="forbid"`. Times are timezone-aware datetimes, durations are
**seconds**, distances are **metres**, probabilities are in [0, 1]. Graph nodes are OSM node ids (`int`).
TypeScript types in `dashboard/src/types/schemas.ts` are generated from these models, so the
Python models are the single source of truth.

| Model | Fields |
|---|---|
| `GeoPoint` | lat [-90,90], lon [-180,180] |
| `ServiceCounts` | ambulance ≥0, police ≥0, fire ≥0 (int) |
| `Camera` | id, lat, lon, junction_node, ward_id, fov_note, *stream_url?* |
| `SeverityFeatures` | vehicle_classes[str], overturned, person_down, fire_visible (bool), num_vehicles ≥0 |
| `Event` | id, camera_id, ts, type ∈ {accident, fire_smoke, crowd, other}, confidence, severity_features, severity_score, required_services: ServiceCounts, clip_ref?, *confidence_components{name: value}* |
| `Call` | id, ts, reported_location: GeoPoint, location_error_m ≥0, caller_text, *language ∈ {en, hi, mr}*, triaged_type?, triaged_severity?, *triage_confidence?*, *synthetic = true* |
| `SourceRef` | kind ∈ {camera_event, call}, ref_id |
| `Incident` | id, *type*, location: GeoPoint, node, ward_id, created_ts, *updated_ts*, sources[SourceRef], posterior, severity, status ∈ {candidate, verifying, confirmed, dispatched, on_scene, resolved, dismissed}, requirements: ServiceCounts |
| `Unit` | id, agency ∈ {ambulance, police, fire}, home_station, node, state ∈ {idle, ready, en_route, on_scene, transporting, at_hospital, returning, relocating, out_of_service}, eta_fn_ref, *position?* |
| `Station` | id, name, agency, location, node, ward_id, *source ∈ {osm, config, synthetic}* |
| `Hospital` | id, *name, location*, node, has_trauma, has_burns, icu_beds ≥0, beds_available ≥0, *capabilities_synthetic = true* |
| `Assignment` | unit_id, eta_s ≥0, role ∈ {respond, verify, pre_alert, green_corridor}, *hospital_id?* |
| `DispatchDecision` | *id*, incident_id, *created_ts*, *policy*, assignments[Assignment], tier ∈ {full, verify, review}, expected_last_arrival_s, objective_breakdown{term: value}, *status ∈ {proposed, approved, overridden, auto_executed, expired}* |
| `Factor` | name, value, weight |
| `Explanation` | decision_id, text, factors[Factor] |

Fields in *italics* are additions to the field list in the brief. Each one is needed by a later
phase (for example, `Explanation.decision_id` needs `DispatchDecision.id`). They are recorded in
`docs/decisions.md`.

Phase 2 adds internal (non-API) ground-truth types: `GroundTruthIncident` (true onset, true
requirements, true type) lives in `sim/`. Ground truth must never be visible to policies.

## 5. Phases, risk and the main risk in each

| # | Phase | Risk | Main risk → mitigation |
|---|---|---|---|
| 1 | Scaffolding + schemas | Low | No GNU make on Windows → `tools/tasks.py` runner that the Makefile also calls |
| 2 | Road network + scenarios | **Med** | OSM speed tags are sparse in Pune and ward boundaries may be missing → default speeds per road class (configured, cited), H3 fallback; travel-time sanity check by you |
| 3 | CCTV streams + vision | **High** | Dataset licences/access, small and heterogeneous data, domain shift (dashcam ≠ fixed CCTV), CPU latency → heuristics first, small classifier, honest per-dataset metrics; the sim depends only on `sim_inputs.yaml` |
| 4 | Fusion + calls | Low–Med | Likelihood ratios must come from calibrated detector output → calibration is a Phase 3 deliverable; a placeholder file is marked UNVERIFIED until then |
| 5 | Forecasting | Low | Circularity (it learns the generator) → no accuracy claim; only sensitivity to forecast error |
| 6 | OR models | **High** | Police-assist linearisation, 1 s limit, wrong incentives in the objective (see §6.1) → brute-force tests, greedy fallback, reviewer subagent |
| 7 | Simulator | **High** | Runtime: routing on a ~10⁵-node graph inside thousands of runs → precomputed hour-bucket ETA matrices, multi-source Dijkstra, profile early |
| 8 | Backend API | Med | Async bus + real-time unit movement on Windows (Redis via Docker) → in-process bus fallback is the default |
| 9 | Dashboard | Med | Playwright on Windows; pnpm not installed → install pnpm via corepack |
| 10 | Experiments | **Med–High** | Compute budget (see §6.6) |
| 11 | Docs + reproduce | Low | `make reproduce` runtime → document it, and offer a `--quick` profile |

## 6. What I think is wrong or missing in the plan

1. **Minimising only the last arrival lets the solver delay the ambulance.** With objective
   `Σ w_i L_i`, any assignment where the ambulance arrives before `L_i` scores the same. The solver
   could send a far ambulance "because fire is late anyway", which costs lives. **Fix:** add a
   secondary term `ε · Σ_i Σ_a A_{i,a}` (or weight ambulance arrival explicitly), with ε from config.
   I'll test that it never changes `L_i` (ε smaller than the minimum ETA resolution / total weight).
2. **Police assist needs pairwise linearisation.** "Ambulance faster if police arrives first"
   depends on *which* police unit and *which* ambulance are chosen. That needs pair variables
   `y[p,m,i] ≤ x[p,i], y ≤ x[m,i]` and precedence `eta_p ≤ eta_m` (precomputed, so the indicator
   is data, not a variable). At about 20 police × 20 ambulances × 10 incidents that is around 4k
   binaries, which is fine for CP-SAT. The physical story ("police clears the route") also assumes
   the police route overlaps the ambulance route. I'll make the overlap condition explicit and
   configurable, not implicit.
3. **RQ1 is largely determined by assumptions.** "How much earlier is CCTV than a 112 call"
   depends almost entirely on the reporting-delay distribution (UNVERIFIED) and the measured
   detector delay. I'll report RQ1 as a curve over the reporting-delay parameters, not one
   number, and use the measured detector delay from Phase 3.
4. **The equity claim needs a non-inferiority test.** The headline RQ says "without increasing
   inequity", but RQ1–4 only test fairness in RQ4. Failing to find a significant increase does
   not show there is no increase. **Fix:** pre-register a ward-level equity metric (Gini or
   max–min of the ward p90 last-arrival time) and a non-inferiority margin δ, then test
   FULL vs B0 with a one-sided paired test. You need to pick δ (question Q5).
5. **The B0 baseline must not be a strawman.** Maharashtra already has a single 112 number (ERSS).
   B0's gain from "joint dispatch" comes from the inter-agency notification delay (112 → 108
   ambulance → fire) and from independent unit choice. Those delays must be config parameters
   with a source or UNVERIFIED, and the experiments need a sensitivity sweep over them. Otherwise a
   reviewer can say the result was built into the assumptions.
6. **Compute budget for Phase 10.** Main grid: 3 scenarios × (5 policies + 5 ablations) × 30
   reps = 900 week-long runs. RQ3 grid: e.g. 5 τ_low × 5 τ_high × 3 T_review × 3 fleet sizes ×
   30 reps = 6,750 runs. That is too much at full fidelity on a laptop. **Fix:** a shorter horizon
   for the RQ3/sensitivity grids (2 days with a warm-up), a coarse grid followed by Bayesian
   optimisation around the optimum, and the full 7-day / 30-rep runs only for the confirmatory
   comparisons (RQ2, RQ4). Tuning thresholds and evaluating them must use **different seeds**,
   or the RQ3 result is overfit.
7. **Requirement-estimation error must be simulated.** Ground-truth requirements come from
   `requirements.yaml`, and the vision severity estimate maps through the same table. If the sim
   gives policies the true requirements, CCTV-based policies look better than they are. **Fix:**
   the sim draws an estimated requirement from a confusion model (measured where possible).
   Under-dispatch triggers a second wave after on-scene assessment, and over-dispatch costs unit time.
8. **The survival curve should not be the primary outcome.** It is UNVERIFIED and would
   dominate the conclusions. Primary: time to last required arrival (RQ2). Expected survival
   is secondary and always shown with its sensitivity.
9. **Missing pieces I'll add:** a `geo/` package (the network is shared by 4 modules); a warm-up
   period and per-replication aggregation (incidents within a run are not independent); a
   held-out seed set for tuned parameters; `meta.json` provenance on every result; separate
   docs pages for ethics and limitations; and a "detector-model" fallback
   `sim_inputs_assumed.yaml` (UNVERIFIED) so Phases 4–7 are not blocked if Phase 3 is late.
10. **Dataset naming.** I could not confirm a public dataset called "MADP". Please send the
    link/paper. CADP and TAD are known. IEEE DataPort usually needs a subscription. Dashcam-style
    datasets (CADP has many) are a domain mismatch for fixed CCTV, so I'll report it as a threat
    to validity.

## 7. Questions for you (only where the answer changes the design)

- **Q1 Study area.** PMC only, or PMC + Pimpri-Chinchwad (PCMC)? This changes graph size (and
  sim runtime) and the facility set. *Default: PMC polygon, with bbox support built in.*
- **Q2 Wards.** Do you have a PMC ward/prabhag GeoJSON, or a preferred delimitation (2017
  prabhags vs newer)? *Default: try a public source, else H3 resolution 8 (≈0.74 km² hexes),
  recorded in decisions.md.*
- **Q3 Status quo (B0).** Is my reading right: one 112 call, then a sequential hand-off to 108
  (ambulance), police PCR and 101 (fire), each agency picking its own nearest unit? Do you have
  any figures (even rough) for hand-off delays? *Default: configurable delays marked
  UNVERIFIED + sensitivity sweep.*
- **Q4 Fleet sizes.** Real counts for Pune 108 ambulances, police PCR vehicles and PMC fire
  stations/tenders, or synthetic defaults? *Default: synthetic, scaled from facility counts.*
- **Q5 Equity margin.** Do you accept a pre-registered non-inferiority margin for ward inequity
  (e.g. Gini of ward p90 last-arrival may not rise by more than 0.02)? You choose δ. It must be
  fixed before Phase 10 runs.
- **Q6 Compute.** How many hours can the experiments run on this laptop (Ryzen 7 7840HS,
  8C/16T, 15 GB, RTX 4050 6 GB)? This sets horizon and grid sizes.
- **Q7 Tooling.** OK if I install GNU make (`winget install ezwinports.make`), ffmpeg
  (`winget install Gyan.FFmpeg`) and pnpm (`corepack enable pnpm`)? Also, is there a GitHub
  remote for CI? Until then, every Make target also works as
  `uv run python -m resqcity.tools.tasks <target>`.
