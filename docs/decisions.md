# Decision log

Design choices made where the brief was ambiguous or silent. Newest last. Each entry records the
context, the decision, alternatives considered, and the consequences. If a decision is reversed,
add a new entry that supersedes the old one; do not edit history.

Template:

```
## D-NNN Title  (YYYY-MM-DD, phase N)
Context: ...
Decision: ...
Alternatives: ...
Consequences: ...
```

---

## D-001 Cross-platform task runner behind the Makefile  (2026-10-20, phase 1)
Context: the development machine is Windows without GNU make; CLAUDE.md specifies `make` targets.
Decision: the `Makefile` stays the documented interface. Every target delegates to
`tools/tasks.py`, which also runs as `uv run python -m resqcity.tools.tasks <target>`.
Targets of later phases exit with status 3 ("not implemented yet (phase N)") instead of 0.
Alternatives: PowerShell `make.ps1` (not portable to CI/Linux); only stubs that print and succeed
(a silent no-op `make reproduce` would be a research-integrity hazard).
Consequences: one implementation per task; CI calls the same runner.

## D-002 Schema additions beyond the brief's field list  (2026-10-20, phase 1)
Context: later phases need fields that the Phase 1 brief does not list.
Decision: added `Camera.stream_url`, `Event.confidence_components` (explanation panel),
`Call.language`, `Call.triage_confidence`, `Call.synthetic`, `Incident.type`, `Incident.updated_ts`,
`Unit.position`, `Station.{name, location, ward_id, source}`, `Hospital.{name, location,
capabilities_synthetic}`, `Assignment.hospital_id`, `DispatchDecision.{id, created_ts, policy,
status}` (`Explanation.decision_id` needs a decision id). Value objects `GeoPoint`,
`ServiceCounts`, `SourceRef`, `Factor`.
Consequences: every addition has a default or is needed for referential integrity. All models
forbid unknown fields.

## D-003 Units and time conventions  (2026-10-20, phase 1)
Decision: durations in seconds, distances in metres, probabilities in [0, 1], timestamps are
timezone-aware (`AwareDatetime`); project time zone `Asia/Kolkata` from `configs/base.yaml`.
Consequences: naive datetimes are rejected at validation, so sim and live code cannot silently
mix time zones.

## D-004 TypeScript types generated without npm  (2026-10-20, phase 1)
Context: `pydantic2ts` needs the npm package `json-schema-to-typescript`; pnpm is not installed yet.
Decision: `tools/gen_ts_types.py` converts the pydantic serialization JSON schema to TypeScript
in pure Python. Unsupported constructs raise an error. A `--check` mode guards against stale
files in pre-commit and CI. CI type-checks the output with `tsc --strict`.
Consequences: no Node dependency for the Python toolchain; supports only the JSON-schema subset
our models use (which is enforced by tests).

## D-005 Literature parameters carry their source  (2026-10-20, phase 1)
Decision: config values from the literature use `SourcedValue {value, source, note}`.
`source: UNVERIFIED` marks unconfirmed values; `common.config.find_unverified` lists them so
experiment metadata and docs can flag them automatically (research-integrity rule 3).

## D-006 Answers to the kickoff questions  (2026-10-20, phase 2)
User answers: Q1 study area = Pune Municipal Corporation only. Q2–Q5 = defaults from the master
plan. Q6 compute budget = "depends on needs". Q7 = install tooling (done: GNU make 4.4.1 via
winget, ffmpeg 9.0.2 via winget, pnpm 9 via `npm i -g`; `corepack enable` needs admin rights).
Consequences:
- Status quo (B0): one 112 call, then sequential hand-off to ambulance (108), police and fire
  (101). Each agency picks its nearest unit. Hand-off delays are configurable, marked UNVERIFIED,
  and swept in the sensitivity analysis.
- Fleet sizes: synthetic defaults scaled from facility counts (phase 6/7).
- Equity non-inferiority margin: **δ = 0.02** on the Gini coefficient of ward-level p90
  time-to-last-required-arrival (FULL vs B0, one-sided paired test). Fixed now, before any
  experiment runs.
- Compute: experiments get a `--quick` profile (short horizon, few reps) for development, plus
  the full confirmatory profile. Runtime is measured and reported, not assumed.

## D-007 Study area = PMC 2022 electoral wards (not the OSM place query)  (2026-10-20, phase 2)
Context: `"Pune, Maharashtra, India"` geocodes to Pune District (15,641 km²). OSM has no PMC
boundary relation (only "Pune City Subdistrict", 312 km², which excludes the villages merged in
2017/2021).
Decision: wards = PMC 2022 electoral wards (58 prabhags) from DataMeet (traced from the PMC
ward PDFs). Study boundary = their union (506.9 km²). Roads and facilities are downloaded within
that union buffered by 1 km, so routes and stations just outside the line exist. Nodes outside
every ward take the nearest ward's id and `in_city = false`. Incidents are generated only on
in-city nodes. `network.bbox` overrides the area (e.g. to add PCMC later).
Alternatives: H3 hexes (kept as `wards.method: h3` fallback); Pune City Subdistrict (too small).
Consequences: ward geometry is a manual trace (positional error of tens of metres is likely);
licence CC BY-SA 2.5 IN, so the data stays out of git and a download script is committed instead.

## D-008 Base speeds by road class, not OSM maxspeed  (2026-10-20, phase 2)
Context: `maxspeed` tags are sparse in Pune, and legal limits overstate urban travel speeds.
Decision: base speed = typical off-peak urban speed per road class (configured), capped by a
tagged maxspeed. Time-of-day multipliers per class (AM peak 9–11, PM peak 17–21, night 23–6),
a network-wide rain multiplier and vehicle profiles (ambulance/police/fire) are applied on top.
All are UNVERIFIED and are exactly the parameters the user is asked to sanity-check.

## D-009 Compact network + static-bucket routing  (2026-10-20, phase 2)
Decision: besides the canonical graphml, the network is cached as two parquet tables and
routed with scipy's Dijkstra on a CSR matrix built per (hour, profile, scenario-state) and cached.
Parallel edges collapse to the fastest one. `shortest_times` uses each edge's time at departure
(static bucket); exact time-dependent routing is phase 6.1.
Consequences: loads in well under 10 s and one-to-all queries take milliseconds, which the
simulator needs.

## D-010 Flooding from a river buffer, no SRTM  (2026-10-20, phase 2)
Decision: an edge is flood-prone if its end nodes or midpoint are within `buffer_m` of the
Mula, Mutha or Mula-Mutha river lines (OSM). Each prone edge has a fixed vulnerability
u ~ U(0,1) (scenario seed), and the flood stage depends on rain accumulated over a trailing
window. Configs are validated so that more rain can never reopen or speed up a road.
Alternatives: SRTM low-elevation cells (more data, 30 m resolution too coarse for urban streets).
Consequences: nalas (storm-water streams) that also flood in Pune are not modelled; recorded as
a limitation.

## D-011 Ganeshotsav routes by OSM street name  (2026-10-20, phase 2)
Decision: procession routes = edges whose OSM name contains Laxmi/Lakshmi Road, Tilak Road,
Kumthekar Road, Kelkar Road, Shivaji Road or Bajirao Road, plus optional hand-drawn polylines.
Evening crowds slow them on days 0–5, and the immersion day (day 6) closes them 09:00–24:00.
Dates and street list are approximate and must be checked against the 2026 calendar and Pune
Traffic Police diversion notices.

## D-012 Requirement rules are monotone by construction  (2026-10-20, phase 2)
Decision: `configs/requirements.yaml` rules may only require a cue to be present (or a minimum),
and matches combine by element-wise max. Every incident type has an unconditional base rule.
Property tests check that adding a cue never reduces any agency's requirement.
The same functions produce ground truth (sim) and estimates (vision). Estimation error will be
modelled explicitly in phase 7 (master plan §6.7).

## D-013 Ambulance bases = all OSM hospitals (candidate set)  (2026-10-20, phase 2)
Context: OSM tags 591 features as `amenity=hospital` in the study area, many of them small
nursing homes.
Decision: every snapped hospital is a *candidate* ambulance base. The fleet configuration
(phase 6/7) decides which bases hold units. Hospital capabilities are synthetic: "major" if
the OSM footprint ≥ 5,000 m² or `beds` ≥ 100, else "minor", with tier-specific probabilities.

## D-014 Speed calibration against a reported TomTom rush-hour speed (approximate)  (2026-10-20, phase 2)
Context: with the first defaults (trunk 45 … residential 20 km/h), route-weighted average speeds on
the 7 sanity routes were 47.6 / 22.0 / 37.9 / 20.3 km/h for departures at 03:00 / 10:00 / 14:00 /
19:00 (`results/geo/sanity_travel_times_precalibration.csv`, reproducible with
`configs/archive/geo_speeds_precalibration.yaml`). That is faster than Pune traffic.
Reference: Punekar News, "Pune Ranks Second in India, Fifth Globally in Traffic Congestion: TomTom
Report", 22 January 2026, reporting TomTom Traffic Index 2025 data for Pune: 33 min 20 s per 10 km,
rush-hour speed 15.1 km/h, congestion level 71.1 %. The TomTom page itself
(https://www.tomtom.com/traffic-index/pune-traffic/) is JavaScript-rendered and was not read.
Decision: base speeds (trunk 28, primary 22, secondary 19, tertiary 16, residential 13, service
10 km/h) and the peak and night multipliers were tuned by hand. The aim was route-weighted averages
near 15 km/h for the 10:00 and 19:00 departures; the result is 16.3 and 15.0 km/h
(`results/geo/sanity_travel_times.csv`).
What this does and does not show:
- Only the rush-hour figure is matched. The model's overall time is 26.0 min per 10 km (all 24
  departure hours, sanity routes) against 33 min 20 s reported. These quantities differ: TomTom
  averages real trips, which are weighted toward busy hours. Still, the model is probably faster
  than reality outside the peaks.
- The calibration used the same 7 routes as the sanity check, so that table is no longer an
  independent check. An independent check needs hold-out routes with dated maps-service times
  (open action for the user).
- All values stay UNVERIFIED. This is order-of-magnitude calibration against one reported
  aggregate, not validation.
Also: the compact network stores only the OSM `maxspeed` cap and computes speeds from config at
load time, so changing a configured speed needs no rebuild. (Parallel-edge collapsing at build
time still uses build-time speeds; the effect is negligible.)
Consequences: response-time results depend on these speeds, so a global speed scale is a
phase-10 sensitivity factor.

## D-015 Ward-equity metric re-specified (supersedes the metric in D-006; δ unchanged)  (2026-10-20, phase 2)
Context (research-auditor, phase 2): the synthetic generator gives about 168 incidents per normal
week over 58 wards (median about 2.3 per ward, about 7 wards with none in a typical replication).
A per-replication ward p90 of last-arrival time, as pre-registered in D-006, is undefined or pure
noise. This was found before any experiment was designed or run.
Decision:
- **Primary equity metric:** ward *full-response accessibility*. For replication r and ward w,
  A(w, r) is a time average over the post-warm-up horizon, sampled every 15 simulated minutes, of
  the demand-weighted mean over the ward's in-city nodes of the expected time until the last
  required agency could arrive. That time is Σ_k P(k) · max over agencies a in k of the travel
  time of agency a's nearest available unit, where k runs over requirement profiles. Profile
  probabilities P(k) are estimated from the generator using the **forecast-training seed set**
  (not the evaluation seeds). Equity(r) = Gini over the 58 wards of A(w, r).
- **Test:** one-sided paired non-inferiority, FULL vs B0, on replication-level Equity under common
  random numbers, margin δ = 0.02 (as agreed in D-006).
- This is defined for every ward in every replication, whatever the incident counts.
- **Secondary (descriptive):** incident-based ward last-arrival times pooled over replications,
  with bootstrap-over-replication 95 % CIs. Wards with fewer than 20 pooled incidents are flagged.
- A power simulation on the tuning seed set will be run and recorded before the confirmatory
  experiments (phase 10).
Consequences: the simulator (phase 7) must log unit availability snapshots. The user should
confirm this re-specification (it changes a pre-registered metric, albeit before any run).

## D-016 Response to the phase-2 research audit  (2026-10-20, phase 2)
Fixed in phase 2:
- Equity metric re-specified (D-015).
- Every behavioural parameter in `incidents.yaml`, `requirements.yaml` and the scenario files is a
  `SourcedValue` (type mix, feature probabilities, vehicle mix, hotspot weights, rain cap,
  severity base/weights, rule-table provenance, flood stages, festival multipliers and type mix).
  `find_unverified` now lists them all.
- D-014 reworded with a dated citation. The pre-calibration speeds are archived, and both sanity
  tables are regenerated by script with `.meta.json` sidecars (git commit, config hashes, seeds,
  synthetic flag). The generator CLI writes sidecars too.
- Flood vulnerability is drawn per road (both directions share it), not per directed edge.
- Festival streets use the exact OSM names ("Lokmanya Tilak Marg", "R. B. Kumthekar Marg",
  "Narsimha Chintaman Kelkar Marg"), restricted to 2.5 km of Mandai. Match counts are recorded,
  zero matches are logged, and a real-data test fails if any street matches nothing.
- Labelling: ambulance bases are `source: synthetic` ("assumed"), `cameras.yaml` says the sites
  are algorithm-chosen and not real CCTV locations, synthetic-scenario figures carry `_synthetic`
  in their names, and the `base_rate_per_day` note says ASSUMPTION.
- Common random numbers: arrivals use a per-hour substream and marks a per-incident substream. A
  test checks that changing `p_person_down` leaves onsets, nodes and types unchanged. Disjoint seed
  bases exist for evaluation, tuning and forecast training (`configs/base.yaml`).
- Leakage guard: `tests/test_architecture.py` fails if any package outside `sim/` imports
  `resqcity.sim.groundtruth` or `resqcity.sim.generator`, and it enforces the master-plan module boundaries.
- Hotspot weights count each road once (two-way roads were double-counted).
- OSM re-downloads are pinned to `osm_snapshot_date` (Overpass attic query).
- Added a dispersion test (variance ≈ mean).
Deferred (recorded, owner phase):
- The same requirement-estimation error model must apply to B0 call triage and to CCTV
  (phases 4 and 7).
- Spatial demand follows OSM road density, not observed crash locations, so equity results partly
  reflect mapping density. Listed in `docs/limitations.md`; an alternative-weighting sensitivity
  run is planned for phase 10.
- Only 14 fire stations are in OSM; check against the PMC fire-brigade list (user action).
- 40 stations lie outside the city and carry the nearest ward's id. Filter on `in_city` before any
  ward-level fleet scaling (phase 6/7).

## D-017 Repository layout: single `src/resqcity` package  (2026-10-20, between phases 2 and 3)
Context: the root held 12 top-level Python package folders (8 of them empty placeholders) next to
configs, data, docs and the kit files; generic top-level import names (`common`, `tools`) could
also clash with other installed packages. The user chose the src-layout option.
Decision: all Python code lives in `src/resqcity/` and is imported as `resqcity.<module>`
(`python -m resqcity.geo.build`, `python -m resqcity.tools.tasks test`, ...). Module folders are
created in the phase that implements them (no empty placeholders). `BUILD_PROMPT.md` and
`SETUP.md` moved to `docs/kit/`. Experiment configs will live in `configs/experiments/` and the
Colab training notebook in `notebooks/vision/` (the kit prompts still show the old paths;
CLAUDE.md, the master plan and the project skills are updated). `PROJECT_ROOT` is found by walking
up to `pyproject.toml`, so it no longer depends on package depth. `tests/test_architecture.py`
scans `src/resqcity/` and asserts it actually found the modules, so boundary checks cannot pass
vacuously.
Consequences: earlier entries and phase plans keep their original paths as historical record.
Committed result sidecars before this change name the old module paths in `command`.

## D-018 User answers to the phase-2 questions; cited base incident rate  (2026-10-20)
User answers:
1. The re-specified equity metric (D-015) is **accepted**.
2. The sanity travel times look "close to correct". The user asked for a Google Maps check and
   will fill in typical times manually on a template (`configs/maps_check_routes.yaml` →
   `results/geo/maps_check_template.csv`). Hold-out routes not used in calibration are included.
3. No own fire-station list. The PMC open-data map of fire stations (data.opencity.in,
   public domain) lists **14 stations**, which matches the 14 found in OSM.
4. No own incident figures. The base rate was replaced by a derivation from cited sources:
   - Fires: mean "Fire" calls 2015–2019 = 1,107 per year (PMC Fire Call Statistics 2011–2020, OpenCity /
     PMC, public domain; 2020 excluded as a COVID year).
   - Road accidents: 334 deaths in Pune city in 2023 (city police via Punekar News, 2024-01-13)
     ÷ national severity index 36.5 deaths per 100 accidents (MoRTH, Road Accidents in India 2022)
     ≈ 915 per year.
   - The result is 5.54 incidents/day (was 24/day, an assumption), with type mix accident 0.44,
     fire 0.53 and crowd/other 0.03 (assumed).
   Caveats: the national severity index probably under-estimates city accidents, and registered
   accidents miss unreported ones. The phase-10 sensitivity sweep covers 0.5×–4× the rate. A lower
   rate means less queueing, so benefits of joint dispatch will come mostly from coordination, not
   capacity. That is reported, not tuned away.
   An unverified press figure of "1,400 road accidents in Pune in 2024" (attributed to a Rajya
   Sabha speech) could not be confirmed from the primary source and is not used.

## D-019 Phase-3 permissions and choices  (2026-10-20)
The user allowed: downloading public datasets that need no account or sign-up (cap ≈ 10 GB; D: had
23 GB free), overnight GPU training on the RTX 4050, and starting Docker Desktop and pulling MediaMTX.
Ultralytics YOLO (AGPL-3.0) is accepted for this research project. The vision stack is the
dependency group `vision` (PyTorch CUDA 12.6 wheels). It is in uv's default groups, so `uv sync` and
`make setup` keep it, while CI skips it with `--no-group vision`. (An optional extra was tried first,
but a plain `uv sync` uninstalls extras.) "MADP" (user link: github.com/MADPToolbox/MADP) is the
Multiagent Decision Process Toolbox, a planning library, not a video dataset. It is dropped as a
vision dataset and may be cited as related work for multi-agent decision making.

## D-020 Vision datasets actually used, and the split policy  (2026-10-20, phase 3)
Downloaded (no account; total ≈ 5.6 GB + D-Fire, under the 10 GB cap):
- UCF-Crime (HF mirror of the original zips): all 150 `RoadAccidents` videos plus 60 seeded-random
  normal test videos (one unreadable, so 59 used), extracted member by member via HTTP range
  requests, with CRC checks. The official temporal annotations give accident onsets for **23**
  accident videos.
- D-Fire (HF mirror) for fire/smoke. The class-id mapping is inferred from the README box totals
  and checked in `dataset_check.json`.
Not obtained:
- CADP frames: Google Drive refuses automated download ("cannot retrieve the public link"). The
  annotation JSON was downloaded. Its keyframe `frame` values are non-integer (e.g. 22.03), so
  the loader reads them as seconds; this is unverified until the frames are available. The user
  can download the frames in a browser into `data/raw/vision/cadp/frames/` and re-run
  `resqcity.vision.datasets.annotations`.
- TAD (no public release found) and IEEE DataPort (needs an account).
Split policy: the onset-annotated accident videos are the only ones that measure detection delay,
so they go only to val (11) and test (12) (`onset_videos_eval_only`). The classifier learns from
the other 127 accident videos through multiple-instance learning (weak labels; Sultani et al.,
2018), with normal videos as negatives. Splits are by video, stratified and seeded.
Consequences: accident evaluation rests on one dataset (UCF-Crime: CCTV, mixed countries, low
resolution, 320×240). Per-dataset reporting currently has one dataset. The detection-delay
estimate uses 12 test videos, so its uncertainty is large and is reported as such.
Domain shift to Pune CCTV is a threat to validity (docs/limitations.md L10).

## D-021 Clip-classifier model selection by within-video discrimination  (2026-10-20, phase 3)
Finding (first run `clip_main`): validation window AUROC was 0.93, but within onset-annotated
accident videos (pre-accident vs. accident windows of the same scene) it was only 0.66, and 0.62
on test. Mean scores were 0.08 on normal videos and 0.53 on pre-accident windows. The overall AUROC
mostly reflects the scene difference between UCF-Crime's normal videos (shops, indoors) and its
road videos, which says little about a road camera, where every scene is a road.
Decision:
1. Model selection uses the VALIDATION within-video AUROC (`val_auroc_within`), not the overall one.
2. The MIL objective gets Sultani et al.'s sparsity and temporal-smoothness terms (weights in
   config). Three settings were trained (`configs/vision_sweeps/`):
   (sparsity, smoothness) ∈ {(0,0), (0.3,1), (1,1)}.
3. The selection is scripted (`resqcity.vision.select_clip` → `results/vision/clip_selection.csv`).
   `clip_s03_t1` was selected: val within-video AUROC 0.736 (vs 0.669 / 0.673), val overall 0.945.
   The test split was NOT looked at for this choice; it is evaluated once, after selection.
Consequences: the three sweep runs and the discarded first run stay recorded under
`results/vision/`. The within-video AUROC is reported alongside the overall one in every result,
because it is the metric that matters for false alarms on road cameras.

## D-022 Accident-detector evaluation: two versions, disclosed  (2026-10-20, phase 3)
v1 (`results/vision/clip_s03_t1/eval_v1/`): the threshold was chosen on val, with the alarm
persistence fixed at 2 windows. Test (combined): video recall 0.38, onsets detected 3 of 12,
false alarms 3.6 per camera-hour.
After seeing v1 we diagnosed it on VAL: the alarm persistence (consecutive windows) matters as much
as the threshold. v2 chooses (threshold, persistence) jointly on val, with persistence ∈ {1, 2, 3, 4},
max video F1 within a budget of 1 false alarm per camera-hour. v2 adds Wilson 95% intervals for
recall and exact Poisson intervals for false-alarm rates.
v2 test (combined, threshold 0.60, 4 windows): video recall 0.35 [0.22, 0.51], onsets detected
1 of 12, false alarms 7.2 per camera-hour [95% upper bound 18.3] on only 0.56 h of normal footage.
Window AUROC is 0.98 overall but 0.61 within videos; ECE is 0.013.
**Disclosure:** v2 was adopted after the v1 test result had been seen, so the v2 test figures are
not strictly held-out. Both versions are kept and reported. Neither meets the 1 false alarm per
camera-hour budget with useful recall. The honest summary: on UCF-Crime this detector is
well calibrated at window level but localises accidents poorly in time.
Diagnosis: several videos alarm at t = 6 s, the earliest time 4 consecutive 3 s windows allow,
i.e. the scene is flagged as accident-like from the start. Track heuristics are weak at 320×240
resolution: collision cues almost never fire, and deceleration fires in 25% of accident frames vs
12% of normal frames.
Live check (`resqcity.streams.acceptance`, RTSP through MediaMTX, the val operating point): 2 of 4
accident clips detected (C02 +3.0 s, C03 +2.6 s after onset), 0 false alarms in about 0.18
camera-hours. This is consistent with the offline result.
Consequences for the research: the simulator reads detector behaviour from
`results/vision/sim_inputs.yaml` (measured, with intervals). Because measured recall is low and
uncertain, phase 10 must treat detector recall, delay and false-alarm rate as sensitivity factors,
and report the break-even detector quality at which CCTV-triggered dispatch beats call-only
dispatch. Improving the detector (more onset-annotated data, e.g. CADP frames downloaded by hand;
higher-resolution footage; a pretrained video model) is future work. It is not tuned further on
this test split.

## D-023 Privacy filter: whole-person blur, minimum region size; verified numerically  (2026-10-20, phase 3)
Whole-person pixelation was replaced by a strong Gaussian blur (kernel = 0.6 × region size), so
every privacy region is checked with the same metric (variance of the Laplacian). Regions smaller
than 24 px are grown to 24 px, because a blur kernel cannot remove detail from a 3-pixel-wide box.
The first privacy-check runs failed because of a measurement bug: outline rectangles were drawn
before the next region was measured. It was fixed by measuring first. The check on a real test frame
passes: 14 regions (1 face, 5 plates, 8 persons), detail reduction ≥ 98.9% in each
(`results/vision/privacy_check.json`, `.png`). Only the anonymised frame is ever written.

## D-024 Run GPU jobs one at a time on the 6 GB laptop GPU  (2026-10-20, phase 3)
One attempt ran fire training, four live vision processes and the privacy check together. GPU
memory and system RAM ran out (about 6 GB of RAM free beside the user's other containers), which
killed all of them. Since then, GPU jobs run sequentially, the live acceptance watches two cameras
per run, and fire training resumes from `last.pt` (`train_fire --resume`). CPU latency is measured
with nothing else running: 44 ms per frame, 22.5 fps on CPU for 320×240 frames.

## D-025 Response to the phase-3 research audit; evaluation v3  (2026-10-20, phase 3)
The audit found two blockers in the v2 export: (a) "recall" counted any alarm in accident videos
without onsets (12 of 13 detected videos), so timing was not checked; (b) the false-alarm rate
pooled road footage before onsets (where all 4 false alarms fell, in 0.032 h) with mostly
non-road UCF "Normal" footage (0.53 h). Both are fixed in v3:
- **Detection rule (fixed before running v3):** an alarm in [onset, offset + 10 s] is a detection.
  Alarms before onset are road false alarms. Alarms later than offset + 10 s are "late"
  (neither). The same rule is used for the live acceptance run.
- False alarms are reported separately for road and non-road footage, with exact Poisson and
  video-clustered bootstrap intervals. Only the ROAD rate is exported to the simulator.
  "any_alarm_rate" (no timing check) is reported, not exported.
- **Operating point:** the max of Youden's J on val videos (onset-detection rate minus the share of
  videos with a false alarm). The 1/h budget could not be resolved with ~0.37 h of val footage.
- Added: per-video AUROC within onset videos with bootstrap intervals over videos, and
  calibration within onset videos and above the operating threshold, with model hashes in
  `sim_inputs.yaml` and `report.json`.
**v3 development-test results** (combined; val operating point threshold 0.45, 4 windows;
`results/vision/clip_s03_t1/`):
- onset detection 3 of 12 = 0.25 (Wilson 95% interval 0.09–0.53); median delay 6.0 s;
- road false alarms 4 in 0.032 h = 126/h (Poisson 34–322; video-clustered bootstrap 30–233);
- non-road false alarms 0 in 0.53 h (≤ 7.0/h);
- any-alarm rate on unannotated accident videos 0.72 (0.52–0.86);
- per-video AUROC within onset videos 0.74 (0.53–0.91) on test and 0.65 (0.47–0.80) on val;
- calibration: ECE 0.013 over all windows (driven by easy normal windows), but ECE 0.23 within
  onset videos. Above the threshold the mean predicted probability is 0.73 while 0.54 are
  accidents, i.e. **over-confident where alarms fire**. "Well calibrated" in D-022 is withdrawn.
  Phase-4 fusion must not treat this confidence as a calibrated likelihood without correction.
**Corrections to earlier entries.** D-021 said the test split was not looked at; in fact the
first run's within-video TEST AUROC (0.62) was seen before the selection criterion was changed
and the MIL sweep was run. D-022's heuristic-cue figures (25% vs 12%) were per-video averages
computed ad hoc. The scripted frame-level figures are in `results/vision/diagnostics.json`
(deceleration cue 18.1% of accident-video frames vs 3.4% of normal-video frames). Every number
quoted in D-021/D-022 now comes from that file or the evaluation outputs. The run selected on val
(`clip_s03_t1`, val within-video AUROC 0.736) has a lower test within-video AUROC (0.572) than
`clip_s1_t1` (0.668): with about 11 val videos the selection is noise, not a significant
improvement.
**Status of the UCF-Crime test split:** it has been looked at three times (v1, v2, v3), so it is
reported as a DEVELOPMENT-TEST set. A held-out claim needs a frozen pipeline (this commit) and
footage never used (e.g. CADP frames downloaded by hand, or Pune footage). v1 and v2 are kept with
ERRATA notes (`eval_v1/`, `eval_v2/`).
**Other fixes:**
- MediaMTX is bound to 127.0.0.1:8554 only (HLS/API ports removed), because the raw streams are
  un-anonymised. Stream URLs use 127.0.0.1 explicitly, since `localhost` resolved to IPv6 and hung.
  Live sources have 10 s open/read timeouts.
- Demo "normal" clips are now road footage (pre-onset parts of other onset videos), so a scene
  change cannot pass for an accident. Live JSON records the operating point and model hashes.
  Live v3 run (`results/vision/live_acceptance_C0*.json` as committed in 2774191; later runs
  overwrite these files): 2 of 4 accident clips detected (C02
  +3.8 s, C03 +1.9 s). There were 8 false events (7 accident, 1 fire_smoke) in 0.18 camera-hours
  of road footage. The earlier "0 false alarms" came from non-road normal clips. This is
  consistent with the offline road false-alarm rate.
- The privacy-check frame (UCF-Crime-derived) is no longer committed. It was removed from the
  local, unpushed history. The privacy check verifies the blur MECHANISM on the regions the
  anonymiser chose; it cannot measure missed faces/plates/persons. A coverage check needs
  hand-labelled frames (open action for the user, L12).
- Base rate: computed by `resqcity.sim.base_rate` from `configs/base_rate.yaml`
  (`results/calibration/base_rate.json`). The 3% crowd/other share is now ADDED on top of
  accidents + fires (5.54/day → 5.71/day), and only the "Fire" call type is used (documented).
  Year and area mismatches are listed as caveats. MoRTH's million-plus-city table for Pune
  (open action) would replace the national severity-index proxy.
- Latency: measured on road footage (320×240), including the fire model: 77.8 ms per frame,
  12.8 fps on CPU (`results/vision/latency_cpu.json`). That is real-time at 5 fps, about two
  cameras per laptop CPU (the D-024 figure of 44 ms excluded the fire model).

## D-026 Phase-2/3 follow-ups after the user's inputs  (2026-10-20, before phase 4)
User inputs: a Google Maps sheet was filled in (10 routes × 3 departure times); CADP frames are not
available; MoRTH "Road Accidents in India 2022" was provided (PDF); the user wants clear images and
videos (chose "blur faces + plates only" and "no blurring at all") and a stronger accident model.
1. **Midday speeds calibrated to Google Maps; peak speeds kept** (`resqcity.geo.calibrate_speeds`,
   `results/geo/speed_calibration.json`). The user's sheet is kept as
   `results/geo/maps_check_google_2026-10-20.csv`; the template refuses to overwrite filled data.
   - Step 1 (pre-specified: fit on 4 routes, hold out 6) over-corrected: hold-out median error
     went from 11.5% to 19.9%, because the 4 fit routes are the most congested central routes.
   - Step 2 (added after seeing step 1): leave-one-route-out (LORO) over all 10 routes. Pooled
     median error 17.2% vs 19.3% for the baseline, and the all-route factors were first adopted.
   - Step 3 (added after the independent audit, item 7): the pooled gain came from 14:00 only.
     Per hour, LORO median error / share inside Google's range: 10:00 14.6% → 13.5% / 90% → 80%;
     14:00 28.9% → 17.8% / 50% → 70%; 19:00 8.6% → 17.2% / 80% → 90%. A period now changes
     only if LORO improves both measures at its hour, so **only a new MIDDAY period (11–17) is
     adopted** (multiplier 0.6714 ≈ 1/1.49). am_peak and pm_peak keep their D-014 values
     (UNVERIFIED).
   - The evidence for midday does not depend on fitting: at 14:00 the baseline is faster than
     Google's midpoint on 10 of 10 routes (sign test p = 0.002), vs 7 of 10 at 10:00 and 19:00
     (p = 0.34). With 10 routes the route-level bootstrap intervals of the LORO gain include
     zero at every hour (14:00: +11.1 points, 95% −9.2 to +25.8), so the error reduction itself
     is not established. Steps 2 and 3 were added post hoc.
   - Shoulder hours (06–09, 21–23) run at base speed, which is probably too fast given midday;
     night is not calibrated (L2).
2. **Base incident rate from the primary source** (`resqcity.sim.morth_extract` →
   `results/calibration/morth_2022_extract.json`, `resqcity.sim.base_rate`):
   - Pune 2022: 871 accidents, 325 killed (MoRTH Table 6.2), replacing the 334/0.365 proxy;
     5.59 incidents/day (fires 1,107/yr + accidents 871/yr, plus 3% crowd/other).
   - Registered crashes in Pune are severe (37.3 killed per 100 accidents vs 22.3 for all 50
     million-plus cities; both computed in `results/calibration/base_rate.json` from Table 6.2,
     whose total row is checked against the city rows), so non-fatal crashes are probably
     under-registered; the rate is likely a lower bound.
   - The **hour profile** now comes from MoRTH Annexure 42 (Maharashtra, 3-hour intervals)
     instead of an assumption. It describes road accidents; fires (54% of incidents) use their
     own profile entry (`type_hour_profiles.fire_smoke`), currently the same values and marked
     UNVERIFIED. Pune weather split (Annexure 45): 82 of 871 accidents (9.4%) in
     rain, recorded but not used (no rain-exposure hours).
   - The Pune vehicle-type table (Annexure 46) is not machine-readable; the vehicle mix stays
     UNVERIFIED.
3. **Ganeshotsav dates verified:** Anant Chaturdashi is Fri 25 Sep 2026 (day 6). The scenario is
   now 8 days, and the immersion closure continues on day 7, 00:00–14:00 (approximate; Pune's
   procession usually runs 24–30 h).
4. **Privacy modes (user request; CLAUDE.md rule 6 updated with the user's approval):**
   - `faces_plates` is the new default: faces plus a narrower plate region (60–100% height,
     10–90% width of each vehicle box); vehicles, people and the crash stay clear.
   - `full` additionally blurs people (the previous behaviour).
   - `off` means unblurred, and is allowed only for public-dataset footage: local files under
     `data/raw/vision` or `data/videos`, or the simulated cameras (`rtsp://127.0.0.1:8554/C01`
     … `C04`, kept in sync with `configs/streams.yaml` by a test). Every other source (a phone or
     IP camera, any other file, a loopback relay under another name) is refused when a
     `CameraPipeline` is built, and the preview is labelled "UNBLURRED". Footage deliberately
     copied into those folders, or published under a simulated camera's name, cannot be
     detected by code.
5. **Stronger accident model attempted; it did not help** (evaluation v4, `results/vision/crossfit_v4/`).
   - Candidates: the per-frame MobileNetV3 head and a Kinetics-400 R(2+1)D-18 video backbone
     (16-frame clips), each with 3 MIL settings.
   - Two-fold cross-fitting: each fold chooses early stopping, temperature, candidate, stacker
     and operating point on its fit split only (`train_clip --fit-split`), and evaluation
     predictions are pooled.
   - The video backbone scored lower in both folds (fit-split within-video AUROC 0.56–0.64 vs
     0.67–0.74 on val; 0.41–0.47 vs 0.60–0.70 on test), so it was never selected.
   - Pooled result (combined, all 23 onset videos): onset detection 6/23 = 0.26 (Wilson 0.13–0.46),
     median delay 6.3 s, road false alarms 5 in 0.075 h = 67/h (Poisson 22–156), non-road 3.5/h,
     per-video AUROC 0.73 (0.60–0.85; 20 onset videos with both classes of windows).
   - The folds may choose different detectors (A: image s03_t1, B: image s1_t1), so the pooled
     numbers describe the cross-fitted PROCEDURE. The **deployed detector** (fold A) on its own
     evaluation split: 3/12 onsets (Wilson 0.09–0.53) and 4 road false alarms in 0.032 h = 126/h
     (Poisson 34–322) (`results/vision/crossfit_v4/per_fold_metrics.csv`).
   - Still over-confident above the threshold (0.67 predicted vs 0.46 observed).
   - `sim_inputs.yaml` exports the pooled v4 estimates and, separately, the deployed detector's
     fold-A estimates; the live detector is fold A's choice (`results/vision/deployed_detector.json`).
   - **Constraint for phase 4:** fusion must use the measured event-level detection and
     false-alarm rates (likelihood ratios) from `sim_inputs.yaml`, not the window confidence as a
     probability.
6. CADP is unavailable, so no never-seen footage exists. v4 cross-fitting is the best available
   estimate and remains a development estimate (L15).
7. **Response to the independent audit of items 1–6** (research-auditor, 2026-10-20). All
   changes below were made after seeing the data and are post hoc:
   - Speeds: per-hour LORO, sign test and bootstrap added; only midday adopted (item 1). The
     earlier claim that calibration "removed the bias" is withdrawn: a LORO mean log-ratio of
     zero holds by construction for a per-hour geometric-mean fit.
   - Detector: per-fold metrics reported; the deployed detector's own estimate (fold A) is weaker
     than the pooled one (item 5). `crossfit` now checks that every run was fitted on its fold's
     split (fold-A runs predate the `fit_split` field and were fitted on val, then hard-coded).
   - Live acceptance re-scored. The earlier v4 live result (3 of 4 detected, in commit 26c8ad3)
     was inflated: one "detection" (C01) came about 4.5 s after the 11.6 s accident clip had
     ended, i.e. it was triggered by the cut back to normal footage, and a fire/smoke event was
     counted as an accident false alarm. Now only `accident` events are scored, the detection
     window also ends when the accident clip ends, and false alarms within 8 s of a scene cut
     (clip switch or loop, each forcing an RTSP reconnect) are flagged. Re-run
     (`results/vision/live_acceptance_C01_C02.json`, `..._C03_C04.json`): 2 of 4 clips detected
     (C02 +2.1 s, C03 +4.9 s); 7 accident false alarms in 0.23 camera-hours, 5 of them within 8 s
     after a scene cut; 1 fire/smoke event. Offline, fold A detects 1 of these 4 test clips
     (RoadAccidents010); live timing and window alignment differ, so the live run checks the
     end-to-end path, not accuracy.
   - Fire hour profile flagged UNVERIFIED (item 2); privacy guard tightened and moved into
     `CameraPipeline` (item 4); Wilson upper bound corrected to 0.46; the million-plus severity
     figure is now extracted, not hand-computed.

## D-027 Phase 3b: detector upgrade, evaluated once on never-seen footage  (2026-10-21…23)
Plan and pre-registration: `docs/plans/phase-3b.md` (committed f7202d8 before any ACCIDENT data
was downloaded). Amendments 1–3 were committed before any holdout result, but amendments 2 and 3
were written AFTER seeing validation results (disclosed in each). All numbers below are from
`results/vision/v5/` (`stage_a.csv`, `stage_c.csv`, `stage_b.csv`, `report.json`,
`exploratory.json`, `alarms_final.csv`, `armed_time_per_unit.csv`, `verifier_smoke/`) and
`results/vision/uca_label_check.json`. The stage-B timeline before 16:52 IST (item 4) rests on
the working-session log, not on a committed file.
1. **Data.** ACCIDENT (Picek et al., 2026): 2,027 real CCTV crash clips, official IID split; the
   1,520 test clips were sealed (labels in a separate file, read once by the final run:
   `holdout_access.log` has one entry). UCA timestamps gave training intervals for 143 UCF-Crime
   accident videos. UA-DETRAC (100 sequences, 1.56 h of normal road CCTV) was added for false
   alarms (amendment 2). A UCF normal video that became readable was excluded so the v1–v4 splits
   stay identical; v4 reproduces bit-for-bit.
2. **Stage A (validation, 111 onset clips):** 12 candidates (MobileNetV3 / SigLIP2 / V-JEPA 2 ×
   MIL / UCA intervals × UCF / UCF + ACCIDENT training). Chosen: **V-JEPA 2 + MIL + UCF +
   ACCIDENT**, validation J 0.548 (best MobileNet 0.487, best SigLIP2 0.460); operating point
   0.30 × 4 windows. ACCIDENT training data raised J for every backbone and label type. UCA
   interval labels were below MIL for MobileNet and V-JEPA 2; for SigLIP2 they tied (0.350, UCF
   only) or were marginally higher (0.460 vs 0.450, UCF + ACCIDENT). With 111 validation clips,
   differences of a few hundredths are indicative only. Three UCF-only runs (MobileNet MIL,
   SigLIP2 MIL, SigLIP2 intervals) reach the same J, 56/111 detections and false-positive video
   rate, so J does not separate them; the SigLIP2 UCF-only "tie" says nothing about label type.
3. **Stage C (run before B, criteria unchanged):** COCO and UVH-26 (Indian vehicle classes) tie
   at J 0.548; COCO kept by the tie rule. The Indian detector made no measurable difference here.
4. **Stage B (verifier):** zero-shot Qwen3.5-2B and Qwen3-VL-2B cannot keep 90 % of validation
   detections (44 and 48 of 69 at the most permissive threshold); Gemma-4-E2B (3.45 s) and
   Molmo2-4B (3.26 s) exceed the 3 s latency limit; HolmesVAU-2B cannot run on transformers 5.
   **TAU-R1-Classifier** (traffic-camera fine-tune) is chosen at τ = 0.2: keeps 63/69, false
   alarms 13 → 10 (0.48 s). Amendment 3's fine-tuned Qwen3-VL-2B (1,108 train-split examples,
   one attempt) passes but is weaker (64/69, 13 → 11, 0.58 s). **Deviation:** amendment 3 was to
   apply only if no original candidate passed. When it was committed (12:11 IST), only the two
   zero-shot Qwen models had validation results; TAU-R1, Molmo2 and Gemma 4 had not been scored.
   The fine-tune was trained before they were; TAU-R1 then passed, so the condition was not met,
   but the fine-tuned candidate stayed in the comparison. It did not change the choice (TAU-R1
   wins by the rule either way). **Measurement fix:** in a later single-process run all
   candidates were loaded one after another, and later ones were slowed, probably by GPU memory
   left by earlier models (fine-tuned Qwen3-VL 10.2 s there vs 0.40 s alone,
   `verifier_smoke/qwen3_vl_2b_ft.json`), so every candidate was re-scored in its own process
   before the choice. That run and the timeline are in `stage_b_first_run_contaminated/`; the
   earliest Qwen results that triggered amendment 3 survive only as the figures quoted in it.
5. **Final run (once, ACCIDENT IID test + UA-DETRAC):**
   - Detection 925/1,520 = **0.61** (Wilson 0.58–0.63; bootstrap over 1,445 source videos
     0.584–0.631), median delay 2.8 s (v4: 0.26 on 23 UCF onsets). US clips 0.65, non-US 0.37.
     **This rate mostly reflects how the clips are trimmed, not the detector** (re-audit, item
     6): 542 of the 925 detections fire at 6.0 s, the first instant any alarm can fire, and a
     content-free baseline that alarms at 6.0 s on every clip scores 1,052/1,520 = 0.69.
   - UCF test split 6/12 (v4 fold A 3/12). This split has been looked at in every evaluation
     (v1–v5; development-test set since D-025): a consistency check, not a clean estimate.
   - Road false alarms as pre-registered (per camera-hour after the warm-up): ACCIDENT pre-impact
     footage 180 in 0.42 h = 433/h; UA-DETRAC normal traffic 50 in 1.39 h = 36/h (27–47); pooled
     127/h (111–145). **These units are flawed** (audit): each alarm is followed by a 60 s
     cooldown, so the rate per camera-hour is capped by the cooldown and depends on clip length,
     and the pooled figure mixes two very different sources. See item 6 for corrected units.
   - ACCIDENT temporal score T = 0.201 / 0.364 / 0.548 (σ 0.5 / 1 / 2 s). A constant guess (the
     median impact time of the train/val clips, 4.375 s) scores 0.194 / 0.369 / 0.604: **the
     detector does not localise the impact better than a constant**: most clips are trimmed so
     the impact comes early (holdout t* quartiles 3.3 / 4.8 / 6.9 s, 95th percentile 13.1 s).
     No comparison with the paper's VLMs is claimed.
   - Verifier: 1,295 checks, median 0.54 s.
6. **Exploratory (not pre-registered, `exploratory.json`; every alarm in `alarms_final.csv`):**
   every final-run alarm was reconstructed and matched to `per_clip.csv` (counts, detection and
   detection time) and `per_sequence_ua_detrac.csv` (counts; the final run stored no times there)
   before analysis. The final run saved its verifier cache before scoring UA-DETRAC, so those 61
   calls were re-scored with the same verifier, using no labels; every per-sequence count
   matched. The save order is fixed in `final_v5.py`.
   - **Clip-start firing** (found by the re-audit). The first alarm can fire at 6.0 s (3 s window
     + 3 strides). 542 of 925 detections and 157 of 180 pre-impact alarms (87 %) fire at exactly
     that instant. Detection therefore tracks the impact time: 0.58 / 0.74 / 0.78 for t* in
     (0, 3] / (3, 5] / (5, 7] s, but 0.35 / 0.29 / 0.15 for (7, 9] / (9, 15] / above 15 s. A
     content-free alarm at 6.0 s on every clip at least 6 s long would score 0.69 (US 0.72,
     non-US 0.53; 103 clips are shorter than 6 s); it would of course alarm on every normal
     camera too.
   - A pre-impact alarm hides the counted detection by construction (60 s cooldown): on the 281
     clips with enough lead time, 0 of 143 clips with such an alarm count as detected, against
     81 of 138 without. **Detection when the camera is armed at the impact** (lead time and no
     earlier alarm) is 81/138 = 0.59 (Wilson 0.50–0.67), non-US 11/31 = 0.35, median delay
     2.9 s. This is the closest estimate to a live camera, where there is no clip start. By
     chance alone (an armed camera alarming as a Poisson process in the 11 s detection
     interval) detection would be 0.15 at UA-DETRAC's 55 per armed hour, 0.21 at its upper
     bound and 0.27 at the ACCIDENT pre-impact rate after the first instant (101 per hour), so
     0.59 is well above chance. The armed subset is selected on "no earlier alarm" and may
     differ systematically from all clips.
   - Timing of pre-impact alarms: 43 % within 3 s of the impact and 66 % within 5 s, against
     67 % and 85 % under a uniform alarm-time null over each clip's armed interval. The alarms
     come EARLY (most at the first instant), not near the impact: there is no evidence that they
     anticipate the crash, and the earlier reading is withdrawn.
   - Rates per ARMED hour (warm-up and cooldowns excluded; clustered bootstrap by clip /
     sequence): UA-DETRAC normal traffic **55/h (39–77)**; 45 of 100 sequences alarm, 5 more than
     once, and 16 at the first instant, so some cameras trigger persistently. The ACCIDENT
     pre-impact figure (792/h) is not a hazard rate: it is about half of the 365 clips with
     pre-impact footage alarming at their first instant, spread over very short armed time. A
     camera that alarmed at every opportunity would give 60 alarms per camera-hour with the
     cooldown; the detector's long-run rate in normal traffic is 29.
7. **For phase 4 onwards:** `sim_inputs.yaml` (v5, re-exported with `final_v5 --export-only`, no
   labels read) now starts with `simulator_defaults` (exploratory, from `exploratory.json`);
   the pre-registered fields below it are kept for the record only:
   - detection probability when the camera is armed **0.59** (0.50–0.67), median delay 2.9 s;
     masking by the cooldown is produced by the simulator's own false-alarm process;
   - false alarms **λ = 55 per armed camera-hour** (UA-DETRAC), then a 60 s cooldown;
   - phase 10 ranges: detection 0.21–0.67 (lower: the non-US armed Wilson bound, below the
     0.24 of lead-time clips counting masked crashes as misses; upper: armed clips), λ 39–77 per
     armed hour, a stress case of one alarm per
     cooldown (60 per camera-hour), and a varied share of persistently triggering cameras.
     Pune is outside the US and the non-US armed estimate is 0.35 on 31 clips, so a low
     detection scenario is part of the base design, not an afterthought.
   These replace the earlier "0.61 / 127 per hour" inputs and the "36–433/h, 0.29–0.70" ranges.
   Fusion uses these rates as likelihoods, not the window confidence.
   **Overall reading:** on trimmed clips the detector detects FEWER crashes than a content-free
   alarm at clip start (0.61 vs 0.69) while alarming about half as often in normal traffic (29
   vs 60 per camera-hour); on an armed camera it detects well above chance (0.59 vs 0.15–0.27).
   In normal traffic it alarms about every two minutes per camera.
   The simulator results for CCTV-triggered policies must be read with that in mind.

## D-028 Phase 4 design choices: CCTV role, offline triage, unverified call parameters  (2026-10-24)
Plan: `docs/plans/phase-4.md` (approved by the user before any phase-4 code).
1. **Why CCTV evidence needs confirmation.** About 3–5 % of synthetic incidents fall within
   50–100 m of one of the 40 camera sites (one generated week; indicative), so a camera sees a
   real crash about once a year, against about 29 false alarms per camera-hour (D-027). A camera
   alarm alone is therefore real with probability of order 10⁻⁶. This is arithmetic on committed
   numbers, to be re-derived by the phase-4 week check, not a result.
2. **User decision: full model.** Fusion uses confidence-dependent likelihood ratios measured from
   the final v5 run (exploratory), an optional operator-review evidence step (sensitivity,
   specificity and review time UNVERIFIED, swept in phase 10), and phase 10 adds a break-even
   analysis of the detector quality needed for CCTV-first dispatch to pay off.
3. **User decision: offline LLM triage.** The optional LLM call triager runs a local model from
   the Hugging Face cache (caller text never leaves the machine, no API key). It is off by default;
   the keyword rule triager is the baseline, and tests and experiments never need the LLM.
4. **Call parameters are UNVERIFIED.** No source was found for crash-to-call delay in India,
   calls per incident or the share of prank calls; they live in `configs/calls.yaml` marked
   UNVERIFIED and are swept in phase 10. Two facts were confirmed at the source: Pune city police
   received 23,887 Dial 112 calls in July 2026 (Free Press Journal, 2026-08-03); Android's
   Emergency Location Service reports about 50 m accuracy, but in India only Uttar Pradesh was
   integrated at launch (press reports, December 2025), so 50 m is a scenario, not the default.

## D-029 Phase 4 results: calls, triage and fusion on the Pune network  (2026-10-24…26)
Plan: `docs/plans/phase-4.md`; design decisions and the user's choices: D-028. Formulation:
`docs/models/fusion.md`. Every number below is read from a committed results file:
`results/fusion/week_check_synthetic.json` (30 replications, 210 days, tuning seeds),
`triage_eval_synthetic.json`, `gate_k_synthetic.json` and `replay_*_synthetic.json`. All of it is
**synthetic**: the call process is UNVERIFIED throughout (L17); only the detector inputs are
measured (D-027). Revised 2026-10-26 after an independent audit; the fixes are itemised in item 9.
1. **Acceptance (BUILD_PROMPT phase 4).** Seven scripted scenarios on the real network and the 40
   camera sites replay correctly. Two camera views plus two callers merge into ONE incident. Two
   crashes **600 m** apart — the closest any two camera sites get, and just outside the 362 m gate
   for handset-located calls — stay separate; that is the real discrimination test. Two crashes
   **180 m** apart, inside the gate, are deliberately fused into one incident, which is a
   limitation of the model, not a success (L19), and `two_crashes_merged` pins it. A third
   scenario at 10.6 km is a sanity check only. A prank call, a rejected camera alarm and a
   confirmed-then-called alarm each behave as scripted. Traces: `notebooks/04_fusion.ipynb`.
2. **Triage.** The keyword rules identify the type in 0.94 of the phrasings their lexicon was
   written from. On held-out paraphrases they score 0.33 — but that is **entirely the `other`
   fallback class being right by default**, with cue recall exactly 0.0 in all three languages:
   the rules do not generalise to unseen wording at all. The offline LLM (Qwen3-VL-2B, 4-bit,
   2.48 s per call) does generalise (0.71–0.86 seen, 0.53–0.73 unseen) but reports cues that are
   absent. Rules stay the deployed default; both figures are a tool check, not a research claim
   (templates and lexicon share an author, L18).
3. **Scale of the synthetic weeks.** 1,202 incidents, 2,349 calls (385 of them false),
   **5,790,412 camera alarms** (27,573/day over 40 cameras), 742,291 candidates. The 30
   replications are 30 draws of the *same* calendar week, so they resample one hour-of-week and
   weekday profile.
4. **The detector masks itself** (`detector_masking` in the results file). At the measured 55.1
   false alarms per armed hour with a 60 s cooldown, a camera stays armed for only **65.3 s** on
   average before it fires, so it is **deaf 47.9 % of the time**. By the alternating-renewal
   argument (incident arrivals are independent of camera state, so PASTA applies), effective
   detection is 0.587 × 0.521 = **0.31**, against 0.587 when armed. Carrying the committed CIs of
   both measured inputs through gives **0.22–0.41**. Over 30 weeks only 10 incidents fell in any
   camera's view and **3** were detected (3.06 expected); n = 10, so that agreement is reassuring,
   not precise. Two caveats: the simulator gives a masked crash **no second chance** once the
   cooldown lapses, although a real crash scene persists (L22), and this assumption is what turns
   0.59 into 0.31. Independent support comes from the detector's own evaluation rather than from
   n = 10: counting masked crashes as misses, D-027 measured 81/281 = **0.288** on all ACCIDENT
   clips with lead time, close to the composed 0.31. The cooldown is a configured design
   parameter, so phase 10 sweeps it with the false-alarm rate.
5. **Camera coverage is thin.** 1.85 % of the incident location mass lies within 50 m road-graph
   of a camera junction, so a week holds well under one incident in view (9.57 expected over 30
   weeks, 10 observed — but that comparison uses the same ward weights that *placed* the
   incidents, so it checks the generator's sampling, not the coverage geometry).
6. **Calls carry the system.** Of 1,156 candidates holding genuine evidence, **1,153 were
   call-only**, 2 had a true camera event plus a call and 1 a true camera event alone. Median time
   from onset to posterior >= 0.5 on the call path is **201 s** — but a single call crosses 0.5 on
   its own, so this largely restates the assumed lognormal first-call delay (median 180 s), and
   the same property means **fusion cannot tell a lone genuine call from a lone prank**: 1.59 of
   the 1.83 false calls per day (87 %) cross the threshold, and the `prank_call` replay ends at
   posterior 0.47 with nothing ever confirming it. 949 of 1,202 incidents (79 %) reached 0.5.
7. **Quality of what fusion hands to dispatch.** Association recall is 0.82 and precision 0.20:
   3,284 false items were absorbed into candidates holding genuine evidence, and 396 real
   candidates were *opened* by a false alarm. Recall falls short of the gate's own 95 % design
   coverage because the gate is applied pairwise and is not transitive, and because later calls
   can fall outside the 900 s window. 1,024 candidates crossed 0.5 but they map to only 949
   distinct incidents, so **at least 75 incidents (7.9 % of those reaching the threshold)
   would be dispatched more than once** (the count is of surplus candidates, so a three-way
   split contributes two). Of the 1,156 real
   candidates the fused **type is right in 835 (72 %)** and the fused required services match
   exactly in 761 (66 %), with **274 (24 %) under-dispatched** and 121 (10 %) over-dispatched —
   driven by the rule triager's failure on unseen wording (item 2), since half the simulated calls
   use held-out phrasings. Requirements drive dispatch from phase 6 on, so this is the number that
   matters most downstream.
8. **Calibration is self-consistent and not uniformly good.** All seven bins, mean posterior
   against the share of candidates holding genuine evidence: `[0, 1e-4)` 0.000/0.00 (740,752
   candidates); `[1e-4, 0.01)` 0.0003/**0.92** (26); `[0.01, 0.1)` 0.045/0.40 (5); `[0.1, 0.5)`
   0.253/**0.72** (197); `[0.5, 0.9)` 0.688/0.59 (795); `[0.9, 0.99)` 0.986/1.00 (2);
   `[0.99, 1]` 1.000/1.00 (514). So the posterior is **markedly under-confident below 0.5** and
   slightly over-confident between 0.5 and 0.9. Three things to keep in mind: fusion's prior uses
   the same incident rate, hotspot weights and call parameters that generated the data, so this
   checks the algebra and not the world (L23); the posterior is evaluated at each candidate's last
   evidence, not at the end of the horizon; and "real" here means "holds genuine evidence", which
   is stricter than the hypothesis H1 (L19). A separate brute-force check against a simulation of
   the generative process finds the posterior exact to 0.01 for 0, 2 and 3 calls but optimistic by
   ~0.21 for a lone call arriving early in its window (L24).
9. **Two bugs found by the synthetic week, and six corrections after the audit.** The bugs, both
   fixed before any result was kept: a camera's recurring false alarms (~2 min apart) chained into
   one endless candidate, so the join window now runs from a candidate's FIRST evidence; and the
   association gate used road-graph distance with only the larger of two location errors, although
   a caller's error is a straight-line displacement — it now uses straight-line distance with the
   errors in quadrature, which raised correctly merged same-incident call pairs from 23/48 to
   47/48. The audit then corrected: the self-masking arithmetic was hand-typed rather than emitted
   (now in the results file); the fusion camera term used the unmasked 0.587, contradicting item 4
   (now uses 0.31); "a camera alarm is worth a likelihood ratio of ~2" confused the likelihood of
   a *confidence value* with that of the *alarm* — the alarmed LR is **1.01–1.04** across the
   three bins (emitted as `camera_alarm_evidence`); the "two separate crashes" scenario was 10.6 km apart and tested nothing; the
   calibration table was quoted selectively; and the gate's k = 3 was defended as a rule "not
   tuned on outcomes".
10. **Gate k = 3, honest status.** The criterion (smallest half-step k covering >= 95 % of
    same-incident pairs under the assumed error model) does select k = 3 — call-call 96.3 %,
    call-camera 96.3 %, against 88.3 %/90.5 % at k = 2 (`gate_k_synthetic.json`). But it was
    written *after* a synthetic week exposed the earlier gate, and it is evaluated on the same
    UNVERIFIED error model that generates those calls, so it is **in-sample by construction**. It
    accompanies a genuine metric fix and is not tuned on any outcome, but it is not independent
    evidence that the gate is right.
11. **For phase 6 onwards.** `FusionEngine` is the single implementation for live mode and the
    simulator. Policies should treat a camera alarm as very weak evidence (LR **1.01–1.04**, emitted as
    `camera_alarm_evidence`), expect
    the call path to dominate, and must not read a high posterior as "this incident is real *and*
    its location, type and severity are right" — H1 is only "an incident occurred somewhere in
    this region during this window", and a call-anchored region can have a 1,050 m radius (L19).
    Plan for roughly a quarter of incidents arriving under-specified (item 7) and ~7 % duplicated.
    The phase-10 break-even sweep (D-028) asks how much better the detector must be; it must also
    sweep the cooldown (item 4) and decouple fusion's assumed parameters from the generator's,
    because as wired the sweep cannot expose the circularity at all (L23).

## D-030 Phase 5: ward-hour demand forecasting  (2026-10-26; corrected 2026-10-27 after audit)
Plan: `docs/plans/phase-5.md`. Numbers from `results/forecasting/metrics_synthetic.json` and
`calibration_synthetic.csv`; figures in `notebooks/05_forecasting.ipynb`. **SYNTHETIC, and
forecast accuracy is not a research claim** (master plan §5): the training data comes from this
project's own generator, whose intensity is a formula we wrote. The claim is only how dispatch
performance depends on forecast quality, which phase 10 sweeps.

**Correction (2026-10-27).** The first version of this entry concluded that "the circularity did
not materialise". That was the wrong conclusion from the right data and is retracted in item 5;
the audit that found it also found two ranking claims stated beyond their uncertainty (item 4)
and five numbers quoted from deleted code (item 6). All three are fixed and every number below
now cites a committed artefact.
1. **History and panel.** `configs/scenarios/history_3y.yaml` is GENERATED by
   `resqcity.sim.make_history`: 156 weeks from 2023-01-02, 752 mm/year of seasonal rain
   (illustrative monthly profile, UNVERIFIED — IMD station normals could not be confirmed at the
   source) and a Ganeshotsav block each year (fixed-date placement, ILLUSTRATIVE: the real
   festival moves with the lunar calendar). Daily rainfall is drawn per day across the whole
   horizon rather than on a 365-day cycle, **deliberately**: on a cycle rain would be a
   deterministic function of the date and the informed/realistic comparison would collapse.
   `resqcity.sim.panel` builds 4,560,192 rows (58 wards × 26,208 hours × 3 target types) in ~14 s.
   The rare crowd class is folded into `other`; measured per-type shares of the generated
   incidents are accident 42.8 %, fire_smoke 54.4 %, other (incl. crowd) 2.8 %.
2. **The oracle column.** The panel carries the generator's expected count per cell, derived
   analytically (festival multiplier and type-mix override applied to route nodes only). It sums
   to 6,102.1 against 6,103 sampled incidents. That aggregate agreement is a **sanity check, not
   a proof**: the Poisson standard deviation on 6,103 is ±78, so it could not detect an error
   below about 1.3 %. The derivation is established cell by cell against
   `IncidentGenerator.node_rates` instead (`tests/sim/test_panel.py`).
3. **Two models side by side.** `realistic` sees calendar, ward identity, lagged counts and a
   NOISY rain forecast; `informed` additionally sees the true rain and the generator's own ward
   hotspot mass. `forecasting.config` refuses a realistic feature set containing either.
4. **Results** (test = the last 52 weeks, 1,520,064 ward-hour-type cells, 2,071 incidents;
   time-based split, never shuffled; 95 % intervals from a bootstrap over test **hours**, the
   unit of dependence, 2,000 resamples):

   | model | Poisson deviance [95 % CI] | top-5 ward hit rate [95 % CI] |
   |---|---|---|
   | uniform (no information) | 0.01799 [0.01729, 0.01863] | 0.083 [0.071, 0.095] |
   | climatology | 0.01740 [0.01666, 0.01812] | 0.226 [0.208, 0.246] |
   | climatology, per cell | 0.05842 [0.05569, 0.06104] | 0.141 [0.126, 0.156] |
   | lightgbm realistic | **0.01659** [0.01595, 0.01718] | 0.220 [0.202, 0.238] |
   | lightgbm informed | 0.01661 [0.01597, 0.01721] | 0.216 [0.199, 0.234] |
   | oracle (ground truth) | 0.01642 [0.01579, 0.01702] | **0.237** [0.219, 0.256] |

   **On top-k these models are statistically indistinguishable** — climatology, both boosters and
   even the oracle have overlapping intervals at this sample size, and the only separation that
   survives is every model against `uniform`. On Poisson deviance the ordering *is* real:
   LightGBM beats climatology, and both beat uniform. The earlier claim that pooling made
   climatology "the best ward-ranking model in the table" went beyond the evidence and is
   withdrawn. MAE is in the file but is useless here: 99.87 % of ward-hour-type cells are empty,
   so every model scores 0.0027 and predicting zero everywhere would score well.
5. **The circularity DID materialise; what did not is the informed model's advantage.**
   The `realistic` model — with no privileged inputs — closes **89.4 %** of the uniform→oracle
   deviance gap and 88.7 % of the top-5 gap (`*_share_of_gap_closed` in the results file). A
   model recovering nine tenths of the generator's own intensity surface is exactly the
   circularity this phase was built to expose. It is not a LightGBM achievement either: plain
   `climatology` closes **93.1 %** of the top-5 gap, more than either booster, so the ward
   ranking this generator produces is recoverable by an hour-of-week average. (Its deviance gap
   is only 37.6 %, which is the honest summary: the baseline ranks wards as well but does not
   predict counts as well.) The first version of this entry mistook the
   absence of an *informed-vs-realistic* gap for the absence of circularity; that comparison was
   never a test of it, for two reasons both now measured:
   - `hotspot_mass` is **constant within a ward** (one distinct value per ward, by construction),
     so given the 58-level `ward` categorical it is an exact relabelling. The informed model
     could not have won through it.
   - the rain channel is nearly inert: the generator's multiplier is 1 + 0.05 × mm/h, whose mean
     over the panel is **1.004** and whose maximum is 1.30.
   The honest statement is therefore "the informed advantage did not materialise, because its
   extra features are redundant or inert", and the two models are *indistinguishable* rather than
   one being better. `docs/limitations.md` L25 already said this; D-030 did not.
6. **Two of our own baselines were broken, and both repairs are now traceable.** The broken
   variants are kept as declared baselines so the claim is checkable rather than prose:
   - A per-cell historical average is degenerate at this sparsity: with 0.138 incidents per
     (ward, hour-of-week, type) cell it predicts exactly zero for **88.2 %** of cells, and
     **1,519 of 2,071** test incidents land on a zero, giving a deviance of 0.058 — three times
     worse than predicting the citywide mean (`per_cell_climatology_is_degenerate` in the
     results file). Pooling the hour-of-week shape across wards is the standard remedy and is
     what `climatology` now does.
   - The top-k metric was arbitrary for `uniform`: with every prediction equal, breaking ties by
     ward name made it silently "choose" the alphabetically first wards and score **0.171**.
     With random tie-breaking it scores 0.083, against the analytic 5/58 = 0.086
     (`tie_break_artefact` in the results file).
   Both baselines are now fitted on train **and** validation, matching the data the boosters see
   when choosing their stopping round.
7. **The quality band is narrow, so phase 10 needs the knob.** From no information (0.083) to
   perfect knowledge (0.237) is the whole top-5 range. The `NoisyForecaster` knob
   (mean-preserving lognormal, σ) gives a continuous lever instead: top-5 0.220 → 0.192
   (σ = 0.5) → 0.155 (σ = 1.0). Phase 10 should sweep σ rather than swapping models, and must
   not present the model differences as a meaningful quality range — item 4 shows most of them
   are not even distinguishable.
8. **For phase 6 onwards.** `predict(rows)` behind one `Forecaster` protocol; pre-positioning
   (C3) gets per-type demand. A saved LightGBM model reloads to **bit-identical** predictions
   (the phase's acceptance criterion, `tests/forecasting/test_models.py`). The oracle is a
   declared ablation in `sim.oracle_forecast`, never an achievable forecast;
   `tests/test_architecture.py` caught the evaluation driver importing `sim` from `forecasting`
   and it was split into `forecasting.metrics` (pure) and `sim.forecast_eval` (driver).

## D-031 Phase 6: the OR models, and two findings that reshape the experiments  (2026-10-27)
Plan: `docs/plans/phase-6.md`. Formulations: `docs/models/{dispatch,preposition}.md`. Code:
`src/resqcity/optimization/`. Results: `results/optimization/`. Every behavioural weight is
UNVERIFIED and listed by `find_unverified`; the fleet counts are press-derived (item 2).
1. **What was built.** C1 joint multi-agency dispatch (CP-SAT over integer seconds), C2
   confidence tiers, C3 pre-positioning (MEXCLP) with a fairness constraint, MCLP camera
   placement, Erlang-C fleet sizing, and the routing layer with the green corridor. Every model
   has a tiny-instance test checked against **brute force** (CLAUDE.md), and the solver's optimum
   is compared with exhaustive enumeration rather than with itself.
   - **Acceptance criterion met, on the model that actually runs.** A live batch of 10 incidents
     across the whole 91-unit fleet solves OPTIMAL in **0.170 s** median, **0.203 s** worst of 5,
     against the 1 s limit (`results/optimization/solver_bench.json`). The first version of this
     benchmark timed a model with **neither** the green corridor nor ward coverage and reported
     0.076 s; an OR review caught the acceptance criterion being read off that reduced model. The
     benchmark now solves every size twice on the same instances and reports both, so the cost of
     the two terms is visible: they roughly double the solve time. At 40 incidents the full model
     takes **1.15 s**, exceeds the limit, and returns FEASIBLE rather than OPTIMAL — the designed
     degradation, reported rather than hidden.
   - Two guards the brief did not ask for but the model needed: (C6), without which an incident
     left short of units would report a *faster* completion and shortfall would be rewarded; and
     the epsilon term of master plan §6.1, which stops the solver parking an ambulance off the
     critical path. A test asserts epsilon never changes any last-arrival time.
2. **Fleet sizes are press-derived, not cited.** No primary source gives Pune's fleet. Defaults:
   **39 police** (CR mobile units, Pune City Police MT department, press, May 2025), **22 fire**
   engines (PMC, press; note OSM has 14 stations against the press's 20-25), **30 ambulances**
   (DERIVED: 937 Dial-108 vehicles statewide scaled by PMC's population share gives ~29; a
   1-per-100,000 norm gives ~35). All UNVERIFIED and swept in phase 10.
3. **FINDING - the simulated city has no queueing, and that changes what phase 10 can claim.**
   At the cited incident rate (5.59/day, D-018) the 91-unit fleet runs at **0.26-0.32 %
   utilisation**; Erlang-C says **2 units per agency** would meet a 60 s wait target
   (`results/optimization/erlang.json`). The cause is a scope mismatch, not an error in either
   number: the fleet serves each agency's *entire* workload, while the generator models only the
   multi-agency subset in scope. Against the verified Pune figure of 23,887 Dial-112 calls in
   July 2026 (771/day), the modelled rate is **1/138 of the real police workload**; at the real
   volume those same 39 units would run at a plausible 41 % utilisation. Consequences, which are
   design decisions rather than observations:
   - RQ2 gains from joint dispatch will come from **coordination and travel, not from capacity**.
     D-018 anticipated this; it is now quantified.
   - **Phase 7 must add a background-workload process** that makes units busy with the
     out-of-scope calls, or the simulator will report unrealistically free fleets.
   - Phase 10 must include a scarce-fleet regime; sweeping the incident rate 0.5x-4x as planned
     leaves utilisation near 1 % and would not probe capacity at all.
4. **FINDING - cameras see very little of the city, however they are placed.** The 40 phase-2
   camera sites cover **3.95 %** of risk-weighted incident demand. Even 50 sites chosen optimally
   by MCLP cover only **13.27 %**, and 10 optimal sites (4.08 %) match all 40 existing ones -- the
   phase-2 siting rule was about four times less efficient than placement against demand.
   Multiplying by the measured effective detection of 0.31 (D-029), the ceiling on
   camera-detected incidents is roughly **4 %** at 50 optimally placed cameras. Coverage is an
   upper bound in another way too: every added camera also adds false alarms at ~29 per
   camera-hour.
5. **C2 inherits D-029's verdict.** A camera-only candidate has a posterior of order 1e-5, far
   below any sensible `tau_low`, so it lands in REVIEW. Confidence tiering therefore does **not**
   turn CCTV into a dispatch trigger; its only route to value is upgrading a candidate the
   moment a call or a human review corroborates it, which phase 10 measures. VERIFY and REVIEW
   incidents are deliberately kept out of the MILP so an unconfirmed candidate cannot take the
   unit a known incident needs.
6. **Price of fairness (RQ4) measured — and the fairness requirement is mostly unreachable.**
   C3 is solved twice, with and without the per-ward availability constraint
   (`results/optimization/preposition.json`, both solves proved OPTIMAL):

   | agency | units | bases | price of fairness | wards left short |
   |---|---|---|---|---|
   | ambulance | 30 | 590 | **10.19 %** | 39 of 58 |
   | police | 39 | 57 | **5.47 %** | 29 of 58 |
   | fire | 22 | 14 | **0.00 %** | 40 of 58 |

   The headline number is cheap — about a tenth of expected covered demand for ambulances — but
   the **"wards left short" column is the real finding and it limits what RQ4 can claim**: the
   requirement (one unit free with probability ≥ α = 0.8 within T_ward = 10 min) is simply
   unreachable for 29–40 of the 58 wards with these fleets, so the measured price is the cost of
   equalising only the wards that *can* be covered. Fire's 0.00 % is the degenerate case: with
   14 stations the constraint binds nowhere it can be satisfied, so it changes nothing. Phase 10
   must either sweep α and T_ward down to where the constraint is satisfiable, or report the
   price of fairness as conditional on the reachable subset. It must not be presented as "fairness
   costs 10 %".
   An impossible requirement is **not** detected and re-solved; (P5) is written as a soft
   constraint always, with a big-M derived from the objective's own coefficients (sum of the
   positive ones, plus 1 — here 991) so it is provably equivalent to a hard constraint wherever
   the hard one is satisfiable. The earlier hard-coded M = 10⁴ gave the same optima, but could
   not carry that guarantee; `docs/models/preposition.md` §5 has the proof and the formulation now
   matches the code.
7. **What the OR model review changed.** The review found three blockers, each reproduced before
   being fixed, each now a regression test: a unit dispatched twice when C1 and a later VERIFY
   tier both claimed it; MEXCLP cover truncated so expected coverage read 8.25 instead of 9.375;
   and a green corridor credited to a police unit dispatched to a *different* incident (510 s
   reported against 900 s actual). Four of my own tests failed afterwards — two encoded the old
   corridor semantics and two brute-forced an uncapped epsilon — which is the point of keeping the
   brute-force checks. The review's smaller findings are items 1 and 6 above plus the brute force
   itself: it previously ran only with the corridor off and no wards, so **C7 and C8 were checked
   by example rather than against the optimum**. The enumeration now covers both, including a case
   where they interact. One term it cannot check independently is the C6 shortfall surrogate,
   whose formula is a chosen definition rather than a derivable quantity; that is stated in the
   test module instead of being glossed over, and the property C6 exists to guarantee is asserted
   separately.

## D-032 Phase 7: the simulator, and the finding that reshapes the study  (2026-10-27, IN PROGRESS)
Plan: `docs/plans/phase-7.md`. Code: `src/resqcity/sim/{config,engine,units,policies,background_load,runner}.py`.
Slices 0-4 and 2+6 are built and committed; hospitals/transport (slice 5), the Erlang-C queue
validation (slice 7) and the phase close (slice 8) are **not yet done**, so this entry is marked
IN PROGRESS and no result file is published from it yet.

1. **FINDING - at the real camera deployment there is almost nothing for CCTV to trigger on.**
   Measured over 8 simulated weeks of the `normal` scenario (seed base 20261020, replications
   0-7), with every step reconciling against an earlier phase rather than against an assumption:

   | stage | count | share | reconciles with |
   |---|---|---|---|
   | incidents generated | 287 | ~36/week | base rate D-018 |
   | within camera range | 7 | **2.4 %** | D-031's 3.95 % risk-weighted coverage (expected 11.3; P(<=7) ~ 0.14) |
   | actually detected by a camera | 1 | **14 % of in-range** | D-029's 0.31 effective detection (expected 2.2; P(<=1) ~ 0.35) |
   | camera alarms raised | 1,543,628 | - | D-029's ~29 per camera-hour x 40 sites |
   | of which true | 1 | **0.00006 %** | one true alarm in 1.5 million |

   So **0.35 % of incidents are camera-detected**. B1-FULL can therefore differ from B0 only
   through the cost of chasing false alarms, not through earlier detection. This is not a new
   assumption: it is the arithmetic of D-029 and D-031 playing out in a simulator that was built
   to let it. Consequences, all design decisions rather than observations:
   - **Phase 10's sweep that matters is camera coverage**, ahead of forecast quality (D-030's band
     is narrow and its models are indistinguishable) and ahead of the confidence thresholds
     (D-031 item 5: no threshold promotes an alarm whose likelihood ratio is 1.01-1.04).
   - The study's honest headline may be **negative**: that CCTV-triggered joint dispatch cannot
     pay for itself at 40 cameras. That is a publishable answer to RQ1 and it must not be
     presented as a failure of the method.
   - The joint-dispatch question (RQ2) is separable and survives: B0 vs B2 is about *who is sent*
     and *whether agencies are asked together*, neither of which needs a camera.
2. **The status quo's hand-off delay largely determines RQ2, and it is assumed.** B0 asks agencies
   one after another; joint dispatch removes exactly that delay. No primary source for Pune's
   inter-agency hand-off time could be found, so `configs/sim.yaml` -> `policies.b0.
   sequential_handoff_s` is UNVERIFIED at 120 s under a heading saying so, and **phase 10 must
   sweep it including 0**, where B0 and B1 differ only in the evidence they admit. If the RQ2
   conclusion only holds at the larger values, the write-up has to say that.
3. **An untiered policy needs a declared credibility gate, and the arithmetic says why.** With no
   gate, B0/B1/B2 would each send a unit to every one of the ~193,000 alarms a week produces,
   saturating a 39-unit fleet within minutes and measuring nothing but detector precision.
   `policies.min_dispatch_posterior` = 0.5 (UNVERIFIED, swept) sits between the tiered thresholds
   (tau_low 0.2, tau_high 0.7) so B2 and B3 stay comparable. B3 and FULL do not use it: their
   tiers replace it, which is the point of that rung.
4. **The background workload is part of the model, not a refinement** (D-031 item 3, L30). Jobs
   arrive Poisson per agency and occupy a unit for an exponential time; the rate is DERIVED from
   the target utilisation (lambda = rho c / S), and for police that reproduces the verified 771
   Dial-112 calls a day. Realised utilisation is **0.41 police / 0.35 ambulance / 0.46 fire**,
   against the 0.26-0.32 % the capacity-blind model showed. Disabling it is a legitimate
   phase-10 ablation but must be explicit: the config rejects `enabled: false` with a non-zero
   target.
5. **Survival is a secondary outcome and its curves are illustrative.** No defensible per-minute
   survival-vs-time curve for road trauma could be confirmed at the source: the largest scoping
   review of the question reports "mixed (i.e., positive, negative, and neutral) associations,
   with conflicting conclusions" and deliberately does not quantify it (Bedard et al., *Int J
   Emerg Med* 2020;13:64, doi:10.1186/s12245-020-00324-7, read 2026-10-27). The familiar
   per-minute decline is from witnessed-VF cardiac arrest, a different patient and clock.
   `configs/survival.yaml` therefore marks every curve ILLUSTRATIVE, lists Valenzuela 1997 with
   `supports: NOTHING IN THIS FILE` so it cannot be mistaken for the source, and a validator
   refuses both a curve that claims a citation and any attempt to make survival the primary
   outcome. **Primary outcome: time to the last required arrival.**
6. **Four bugs worth recording, because each would have flattered a policy rather than crashed.**
   - Two decisions in one batch could claim the same unit, because the state change sat inside a
     SimPy process that had not started. Claims are now synchronous.
   - Every policy chose among all 91 units as though none were busy: policies get `Unit` records
     whose `state` is the roster's, not the live fleet's. So capacity could not bite at all, and
     the phase-10 headline would have run on a permanently free fleet - undoing item 4 silently.
     Unavailable units are masked where the ETA table is built. Dispatch batches fell from 1,544
     a day to 13, and every real incident now reaches a full response.
   - A short response was retried whether the engine *could not* provide a unit or the policy
     *chose* to withhold one. Retrying the latter fights a confidence tier and never terminates
     usefully; the engine now records refused asks and re-offers only those.
   - Background utilisation came out 14 % low (0.4297 against a 0.5 target) because a per-job RNG
     lookup returned a fresh generator and replayed one draw for ever: 0.4297/0.5 = 0.859 is that
     seed's first exponential draw.
7. **Runtime, measured.** 20-31 s per simulated day: 2.3-3.6 min per replication-week, and 5-9
   hours for 30 replications x 5 policies. CP-SAT contributes **0.22 s of a whole day**, so the
   solver is not the bottleneck - the camera alarm stream is, exactly as the plan predicted. The
   documented fallback stands: full 7-day/30-replication runs for the confirmatory comparisons,
   shorter horizons for the sensitivity grids.
