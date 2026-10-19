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
