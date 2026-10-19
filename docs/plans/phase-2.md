# Phase 2 — Road network, facilities, wards and scenarios

Status: implemented · Date: 2026-10-20 · Decisions D-006…D-014 in docs/decisions.md.

## Findings that changed the brief
- `"Pune, Maharashtra, India"` geocodes to **Pune District (15,641 km²)**. OSM has no Pune
  Municipal Corporation boundary relation (only "Pune City Subdistrict", 312 km²).
- DataMeet publishes the **PMC 2022 electoral wards** (58 prabhags, union 506.9 km², traced
  from the PMC ward PDFs, CC BY-SA 2.5 IN). **Decision:** wards = these 58 prabhags. Study-area
  boundary = their union, buffered by 1 km for the road download, so roads and stations just
  outside the line still exist. H3 hexes remain as a configurable fallback.

## Modules
| File | Responsibility |
|---|---|
| `configs/geo.yaml` | ward source, download settings, road classes, base speeds, traffic periods, vehicle profiles, rivers, facility tags, synthetic hospital capabilities, camera-site selection |
| `configs/scenarios/{normal,monsoon,ganeshotsav}.yaml` | horizon, rain profile and effects, flooding stages, processions |
| `configs/incidents.yaml` | synthetic NHPP: base rate, hour/weekday profiles, hotspot weights, rain/festival multipliers, type mix, feature model |
| `configs/requirements.yaml` | severity-score weights + rule table → required services (element-wise max of matching rules) |
| `common/rng.py` | named random streams per (seed_base, replication, process family) for common random numbers |
| `common/requirements.py` | `severity_score(type, features)`, `required_services(type, features)`. Shared by vision (phase 3) and the generator |
| `geo/config.py` | typed models for `geo.yaml` |
| `geo/download.py` | **only network code**: wards, boundary, OSM graph, rivers, facilities → `data/raw/` + `download_meta.json` |
| `geo/network.py` | graph preparation (road class, speeds, largest SCC) and `RoadNetwork`, a compact array form cached as parquet, with fast routing (scipy Dijkstra) |
| `geo/traffic.py` | time-of-day speed multipliers per road class, vehicle profiles |
| `geo/scenario.py` | `ScenarioConfig`, `ScenarioState`: rain(t), flooded/closed/slowed edges at t, edge speed factors |
| `geo/wards.py` | load wards, assign nodes (within, else nearest), H3 fallback |
| `geo/facilities.py` | police/fire/hospital from OSM → `Station`/`Hospital`, snapped to nodes; synthetic capabilities |
| `geo/cameras.py` | camera junction selection (degree + approximate betweenness, arterial, min spacing) → `configs/cameras.yaml` |
| `geo/build.py` | `data/raw` → `data/processed` (graphml, parquet network, wards, facilities, cameras.yaml) |
| `geo/sanity.py` | travel times between known Pune places, off-peak and peak → `results/geo/sanity_travel_times.csv` |
| `sim/groundtruth.py`, `sim/generator.py` | `GroundTruthIncident` + NHPP generator; CLI writes `*_synthetic_*.parquet` |
| `notebooks/02_network.ipynb` | wards, facilities, cameras, flooded edges, processions |

## Key modelling choices
- **Travel time:** `t_e = length_e / (v_class × m_period(class, hour) × m_rain(rain) × m_flood_e × m_procession_e × m_profile)`.
  Impassable edges get `inf`. The base speeds are typical urban speeds per class, capped by
  `maxspeed` where tagged. Every speed and multiplier is a `SourcedValue` and UNVERIFIED for now.
- **Parallel edges** are collapsed to the fastest one (needed for the CSR routing matrix).
- **Flooding:** edge is flood-prone if within `buffer_m` of the Mula/Mutha/Mula-Mutha river lines (no
  SRTM; recorded). Each prone edge gets a fixed vulnerability `u ~ U(0,1)` (scenario seed). The
  flood stage depends on rain accumulated over a trailing window. An edge is impassable if
  `u < f_imp(stage)` and slowed if `u < f_imp + f_slow`. This is monotone: more rain never reopens a road.
- **Processions:** route = edges whose OSM `name` matches configured street names (Laxmi Road,
  Tilak Road, Kumthekar Road, Kelkar Road), plus optional polylines. In active windows they are closed and
  nearby edges are slowed. Incident rate near routes is multiplied.
- **Incident NHPP:** piecewise-constant per node-hour:
  `λ_n(t) = Λ/24 · s_n · h(hour) · d(weekday) · r(rain) · f_n(t) · scenario_mult`, where `s_n` is the
  normalised hotspot weight (road class and junction). Ward-hour rates are sums over nodes, so this is
  exactly the per-ward-hour NHPP of the brief at finer resolution.
- **Requirements:** rules only use "present" conditions with an element-wise max, so adding a
  severity cue can never reduce required services (property-tested).

## Acceptance
- Graph loads offline in < 10 s (graphml and the fast parquet cache, both timed).
- Sanity travel times for known pairs reported for the user to check.
- `notebooks/02_network.ipynb` executed: wards, facilities, cameras, flooded edges.
- Generator reproducible for a seed. Empirical ward-hour counts match `ward_hour_rates`.
- Tests use toy graphs only (no network).
