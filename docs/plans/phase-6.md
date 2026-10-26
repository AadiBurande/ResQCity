Status: IN PROGRESS (started 2026-10-27). Approved by the user before any Phase 6 code; the
user also authorised continuing through the remaining phases unattended.

# Phase 6 plan: Operations Research models (the core contribution)

## Context
Phases 1–5 are closed and audited. Phase 6 is where the research question is actually answered:
it builds the dispatch models whose performance Phase 7 simulates and Phase 10 tests. The master
plan flags it as the highest-risk phase.

Two measured results from this week constrain the design, and both cut against the project's
original premise — the models must be built so the experiments can show this honestly:
- **A camera alarm is almost worthless on its own.** Measured likelihood ratio **1.01–1.04**
  (D-029). A confidence-aware policy cannot treat a CCTV alarm as a dispatch trigger; its value
  has to come from arriving *earlier* once a call or a human review corroborates it.
- **Forecast quality is a weak lever.** Top-5 ward hit rate spans only 0.083 (no information) to
  0.237 (perfect knowledge), with every fitted model at 0.216–0.227 (D-030). Pre-positioning has
  less headroom than assumed, so the "price of fairness" may dominate its benefit.

You are asleep, so every ambiguous choice below takes a documented default rather than a question
(CLAUDE.md working style), and each one is recorded in `docs/decisions.md` and swept in Phase 10.

### Fleet sizes — the one input I could not verify
I searched for real Pune fleet counts rather than inventing them. Only **press figures** exist;
none is a primary source, so all are UNVERIFIED and become Phase-10 sensitivity factors:

| Agency | Default | Basis |
|---|---|---|
| Police | **39** responder units | "39 CR mobile units … across 39 police stations", Pune City Police MT dept, press, May 2025. (123 beat-marshalling vehicles are patrol, not dispatch — they become the sensitivity upper bound.) |
| Fire | **22** engines + 5 tankers | PMC fire brigade, press. Note the conflict: OSM gives 14 stations, press says 20–25. |
| Ambulance | **30** | Derived: 937 Dial-108 ambulances statewide (press) × PMC's ~3.5 M of Maharashtra's ~112 M ≈ 29; a 1-per-100,000 norm gives ~35. Recorded as a derivation, not a citation. |

Units are placed at the OSM stations already built in Phase 2 (`data/processed/facilities.json`:
590 candidate ambulance bases = hospitals, 57 police stations, 14 fire stations), by a documented
rule in `configs/fleet.yaml`.

## Approach (slices; each ends with tests and a commit)

**0. Dependency and configs.** `uv add ortools`. New `configs/fleet.yaml` (roster + placement
rule) and `configs/optimization.yaml` (objective weights, solver limits, ρ, coverage targets,
fairness α, relocation budget). Typed in `src/resqcity/optimization/config.py` with
`SourcedValue`, so `find_unverified` lists every assumption.

**1. Routing and ETA matrices** (`optimization/routing.py`, `docs/models/routing.md`).
Reuses `RoadNetwork.shortest_times` (already hour-bucket cached) and `ScenarioState`. Adds:
precomputed station × node ETA matrices per hour bucket, saved to `data/processed/eta/`; a
green-corridor speed multiplier applied to the ambulance's route when police arrive first; and
flooded-edge handling, which already exists via `scenario.edge_factors`.

**2. Joint multi-agency dispatch, model C1** (`optimization/dispatch.py`,
`docs/models/dispatch.md`). CP-SAT (OR-Tools), integer seconds:
- `x[u,i] ∈ {0,1}`; requirement cover `Σ_{u∈U_a} x[u,i] + short[i,a] ≥ r[i,a]`;
- arrival `A[i,a] ≥ eta[u,i]·x[u,i]`; synchronisation `L[i] ≥ A[i,a]` for every required `a`;
- police-assist: pair variables `y[p,m,i] ≤ x[p,i], x[m,i]` with **precomputed** precedence
  (`eta_p ≤ eta_m`), so the indicator is data, not a variable; ambulance ETA scaled by ρ;
- coverage loss from the Phase-5 forecaster (`Forecaster.predict`), penalising wards left
  uncovered;
- objective `Σ_i w(severity_i)·L[i] + λ·coverage_loss + μ·shortfall`, 1 s limit, **greedy
  fallback** if unsolved;
- returns `DispatchDecision` + `Explanation` (both already in `common/schemas.py`).

**3. Confidence-aware policy, model C2** (`optimization/policy.py`). Tiers on the fused
posterior: `≥ τ_high` full dispatch; `[τ_low, τ_high)` nearest police verifies and others
pre-alert; `< τ_low` operator review with timeout `T_review`. Given D-029, the policy must also
accept a *review outcome* as an escalation trigger. `(τ_low, τ_high, T_review)` are exposed for
Phase-10 search.

**4. Pre-positioning with fairness, model C3** (`optimization/preposition.py`,
`docs/models/preposition.md`). MEXCLP over idle units on forecast demand, with a per-ward
coverage constraint (`T_ward` minutes with probability ≥ α) and a relocation budget. Solved with
and without the fairness constraint to report the **price of fairness**. Ties to the agreed
equity metric: Gini over the 58 wards of ward full-response accessibility, δ = 0.02 (D-015,
accepted in D-018).

**5. Camera placement, MCLP** (`optimization/placement.py`, `docs/models/placement.md`).
Choose K junctions maximising risk-weighted incident coverage; marginal-gain curve K = 0…50.
Honest framing: with effective detection 0.31 (D-029), coverage gain is an upper bound on value.

**6. Erlang-C fleet sizing** (`optimization/erlang.py`, `docs/models/erlang.md`). M/M/c per
agency and zone: utilisation, P(wait), expected wait — the analytical cross-check Phase 7
validates the simulator against.

**7. Review and close.** Every model gets a tiny-instance test **checked against brute force**
(CLAUDE.md, non-negotiable), the `or-model-builder` skill is used when writing each one, and the
`or-model-reviewer` subagent reviews formulation against code before the phase closes. Then
D-031, limitations, `make optimize`, and a `research-auditor` pass.

## After Phase 6 (your instruction to continue overnight)
I will keep going through Phases 7 → 11 without waiting for approval, writing
`docs/plans/phase-N.md` before each phase as CLAUDE.md requires, recording every default in
`docs/decisions.md`, and auditing each phase before closing it. I will **not** fabricate results
to finish faster: if a phase cannot be completed to the standard the earlier phases were held
to, I will stop at a clean commit and say exactly where I got to. The Phase 5 audit was cut off
by the session limit and will be resumed once it resets.

## Critical files
- **New:** `src/resqcity/optimization/{__init__,config,routing,fleet,dispatch,policy,preposition,placement,erlang,explain}.py`;
  `configs/{fleet,optimization}.yaml`; `docs/models/{routing,dispatch,preposition,placement,erlang}.md`;
  `tests/optimization/*`.
- **Changed:** `pyproject.toml` (ortools), `src/resqcity/tools/tasks.py` + `Makefile`
  (`make optimize`), `docs/decisions.md`, `docs/limitations.md`, `README.md`.
- **Reused:** `geo.network.RoadNetwork.{shortest_times,shortest_lengths,nearest_nodes}`,
  `geo.scenario.ScenarioState`, `geo.facilities.load_facilities`,
  `common.schemas.{Unit,Station,Assignment,DispatchDecision,Explanation,ServiceCounts}`,
  `common.requirements.required_services`, `forecasting.models.Forecaster`,
  `common.config.{SourcedValue,load_model,find_unverified}`, `common.provenance.write_meta`.

## Verification
- **Unit tests** (`uv run pytest`): every model against brute force on tiny instances — dispatch
  by enumerating all assignments on 2 incidents × 4 units; MEXCLP and MCLP against exhaustive
  subset search for small K; Erlang-C against a textbook table; routing ETAs against
  `shortest_times` directly; the greedy fallback never beats the solver on solvable instances;
  ρ = 0 reproduces the no-police-assist case; `find_unverified` flags every assumed weight.
- **Performance** (the phase's acceptance criterion): a realistic batch (10 incidents, 60 units)
  solves under **1 s**, measured and written to `results/optimization/solver_bench.json`; if it
  does not, the documented greedy fallback is used and the measurement is reported either way.
- **Review:** `or-model-reviewer` on formulation vs code; `research-auditor` on the phase's
  claims before it closes.
