# Phase 7 plan — SimPy discrete-event simulator (the digital twin)

Status: planned 2026-10-27. Phases 1–6 are closed and audited. This phase is where the research
question becomes measurable: every policy comparison in phase 10 runs on this simulator.

## What phases 3b–6 force this design to do

Four measured results constrain Phase 7. They are not caveats to add later — each one changes a
process the simulator has to contain.

| Measurement | Where | What Phase 7 must do about it |
|---|---|---|
| A camera alarm multiplies the odds by **1.01–1.04** | D-029 | CCTV cannot be a dispatch trigger. The simulator must be able to show CCTV arriving *earlier* once corroborated, and must show that honestly if the answer is "no gain". |
| Effective detection is **0.31**, not 0.59 | D-029 | Use `sim/detection.py` as it is (self-masking included). Do not "fix" the conservative masking; sweep it in phase 10. |
| Cameras cover **3.95 %** of risk-weighted demand | D-031 | Most incidents will never be seen by a camera. The B1–FULL policies must degrade gracefully to call-only, and the per-incident log must record which incidents a camera could even have seen. |
| The modelled city runs at **0.3 % utilisation** | D-031, L30 | **A background-workload process is required**, not optional. Without it every unit is always free, travel time is the only cost, and joint dispatch cannot be distinguished from sequential dispatch for the right reason. |

## Design decisions taken here (the user is asleep; each is recorded in `docs/decisions.md` as D-032 and swept in phase 10)

1. **Primary outcome is time, not survival.** Master plan §6 item 8 and the research question both
   say time-to-last-required-arrival. My own searches today could not find a defensible per-minute
   survival-vs-time curve for road trauma: the largest scoping review of the question (Bedard et
   al., *Int J Emerg Med* 2020;13:64, doi:10.1186/s12245-020-00324-7) reports "mixed (i.e.,
   positive, negative, and neutral) associations, with conflicting conclusions" and explicitly
   does not quantify the relationship. The out-of-hospital-cardiac-arrest literature does give a
   per-minute decline, but for witnessed VF arrest, which is **not** the modelled incident type.
   So `configs/survival.yaml` exists, is used, and is labelled: every curve carries
   `source: UNVERIFIED` or `source: ILLUSTRATIVE` (which `find_unverified` now reports, D-030),
   expected survival is reported **only** as a secondary outcome beside its sensitivity, and no
   conclusion in the paper may rest on it. I will not type a coefficient I could not confirm into
   a config with a citation next to it.
2. **Background workload is a two-parameter process, calibrated to a verified number.** Each
   agency draws out-of-scope jobs as a Poisson process with an exponential service time, sized so
   that utilisation matches a target. The anchor is the one verified volume figure the project
   has: 23,887 Dial-112 calls in Pune in July 2026 (771/day), against the modelled in-scope rate
   of 5.59/day (D-018, D-031). Default target utilisation is therefore **set from that ratio**,
   not chosen: `background.target_utilisation` per agency, with the derivation in the config's
   `note`. Phase 10 sweeps it from 0 (today's capacity-blind model, kept as an ablation) upwards.
3. **Requirement-estimation error is simulated** (master plan §6 item 7). Policies never see the
   true `ServiceCounts`. They see an estimate drawn from a confusion model over the severity
   class, so a CCTV-led policy can under- or over-dispatch. Under-dispatch triggers a **second
   wave** after on-scene assessment; over-dispatch costs unit time. Without this, every
   CCTV policy would be flattered.
4. **False alarms cost real units.** A false alarm that reaches VERIFY sends a police unit, which
   travels, finds nothing, and returns. This is the main cost of a low-precision detector and it
   must appear in the metrics, not in prose.
5. **One week, 30 replications, common random numbers.** The horizon is 7 days with a warm-up
   that is excluded from the metrics (the fleet starts idle, which flatters early incidents).
   Streams are per process family, keyed by replication, so B0 and FULL see the *same* incidents,
   the same detector outcomes and the same calls. Seed bases stay disjoint from the tuning seeds
   (D-030 pattern).
6. **No en-route reassignment.** A dispatched unit continues to its incident. Stated as a
   limitation rather than modelled, to match `docs/models/dispatch.md` §8.

## Slices (each ends with tests and a commit)

**0. Dependency and configs.** `uv add simpy`. New `configs/sim.yaml` (horizon, warm-up,
replications, on-scene and handover times, transport decision, background workload, requirement
confusion) and `configs/survival.yaml` (per-type curves, all UNVERIFIED/ILLUSTRATIVE). Typed in
`src/resqcity/sim/config.py`, with a `find_unverified` test pinning the exact list (the pattern
`tests/forecasting/test_config.py` established).

**1. The engine and the unit state machine** (`sim/engine.py`, `sim/units.py`). SimPy
`Environment` over simulated seconds, with a wall-clock mapping for the live demo later. A unit
moves through IDLE → DISPATCHED → ON_SCENE → (TRANSPORTING → AT_HOSPITAL) → RETURNING → IDLE,
driven by `RoadNetwork.shortest_times` for travel. Every transition appends to the per-incident
and per-unit logs. Tests: a single incident end to end; a unit cannot be in two places; the log's
timestamps are monotone.

**2. Processes wired in** (`sim/processes.py`). Reuse, do not rewrite: `IncidentGenerator`
(ground truth), `DetectionProcess` (measured detector), `CallSimulator` (112 calls),
`FusionEngine` (posterior + requirement estimate). The simulator owns the clock; each process is
asked for the events in the next interval. `tests/test_architecture.py` already forbids
`fusion`/`forecasting`/`optimization` importing `sim`, so the wiring lives here.

**3. Policies behind one protocol** (`sim/policies/`). `decide(incidents, units, etas, now)` for
each of:
- **B0** calls only, independent nearest unit per agency, sequential hand-off (status quo)
- **B1** CCTV + calls, independent nearest unit
- **B2** B1 + joint MILP (C1)
- **B3** B2 + confidence tiers (C2)
- **FULL** B3 + fairness pre-positioning (C3) + green corridor
Tests: B0 never uses an `Event`; B2's assignment equals `dispatch.solve` on the same inputs;
FULL with ρ = 1 and fairness off reproduces B3.

**4. Background workload** (`sim/background_load.py`). Poisson arrivals per agency occupying a
unit for an exponential service time, with the measured target utilisation. Tests: the realised
utilisation matches the target within sampling error over a long run, and with the target at 0 the
simulator reproduces slice 1 exactly (so the ablation is clean).

**5. Hospitals** (`sim/hospitals.py`). Transport decision by capability and distance; handover
occupies the ambulance; hospital capacity is a `simpy.Resource`. Transported patients that find no
capable hospital are logged, not dropped.

**6. Outcomes and logs** (`sim/outcomes.py`, `sim/runner.py`). Per-incident parquet with onset,
first detection, first call, fused posterior, decision, each agency arrival, hospital arrival and
handover; per-unit utilisation log. Primary metric time-to-last-required-arrival; secondaries
including the survival transform, with its sensitivity reported beside it. Provenance sidecars.

**7. Validation against Erlang-C** (`sim/validate_queue.py`, notebook). Strip the simulator to
M/M/c (one agency, exponential service, no travel) and compare utilisation, P(wait) and E[W_q]
with `optimization/erlang.py`. Document the comparison and the residual gap. This is the one check
that the engine's queueing is right rather than plausible.

**8. Close.** `make sim SCENARIO=normal POLICY=B0|FULL`, notebook `07_simulator.ipynb`, D-032,
limitations, then `/verify` and a `research-auditor` pass before the phase is marked done.

## Acceptance criteria

- `make sim SCENARIO=normal POLICY=B0` and `POLICY=FULL` each run a 7-day week in reasonable time
  on this laptop, and the figure is reported either way.
- Logs pass schema checks; every result file has a `dirty: false` sidecar.
- The Erlang-C comparison is saved and discussed, including where it disagrees.
- With the background workload at its measured target, fleet utilisation is no longer ~0.3 %; the
  realised figure is reported, not assumed.
- Common random numbers verified: two policies on the same seed see identical incidents, events
  and calls (a test, not a claim).

## Critical files

- **New:** `src/resqcity/sim/{config,engine,units,processes,background_load,hospitals,outcomes,runner,validate_queue}.py`,
  `src/resqcity/sim/policies/{__init__,base,b0,b1,b2,b3,full}.py`, `configs/{sim,survival}.yaml`,
  `tests/sim/test_{engine,units,policies,background_load,hospitals,outcomes,queue_validation}.py`,
  `notebooks/07_simulator.ipynb`, `docs/models/simulator.md`.
- **Changed:** `pyproject.toml` (simpy), `src/resqcity/tools/tasks.py` + `Makefile` (`make sim`),
  `docs/decisions.md`, `docs/limitations.md`, `README.md`.
- **Reused unchanged:** `sim/{generator,detection,calls,background,groundtruth}.py`,
  `fusion/engine.py`, `optimization/{dispatch,policy,preposition,routing}.py`,
  `geo/{network,scenario,facilities}.py`, `common/{schemas,requirements,rng,provenance,config}.py`.

## Risks

1. **Runtime.** A 7-day week with CP-SAT on every batch could be slow. The measured live-batch
   solve is 0.170 s (D-031), and a week has a few hundred batches, so C1 is not the bottleneck —
   the camera false-alarm stream (~29 per camera-hour × 40 cameras) is. `fusion_check.py` already
   handles a week at this rate, so the approach is known to work; if it is too slow, the documented
   fallback is a shorter horizon for the sensitivity grids only (master plan §6 item 6).
2. **The simulator could flatter CCTV through a modelling choice rather than a finding.** The
   guards are slice 3's requirement-estimation error, slice 4's background load and slice 7's
   queue validation. If FULL still beats B0 only through an assumption, that must be reported as
   such.
3. **Survival could creep into the headline.** Mitigation: the runner reports the primary time
   metric and the survival metric in separate blocks, and `docs/limitations.md` carries the
   contested-evidence note with the Bedard et al. citation.
