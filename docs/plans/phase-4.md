Status: IN PROGRESS (started 2026-10-24). Approved by the user before any Phase 4 code.

# Phase 4 plan: simulated 112 calls + Bayesian fusion

## Context
Phase 3b is closed (48c3fa8). The simulator now has measured detector inputs:
- a camera that is armed when a crash happens detects it with probability 0.59;
- median delay 2.9 s;
- 55 false alarms per armed hour, then a 60 s cooldown.

The Phase 4 brief (`docs/kit/BUILD_PROMPT.md`) asks for four things:
1. A call simulator.
2. A `CallTriager`: a rule-based version, plus an optional LLM version.
3. Fusion: group camera events and calls by graph distance and time, compute a Bayesian posterior, combine severity.
4. Acceptance: a scripted replay merges events correctly, and the posterior trace is plotted in `notebooks/04_fusion.ipynb`.

Two findings change the design (my arithmetic from committed numbers):
- **Coverage:** about 3–5 % of incidents fall within 50–100 m of one of the 40 cameras. Each camera therefore sees a real crash about once a year, against about 29 false alarms an hour. A camera alarm alone is real with probability of order 10⁻⁶, so CCTV helps only when a call confirms it, when a human reviews the clip, or with a better detector.
- **Call parameters:** I found no source for Indian crash-to-call delay, calls per incident, or prank-call share, so they are UNVERIFIED and will be swept.
  - Confirmed: Pune police received 23,887 Dial 112 calls in July 2026 (Free Press Journal, 2026-08-03).
  - Confirmed: Google's Android Emergency Location Service gives about 50 m accuracy, but in India only Uttar Pradesh is integrated (press reports, December 2025).

Your decisions (2026-10-24):
- **Full model:** confidence-dependent likelihood ratios measured from the final run, an optional "operator reviews the clip" evidence step, and a break-even analysis later in Phase 10.
- **Offline local LLM** for the optional triager. It is off by default, and the rule-based triager is always the baseline.

## Approach (vertical slices, each with tests; commits at slice ends)

**0. Plan and decision.** Copy this plan to `docs/plans/phase-4.md`. Record D-028 in `docs/decisions.md`: the CCTV role, the offline LLM, and the list of UNVERIFIED parameters. Commit.

**1. Measured likelihoods by confidence (exploratory, vision side).**
- Extend `src/resqcity/vision/eval/explore_v5.py`. `reconstruct_alarms` already scores every window, so also record each alarm's `combined_p` and the verifier's `p_yes` in `alarms_final.csv`.
- Build coarse confidence bins (2–3, since there are only 50 UA-DETRAC false alarms). Compare true detections (armed lead-time clips, n = 81; all detections as a sensitivity) with UA-DETRAC false alarms. Estimate likelihood ratio per bin with a bootstrap CI.
- Add a `confidence_model` block to `simulator_defaults` and re-export with `final_v5 --export-only`. No labels are re-read.

**2. Configs.** These files are new: `configs/calls.yaml`, `configs/fusion.yaml`, `src/resqcity/fusion/config.py` (pydantic, `SourcedValue`). Every value carries `source:` or UNVERIFIED. Proposed defaults, all UNVERIFIED unless a source is noted, and swept in Phase 10:

| Group | Parameter | Proposed default |
|---|---|---|
| Calls | Delay to the first call | lognormal, median 180 s, σ 0.8 |
| Calls | Share of incidents with no call | by severity class (minor > serious > critical) |
| Calls | Extra calls per reported incident | Poisson, mean by severity |
| Calls | Delay of each extra call | exponential gaps |
| Calls | Location error, default | phone/cell location, median 300 m |
| Calls | Location error, ELS scenario | 50 m (cited press source; ELS is not integrated in Maharashtra) |
| Calls | Language mix | en/hi/mr |
| Calls | Prank and mistaken calls | citywide Poisson rate; locations weighted like incidents |
| Calls | Misreported type | per-type probability |
| Fusion | Gate | graph distance ≤ base + k × location error; time window |
| Fusion | Camera coverage radius | 50 m |
| Fusion | Operator review | sensitivity, specificity, review time |
| Fusion | Incident update | emit when posterior changes > ε, or severity class or requirements change |
| Fusion | Candidate expiry | after a set time without new evidence |

**3. Call simulator.** New file `src/resqcity/fusion/call_sim.py`.
- Input: `GroundTruthIncident` from `src/resqcity/sim/groundtruth.py`, the network and the config.
- Output: `Call` objects (`synthetic=True`, `src/resqcity/common/schemas.py`) plus a separate truth link table (call → incident or prank). Fusion never sees the link table.
- RNG: `common/rng.stream` with per-incident sub-streams (`"calls"`, incident index), so the same random numbers are reused across settings. Changing the prank rate never shifts the genuine calls.
- Text: synthetic templates in English, Hindi and Marathi (a templates file in configs). Each type/severity cue (injured, fire, overturned, trapped, vehicles) has several phrasings. Some details are left out at random.
- A paraphrase set is kept out of the rule lexicon so the triage evaluation isn't fully circular.

**4. Statistical camera-event process.** New file `src/resqcity/sim/detection.py`, sim side, so `sim` doesn't import `vision`.
- **True detections:** an incident within the coverage radius of a camera is detected with probability 0.59. The delay is drawn from the measured delays, and the confidence from the true-detection confidence distribution measured in slice 1.
- **False alarms:** per camera, Poisson λ per armed hour with the 60 s cooldown, plus a share of persistently triggering cameras. Confidence comes from the false-alarm distribution.
- **Severity cues:** the true cues with flip probabilities (master plan §6.7). `required_services` comes from the existing rule table in `src/resqcity/common/requirements.py` (`severity_score`, `severity_class`, requirement rules).
- Reads `results/vision/sim_inputs.yaml` (`simulator_defaults`).

**5. Triage.** New file `src/resqcity/fusion/triage.py`.
- A `CallTriager` protocol that returns type, severity, confidence and cues.
- `RuleTriager`: keyword lexicons per language in config.
- `LocalLLMTriager`: offline, using a model already in your Hugging Face cache (Qwen3-VL-2B in text mode, or Qwen3.5-2B; revision pinned). It uses 4-bit quantisation like `vision/verifier.py`, deterministic decoding and a JSON answer. It is lazily imported and off by default; tests and experiments never need it.
- New file `src/resqcity/fusion/eval_triage.py` writes `results/fusion/triage_eval_synthetic.json`: per-language accuracy on held-out paraphrases for both triagers. This checks the tool and is not a research claim.

**6. Schemas.** In `src/resqcity/common/schemas.py`:
- `SourceRef.kind` gains `operator_review`.
- A new `Review` schema: id, ts, event/incident ref, outcome confirmed/rejected, `synthetic`.
- `Incident` gains an optional `evidence_log_odds: dict[str, float]`, for explanations in Phase 6.

Regenerate the dashboard TypeScript types (`tools/gen_ts_types.py`; the pre-commit hook checks them).

**7. Fusion engine.**
- `RoadNetwork.shortest_lengths` in `src/resqcity/geo/network.py`: road-graph distance in metres from a cached length matrix, next to `shortest_times`.
- New file `src/resqcity/fusion/engine.py`: `FusionEngine.on_event / on_call / on_review / tick(now)` returns `Incident` updates.
- **Association:** gated by graph distance and time; the call's location is snapped with `nearest_nodes`. Several events from one camera within the window count once (they are not independent).
- **Posterior in log-odds:**
  - first-evidence odds = rate of true evidence of that kind ÷ rate of false evidence of that kind (camera, by confidence bin; call);
  - each later piece of evidence multiplies by P(it matches | real) ÷ P(it matches by coincidence | false candidate). The coincidence term comes from the background rates of calls and alarms in the gate area and window. That is why a call confirming a camera alarm is strong evidence.
  - operator review contributes its sensitivity/specificity ratio.
  - Conditional independence given the true state is an explicit, documented assumption.
- **Severity:** a reliability-weighted combination of the camera and call severity estimates. Requirements are re-derived through the rule table and can only grow.
- The formulation goes in `docs/models/fusion.md` (LaTeX).

**8. Replay and acceptance.**
- `src/resqcity/fusion/replay.py` runs scripted scenarios in `configs/fusion_scenarios/`:
  - two cameras + one call on the same crash → one incident;
  - two nearby but separate crashes → two incidents;
  - a prank call;
  - a camera false alarm that is rejected on review.
- It writes `results/fusion/replay_<name>_synthetic.json`, with provenance.
- `notebooks/04_fusion.ipynb` plots the posterior traces from those files.

**9. Synthetic-week model check.**
- `src/resqcity/fusion/check_week.py` runs: generator → calls + camera events → fusion.
- It reports merge precision/recall against the truth, posterior calibration, and time from onset to posterior ≥ 0.5 by evidence path (call only, camera then call, camera + review).
- It also counts false candidates per hour.
- Output: `results/fusion/week_check_synthetic.json`. These are model checks, labelled synthetic, not research results.

**10. Wrap-up.**
- Makefile / `tools/tasks.py` target `fusion`.
- Update `docs/limitations.md` (call parameters UNVERIFIED, circular templates, the independence assumption) and `docs/data-sources.md` (the two press sources).
- `/verify`, then a `research-auditor` review, then commit.

## Critical files
- **New:** `src/resqcity/fusion/{__init__,config,call_sim,triage,eval_triage,engine,replay,check_week}.py`, `src/resqcity/sim/detection.py`, `configs/{calls,fusion}.yaml`, `configs/fusion_scenarios/*.yaml`, `docs/models/fusion.md`, `notebooks/04_fusion.ipynb`, `tests/fusion/*`, `tests/sim/test_detection.py`.
- **Changed:** `src/resqcity/common/schemas.py`, `src/resqcity/geo/network.py`, `src/resqcity/vision/eval/explore_v5.py`, `src/resqcity/vision/eval/final_v5.py` (export), `dashboard/src/types/schemas.ts` (generated), `docs/decisions.md`, `docs/limitations.md`, `docs/data-sources.md`, `tools/tasks.py`, `Makefile`.
- **Reused:** `common/rng.stream`, `common/config.{SourcedValue,load_model,find_unverified}`, `common/provenance.write_meta`, `common/requirements`, `geo/network.RoadNetwork.{nearest_nodes,shortest_times}`, `geo/cameras.load_cameras`, `sim/generator.IncidentGenerator`, the 4-bit loading code in `vision/verifier.py`.

## Verification
- **Unit tests** (`uv run pytest`):
  - association: two cameras + one call make one incident; two separate crashes make two;
  - posterior monotonicity: confirming evidence never lowers it;
  - the posterior equals brute-force Bayes on a tiny enumerated model;
  - call simulator: reproducible and common-random-number safe (changing the prank rate leaves genuine calls unchanged);
  - detection-process rates match the config within tolerance;
  - rule triage in en/hi/mr;
  - `shortest_lengths` on a toy graph;
  - the LLM triager test is skipped unless the model is cached and a flag is set.
- **Lint:** `uv run ruff check`.
- **Runs:**
  - `uv run python -m resqcity.fusion.replay --all`: merges match the scripted truth, and the notebook renders the traces;
  - `uv run python -m resqcity.fusion.check_week`: the JSON is written with provenance (`dirty: false` after commit);
  - the slice-1 re-export leaves `results/vision/holdout_access.log` at one line.
- **Review:** a `research-auditor` review of the Phase 4 results and claims before the phase is marked done.

## Deviations during implementation
- 2026-10-24: the call simulator is `src/resqcity/sim/calls.py` (not `fusion/call_sim.py`) and the
  synthetic-week check is `src/resqcity/sim/fusion_check.py` (not `fusion/check_week.py`): both
  read ground truth, and `tests/test_architecture.py` allows that only inside `sim/`.
- 2026-10-24: the scripted replay is `src/resqcity/sim/fusion_replay.py` (it needs the incident
  background built from `sim.generator`), with `sim/background.py`; `geo.cameras.camera_coverage`
  is shared by the simulator and fusion.
- 2026-10-24: the call term was first count-based ("how many calls by now"); the replays showed
  that a camera-pinned onset then LOWERED the posterior of a typically timed call. Replaced by
  the point-process likelihood (docs/models/fusion.md §3.2) before any results were kept.
