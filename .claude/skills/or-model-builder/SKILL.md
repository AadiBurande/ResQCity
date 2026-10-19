---
name: or-model-builder
description: Build or change an Operations Research model (MILP, covering/location model, dispatch, routing, queueing) in optimization/. Use whenever writing or editing a mathematical optimization model, its objective, constraints or solver code.
---

# Building an OR model in ResQCity

Follow these steps in order for every model. Do not skip the formulation document.

## 1. Write the formulation first
Create or update `docs/models/<model_name>.md` with:
- Sets and indices, parameters (with units: seconds, metres, counts), decision variables (domain).
- Objective function and every constraint in LaTeX, each numbered and given a one-line meaning.
- Linearisation tricks used (big-M values and why they are valid upper bounds).
- Assumptions and what is deliberately left out.
- Complexity / expected instance size for live use and for experiments.

## 2. Implement to mirror the document
- One module per model in `optimization/`. Variable and constraint names in code match the
  equation numbers in the doc (e.g. a comment `# (C3) synchronization`).
- Parameters come from typed config, never literals. Big-M computed from data, not hard-coded.
- Solver: OR-Tools first; PuLP + CBC fallback. Always set a time limit and return solver status,
  objective, gap and solve time.
- Provide a greedy/heuristic fallback with the same interface for when the solver times out.
- Return a structured result plus an `Explanation` with the top contributing factors in plain English.

## 3. Test on tiny instances
- In `tests/optimization/test_<model>.py`, create instances small enough to brute-force
  (e.g. 2–3 incidents, 4–6 units) and assert the solver's optimum equals brute force.
- Test edge cases: no units available for an agency (shortfall), all incidents need fire,
  flooded node unreachable, ties in ETA.
- Property tests (hypothesis) for monotonicity where it should hold (more units never worsens
  the optimum; lower ETAs never increase L[i]).

## 4. Benchmark
- Script `src/resqcity/optimization/bench_<model>.py` reports solve time vs instance size; save to
  `results/bench/<model>.csv`. Live dispatch must answer within 1 s or use the fallback.

## 5. Review
Ask the `or-model-reviewer` subagent to compare `docs/models/<model>.md` with the code and tests.
Fix every mismatch it reports before committing.
