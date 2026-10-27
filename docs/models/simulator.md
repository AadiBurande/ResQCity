# The discrete-event simulator

Status: phase 7 (D-032). Code: `src/resqcity/sim/{engine,units,policies,background_load,hospitals,runner,validate_queue}.py`.
Parameters: `configs/sim.yaml`, `configs/survival.yaml`. Tests: `tests/sim/test_{engine,policies,background_load,hospitals,runner,queue_validation}.py`.

This is not an optimisation model, so it has no objective. What it has is a set of processes, an
ordering discipline and an accounting rule, and the claims of RQ1–RQ4 are only as good as those.
This page states them so a reviewer can disagree with a specific one.

## 1. State and time

Time is simulated seconds since the scenario start; the schemas carry timezone-aware datetimes
and the engine converts. One replication of one policy on one scenario is one `Engine`.

A unit moves through

```
IDLE → READY → EN_ROUTE → ON_SCENE → [TRANSPORTING → AT_HOSPITAL] → RETURNING → IDLE
IDLE → ON_OTHER_TASK → IDLE                     (out-of-scope workload, §4)
IDLE → RELOCATING → {IDLE, EN_ROUTE}            (pre-positioning, FULL only)
```

and **illegal transitions raise** rather than being tolerated: a teleporting unit would corrupt
the arrival and utilisation numbers every result rests on.

| rule | choice | why it is not neutral |
|---|---|---|
| Busy | every state except IDLE, READY and OUT_OF_SERVICE | READY is a pre-alert: the crew is in the vehicle, no fuel is burned |
| Claimable | IDLE and READY | a confirmed incident should be able to take a pre-alerted vehicle |
| Pre-emption | never | pessimistic: a real control room would recall a patrol for a serious crash (L38) |

## 2. Evidence is pre-generated, then replayed

A whole week of incidents, camera alarms and calls is drawn **before the clock starts**, by the
phase 3b–4 processes (`sim/generator.py`, `sim/detection.py`, `sim/calls.py`), and then replayed
in timestamp order into `fusion/engine.py`.

This is what makes common random numbers **exact** rather than approximate: B0 and FULL see the
identical week, so any difference between them is the policy. `Week.digest()` fingerprints the
evidence and a test asserts two runs at the same seed agree while a different replication differs.

## 3. What a policy may see

A policy receives fused `Incident` objects, `Unit` records and an ETA table. It never receives a
`GroundTruthIncident`; `tests/test_architecture.py` enforces the import boundary. In particular
the requirement it acts on is the **estimate** from camera cues (each flipped with the detector's
measured `cue_flip_probability`) or from call triage, so a CCTV-led policy can under- or
over-dispatch. Without that, every CCTV policy would be flattered.

Two details caused real bugs and are now invariants:

1. **Availability must be visible.** `Unit.state` is the roster's, not the live fleet's, so the
   runner masks unavailable units' ETAs to infinity before calling a policy. Without this every
   policy chose among all 91 units as though none were busy — capacity could not bite at all, and
   the background workload of §4 would have been invisible to the dispatcher.
2. **A claim takes effect immediately.** A SimPy process does not start until the environment
   next runs, so a unit is moved to EN_ROUTE synchronously, before the next assignment in the
   same batch is examined. Deferring it let two decisions claim the same unit.

## 4. The out-of-scope workload

Required, not a refinement (D-031 item 3, L30). Per agency, jobs arrive Poisson and occupy one
unit for an exponential time. The rate is **derived**, not configured:

$$
\lambda_a = \frac{\rho_a\, c_a}{S_a}
$$

for $c_a$ units, mean service $S_a$ and target utilisation $\rho_a$, giving an offered load of
exactly $\rho_a c_a$ erlangs. $S_a$ is read from `configs/optimization.yaml` →
`erlang.service_time_s` rather than copied, so this process and the phase-6 Erlang-C cross-check
cannot drift apart. Jobs **queue** when no unit is free, which is what makes §7's validation
meaningful.

## 5. Dispatch batching

C1 is a batch model, so dispatch is re-solved at most every `run.dispatch_interval_s`, and only
when there is something to decide:

- new evidence moved a candidate, **or**
- a **refused** ask can now be filled.

"Refused" is doing real work here. A response can be short because the engine could not provide a
unit, or because the policy *chose* to hold units back — which is exactly what a confidence tier
does. Retrying the second kind fights the policy and, since the answer never changes, rebuilds an
ETA table (a Dijkstra over the Pune graph) every interval for the rest of the week. Only refused
asks are re-offered, and only for what was refused.

A re-offer never *lowers* a requirement: an incident already found to need a fire engine still
needs one if a later, weaker estimate disagrees.

## 6. Outcomes

**Primary: time to the last required arrival.** Only units that count towards the requirement set
the clock, so a pre-alerted vehicle or a fourth police car at a one-car incident cannot; and an
incident still short of units reports **no** completion at all — the same rule as the dispatch
model's (C6) constraint.

**Secondary: expected survival**, and only ever secondary. No defensible per-minute
survival-vs-time curve for road trauma could be confirmed at the source (`configs/survival.yaml`
has the reasoning and the citation), so every curve is ILLUSTRATIVE, a validator refuses to let
survival become the primary outcome, and phase 10 must report it beside its own sensitivity sweep.

Incidents whose onset falls inside `run.warmup_hours` are excluded from the metrics but still
consume units: the fleet starts idle at every base, which flatters the first incidents.

## 7. What is validated, and what is not

`sim/validate_queue.py` strips the simulator to one agency with no incidents. That system **is**
M/M/c — Poisson arrivals, exponential service, $c$ identical servers, one FIFO queue, no
abandonment, no pre-emption — so `optimization.placement.queue_metrics` solves exactly the same
thing in closed form. Over 60 simulated days at three loads the worst absolute differences are

| quantity | worst difference |
|---|---|
| utilisation | 0.0006 |
| P(wait) | 0.0044 |
| E[W_q] | 1.05 s |

with the meaningful case at $\rho = 0.85$ (simulated 0.8494 / 0.2359 / 75.0 s against analytic
0.8500 / 0.2402 / 73.9 s over 95,631 jobs). A light load alone could have agreed by luck.

**This validates the clock, the queue discipline and the busy-time accounting.** It does **not**
validate travel, on-scene time, the second wave or the policies, none of which has a closed form.
The PASS must not be read as more than it is.

## 8. Assumptions, in the order they matter

1. **The status quo's hand-off delay is assumed** and is exactly what joint dispatch removes, so
   it largely sets the RQ2 effect size. UNVERIFIED at 120 s; phase 10 must sweep it including 0
   (L36).
2. **An untiered policy needs a credibility gate** or it would chase all ~193,000 weekly camera
   alarms. 0.5 is assumed and sets how much false-alarm load B1 and B2 carry (L37).
3. **Travel is the hour-bucket estimate** of phase 2 throughout, and a journey crossing a bucket
   boundary uses its departure bucket (L2).
4. **No en-route reassignment**: a dispatched unit continues to its incident
   (`docs/models/dispatch.md` §8).
5. **Hospital capabilities and bed counts are synthetic**; only the locations are real OSM data.
6. **Operators are always available** to review, which is the same capacity blindness as L30 but
   for people rather than vehicles.
