# C1 — Joint multi-agency dispatch

Status: phase 6 (D-031). Code: `src/resqcity/optimization/dispatch.py`. Parameters:
`configs/optimization.yaml` → `dispatch`, `routing.green_corridor`. Tests:
`tests/optimization/test_dispatch.py` (brute force on tiny instances).

One decision assigns ambulance, police and fire units to a batch of open incidents **together**,
so the quantity being minimised is the time until the *last* required agency arrives — the point
of the whole study. Dispatching each agency independently (policies B0/B1) is the baseline this
is measured against.

## 1. Sets and indices

| Symbol | Meaning |
|---|---|
| $i \in I$ | open incidents in this batch |
| $u \in U$ | units available to dispatch |
| $a \in A = \{\text{amb}, \text{pol}, \text{fire}\}$ | agencies |
| $U_a \subseteq U$ | units of agency $a$ |
| $P = U_{\text{pol}}$, $M = U_{\text{amb}}$ | police and ambulance units (for the corridor) |
| $w \in W$ | wards (for coverage loss) |

## 2. Parameters

| Symbol | Units | Meaning |
|---|---|---|
| $\eta_{u,i}$ | s | travel time of unit $u$ to incident $i$ (`routing.Router.eta_table`); $\infty$ if unreachable |
| $\tilde\eta_{u,i}$ | s | ETA **after** the green corridor, for $u \in M$ (§6); data, not a variable |
| $r_{i,a}$ | count | units of agency $a$ required by $i$ (`common.requirements.required_services`) |
| $\omega_i$ | — | severity weight of $i$ (`dispatch.severity_weights`) |
| $\lambda$ | s | coverage-loss price (`lambda_coverage`) |
| $\mu$ | s | shortfall price (`mu_shortfall`) |
| $\varepsilon$ | — | secondary ambulance-arrival weight (`epsilon_ambulance`) |
| $c_w$ | — | coverage value of ward $w$ (forecast demand, phase 5) |
| $\rho$ | — | green-corridor speed factor, $0 < \rho \le 1$ |
| $T$ | s | horizon, $T = \max_{u,i}\{\eta_{u,i} : \eta_{u,i} < \infty\}$ — the big-M |

$T$ is a valid upper bound because no arrival time in a feasible solution can exceed the slowest
finite ETA in the instance. It is computed from the data, never hard-coded.

## 3. Decision variables

| Symbol | Domain | Meaning |
|---|---|---|
| $x_{u,i}$ | $\{0,1\}$ | unit $u$ is sent to incident $i$ |
| $A_{i,a}$ | $[0, T]$ | time the last **required** unit of agency $a$ reaches $i$ |
| $L_i$ | $[0, T]$ | time the last required agency reaches $i$ |
| $s_{i,a}$ | $\mathbb{Z}_{\ge 0}$ | unmet requirement (shortfall) of agency $a$ at $i$ |
| $g_{u,i}$ | $\{0,1\}$ | ambulance $u$ is sent to $i$ **and** is eligible for the corridor |
| $z_w$ | $\{0,1\}$ | ward $w$ is left uncovered after this batch |

## 4. Objective

$$
\min \;\; \sum_{i \in I} \omega_i L_i
\;+\; \varepsilon \sum_{i \in I} A_{i,\text{amb}}
\;+\; \lambda \sum_{w \in W} c_w z_w
\;+\; \mu \sum_{i \in I} \sum_{a \in A} s_{i,a}
$$

The first term is the research quantity: severity-weighted time to full response. The second
exists because the first does not constrain an ambulance that is not on the critical path —
without it the solver may send a distant ambulance "because fire is late anyway", which costs
lives (master plan §6.1).

$\varepsilon$ must not outweigh one second of the lightest incident's last arrival. The config
check $\varepsilon < \min_i \omega_i$ is **not sufficient**, because two arrivals can differ by
the whole horizon; the condition is $\varepsilon\,T < \min_i \omega_i$. Since $T$ belongs to the
instance rather than the configuration, the effective value is capped at solve time
(`dispatch.effective_epsilon`, which also quantises it to the integer units the objective is
built in). A test asserts the capped $\varepsilon$ leaves every $L_i$ unchanged over a long
horizon; the OR review found a case where the unconstrained value did not.

## 5. Constraints

$$
\sum_{i \in I} x_{u,i} \le 1 \qquad \forall u \in U \tag{C1}
$$
A unit goes to at most one incident in this batch.

$$
\sum_{u \in U_a} x_{u,i} + s_{i,a} \;\ge\; r_{i,a} \qquad \forall i \in I,\; a \in A \tag{C2}
$$
Requirements are met, or the shortfall is paid for in the objective. Shortfall keeps the model
feasible when the city is out of units, which is exactly the situation the experiments study.

$$
x_{u,i} = 0 \qquad \text{whenever } \eta_{u,i} = \infty \tag{C3}
$$
Unreachable units are not assignable (flooded or disconnected).

$$
A_{i,a} \;\ge\; \eta_{u,i}\, x_{u,i} \qquad \forall u \in U_a,\; i \in I,\; r_{i,a} > 0 \tag{C4}
$$
Agency arrival is at least the slowest of its assigned units. For ambulances $\tilde\eta$ replaces
$\eta$. The product is linear because $\eta$ is a constant: it is simply
$A_{i,a} \ge \eta_{u,i}$ enforced when $x_{u,i} = 1$, written as an indicator.

$$
L_i \;\ge\; A_{i,a} \qquad \forall i \in I,\; a : r_{i,a} > 0 \tag{C5}
$$
Full response waits for the last required agency. Agencies that are not required do not hold it
up.

$$
L_i \;\ge\; T\,\frac{\sum_a s_{i,a}}{\max(1, \sum_a r_{i,a})} \tag{C6}
$$
An incident that is not fully served cannot report a short completion time. Without this, leaving
a requirement unmet would *lower* $L_i$ and the shortfall price would be competing against a
reward.

$$
g_{u,i} \le x_{u,i}, \qquad g_{u,i} \le \textstyle\sum_{p \in P} x_{p,i}, \qquad
g_{u,i} \ge x_{u,i} + \textstyle\sum_{p \in P} x_{p,i} - 1 \tag{C7}
$$
for $u \in M$ eligible at $i$ (§6). $g$ is the AND of "this ambulance is sent" and "some police
unit is sent to the same incident".

$$
z_w \;\ge\; 1 - \sum_{u \in U_w^{\text{idle}}} \Big(1 - \sum_{i} x_{u,i}\Big) \tag{C8}
$$
A ward counts as uncovered if this batch consumes every unit that could reach it within the
coverage radius. $U_w^{\text{idle}}$ is precomputed from the ETA table. A ward whose reachable
set is **empty** is fixed to $z_w = 1$: no assignment can cover it, so it is priced and reported
rather than silently dropped.

## 6. The green corridor as data

Police clearing a route only helps an ambulance that arrives *after* the police do. Making that
an optimisation decision needs pairwise variables over (police, ambulance, incident); instead the
precedence is settled before solving (`Router.green_corridor_etas`):

- $\pi_i = \min_{p \in P} \eta_{p,i}$ — the earliest any police unit could reach $i$;
- ambulance $u$ is **eligible** at $i$ when $\rho\,\eta_{u,i} \ge \pi_i$, i.e. some police unit
  could reach $i$ by the time the sped-up ambulance would;
- for an eligible pair, $\tilde\eta_{u,i} = \rho\,\eta_{u,i}$; otherwise
  $\tilde\eta_{u,i} = \eta_{u,i}$.

Eligibility is necessary but **not sufficient**: it asks only whether *some* police unit could
open the corridor. (C7) additionally requires that a police unit **actually sent to this
incident** is fast enough to realise it,
$$
g_{u,i} \;\le\; \sum_{p \in P\,:\,\eta_{p,i} \le \tilde\eta_{u,i}} x_{p,i},
$$
so a fast police unit dispatched to a *different* incident cannot lend its speed here. An earlier
version used "any police unit sent to $i$" together with a $\max(\cdot,\pi_i)$ floor; the OR
review found that it credited corridors across incidents, and that the reported ambulance ETA
could beat every police unit actually present.

$\tilde\eta$ is used in (C4) only when $g_{u,i} = 1$; otherwise the ambulance travels at $\eta$.
In code this is the pair of indicator constraints
$A_{i,\text{amb}} \ge \tilde\eta_{u,i}$ under $g_{u,i}$ and $A_{i,\text{amb}} \ge \eta_{u,i}$
under $x_{u,i} \wedge \neg g_{u,i}$.

Setting $\rho = 1$ disables the corridor exactly, which is the phase-10 ablation.

## 7. Solver, fallback and size

CP-SAT (OR-Tools) over **integer seconds**; all times are rounded to whole seconds, which is far
below the resolution of anything being modelled. Time limit `solver.time_limit_s` (1 s for live
use). If no feasible solution is returned in time, a greedy fallback runs: incidents in
decreasing $\omega_i$, each taking its nearest available units per agency. The fallback has the
same interface and is reported in the result's `status`, so a timeout is visible rather than
silent.

Live size is ~10 incidents × ~91 units: about 900 assignment variables plus the eligible
ambulance pairs. Benchmarked in `results/optimization/solver_bench.json`.

## 8. Assumptions and what is left out

1. **Batch, not sequence.** The model dispatches one batch; it does not plan future arrivals.
2. **Travel time is static within an hour bucket** (inherited from phase 2's routing).
3. **A unit serves one incident at a time**; no en-route reassignment within a batch.
4. **Requirements are deterministic** given the incident's estimated cues. The error in those
   cues is modelled upstream (fusion, phase 4) — measured at 24 % under-dispatch (D-029).
5. **Coverage loss is a one-step proxy**, not a queueing model; it uses forecast demand whose
   quality is itself narrow (D-030).
6. $\rho$ is a single factor for the whole route. Whether police can actually clear a route in
   Pune traffic is UNVERIFIED and is a phase-10 ablation.
