# Camera placement (MCLP) and Erlang-C fleet sizing

Status: phase 6.5 and 6.6 (D-031). Code: `src/resqcity/optimization/placement.py`. Parameters:
`configs/optimization.yaml` → `placement`, `erlang`. Tests:
`tests/optimization/test_placement_erlang.py`.

## Part A — Maximal Covering Location Problem

Choose $K$ junctions for cameras so that as much risk-weighted incident demand as possible falls
within the coverage radius of some camera.

| Symbol | Meaning |
|---|---|
| $i \in D$ | demand points (road nodes), weight $w_i$ = the generator's hotspot mass |
| $s \in S$ | candidate junctions |
| $S_i = \{s : d_G(s, i) \le r\}$ | sites covering $i$, $r$ = `placement.coverage_radius_m` |
| $x_s \in \{0,1\}$ | a camera is placed at $s$ |
| $y_i \in \{0,1\}$ | demand point $i$ is covered |

$$
\max \sum_{i \in D} w_i y_i
\quad\text{s.t.}\quad
y_i \le \sum_{s \in S_i} x_s \;\;\forall i, \qquad
\sum_{s \in S} x_s \le K .
$$

Solved exactly by CP-SAT. The **marginal-gain curve** for $K = 1 \dots K_{\max}$ is produced by
the greedy algorithm instead: maximum coverage is submodular, so greedy is within
$1 - 1/e \approx 63\%$ of the optimum, and a test checks greedy against exhaustive search on small
instances. `results/optimization/camera_placement.json` reports the **greedy** curve and says so.

**Coverage is an upper bound on value, twice over** (D-029, D-031):
1. A covered incident is detected only with the measured effective probability of **0.31**,
   because the detector's own false alarms silence it for 48 % of the time.
2. Every camera added also contributes false alarms at about **29 per camera-hour**, which the
   coverage objective does not charge for.

Measured: the 40 phase-2 sites cover 3.95 % of risk-weighted demand; 50 optimally chosen sites
cover 13.27 %. Multiplying by 0.31 caps camera-detected incidents near 4 %.

## Part B — Erlang-C (M/M/c)

An analytical cross-check for the simulator, not a result. Per agency: arrivals Poisson at
$\lambda$, service exponential with mean $S$ (`erlang.service_time_s`, UNVERIFIED), $c$ servers.
Offered load $a = \lambda S$ erlangs, utilisation $\rho = a/c$.

The blocking probability is computed with the numerically stable Erlang-B recurrence
$$
B(0) = 1, \qquad B(n) = \frac{a\,B(n-1)}{n + a\,B(n-1)},
$$
and then
$$
C(c, a) = \frac{B(c)}{1 - \rho\,(1 - B(c))}, \qquad
W_q = \frac{C(c,a)\,S}{c\,(1-\rho)} .
$$
For $\rho \ge 1$ the queue is unstable and $P(\text{wait}) = 1$. `servers_for_target` returns the
smallest $c$ whose $W_q$ meets `erlang.target_wait_s`, starting from $\lceil a \rceil$.

The recurrence is checked against the textbook factorial formula and against published values.

**What it showed** (D-031): at the cited incident rate the fleet runs at 0.26–0.32 % utilisation
and two units per agency would meet a 60 s target. The fleet serves each agency's whole workload
while the generator models only the in-scope subset, so the simulated city is capacity-
unconstrained — see L30.
