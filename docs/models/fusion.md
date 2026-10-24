# Fusion model: camera events, 112 calls and operator reviews → incident posterior

Status: phase 4 (D-028). Code: `src/resqcity/fusion/model.py` (terms and posterior),
`src/resqcity/fusion/engine.py` (association and updates). Parameters: `configs/fusion.yaml`,
`configs/calls.yaml`, `results/vision/sim_inputs.yaml` (`simulator_defaults`). Most parameters
are UNVERIFIED (listed by `common.config.find_unverified`) and are phase-10 sensitivity factors.

## 1. Candidates and association

A **candidate** $j$ collects evidence that may describe one incident. It has an anchor node
$a_j$ (the most precise location seen: a camera junction beats a call), a **region** $R_j$ (set
of road-graph nodes) and the time of its latest evidence.

New evidence $e$ at node $x_e$ with stated location error $\varepsilon_e$ (call; $0$ for a
camera) joins the candidate with the highest posterior among those with

$$
d_G(x_e, a_j) \le D_0 + k\,\max(\varepsilon_e, \varepsilon_{a_j}), \qquad
t_e - t_j^{\text{latest}} \le W ,
$$

where $d_G$ is the road-graph distance (direction ignored), $D_0$ = `gate.base_distance_m`,
$k$ = `gate.k_location_error`, $W$ = `gate.time_window_s`. Otherwise $e$ starts a new candidate.
The region grows as the union of evidence regions: a camera contributes its coverage
$C_c = \{v : d_G(v, \text{junction}_c) \le r_{\text{cov}}\}$, a call the ball of radius
$D_0 + k\varepsilon_e$ around its snapped node. Several alarms of one camera count once (only
its first alarm in the window is used: repeated alarms are not independent).

## 2. Hypotheses

$H_1$: at least one real incident occurred in $R_j$ during a window of length $W$.
$H_0$: none did; all evidence of $j$ is background (false alarms, false calls).

Let $\pi(v)$ be the incident location distribution over nodes (`node_share`, the generator's
hotspot weights), $m(R) = \sum_{v\in R}\pi(v)$, $\Lambda$ the citywide incident rate (s⁻¹),
$\Lambda_F$ the false-call rate (s⁻¹).

## 3. Posterior (log-odds, recomputed from all evidence)

$$
\operatorname{logit} P(H_1 \mid \text{evidence}) =
\underbrace{\log\!\left(e^{\Lambda\, m(R)\, W} - 1\right)}_{\text{prior (Poisson)}}
+ \sum_{c} \log \mathrm{LR}_c
+ \log \mathrm{LR}_{\text{calls}}
+ \sum_{r} \log \mathrm{LR}_r .
$$

Because every term is a function of the whole evidence set (not of the order of arrival), the
posterior does not depend on arrival order (tested).

### 3.1 Cameras

For each camera $c$ whose coverage intersects $R$, let
$q_c = \frac{m(C_c\cap R)}{m(R)}\,\sigma_{\text{det}}$ (share of the region it sees, times the share
of incident types the detector detects, $\sigma_{\text{det}}$), $p$ = detection probability
when armed, $\mu_c$ the long-run false-alarm rate $\lambda/(1+\lambda\,\text{cooldown})$ (s⁻¹),
and $s_1(b), s_0(b)$ the shares of real and false alarms in confidence bin $b$ (measured,
exploratory; D-028).

- Camera alarmed (first alarm in bin $b$): the alarm time is a density event; under $H_1$ the
  onset is uniform in the window, so
  $$\mathrm{LR}_c = \frac{q_c\, p\, s_1(b)/W + \mu_c s_0(b)}{\mu_c s_0(b)} = 1 + \frac{q_c\, p\, s_1(b)}{W\, \mu_c\, s_0(b)} \;\ge 1 .$$
  For small values the window cancels in $\text{prior}\times(\mathrm{LR}_c-1)$, as it should.
- Camera silent (no alarm, counted only after the detector delay has passed):
  $\mathrm{LR}_c = 1 - q_c\, p \le 1$ (masking by a running cooldown is ignored here).

### 3.2 Calls

Let $n$ be the number of calls in the candidate. Under $H_0$ they are background:
$n \sim \text{Poisson}(\beta)$, $\beta = \Lambda_F\, m(R)\, W$. Under $H_1$, $n = G + B$ with $B$ the
same background and $G$ the genuine calls observed by now:

$$
P(G = 0) = p_u + (1-p_u)\,(1 - F_1(\tau)), \qquad
P(G = 1 + k) = (1-p_u)\,F_1(\tau)\, P(\min(N, A) = k),
$$

with $p_u$ = `p_unreported` of the candidate's severity class, $F_1$ the lognormal CDF of the
first-call delay, $\tau$ the time since onset (camera-pinned: alarm time − detector delay;
otherwise the onset is averaged uniformly over the window before the first evidence),
$N \sim \text{Poisson}(m)$ the extra calls in total and $A \sim \text{Poisson}(\tau_1/g)$ the arrivals
of the chained exponential gaps (mean $g$) since the first call ($\tau_1$). Then
$\mathrm{LR}_{\text{calls}} = P(n\mid H_1)/P(n\mid H_0)$. It increases with $n$ (the ratio
$\sum_k P(G{=}k)\,\frac{n!}{(n-k)!\,\beta^k}$ is increasing in $n$), so a confirming call never
lowers the posterior (tested), and $P(G{=}0)$ falls with time, so a candidate without calls loses
belief as time passes. Checked against a brute-force simulation of `sim.calls`
(`tests/fusion/test_model.py`).

### 3.3 Operator review

A review of an alarm clip with sensitivity $s_e$ and specificity $s_p$:
$\mathrm{LR} = s_e/(1-s_p)$ if confirmed, $(1-s_e)/s_p$ if rejected.

## 4. Severity, type and requirements

Severity = reliability-weighted mean of the camera estimates and the call-triage estimates
(`severity.camera_weight`, `severity.call_weight`). Requirements = element-wise maximum of the
requirements implied by every source (`configs/requirements.yaml`), so adding evidence can only
add units. Type = confidence-weighted vote of the triaged calls, else the camera event type.

## 5. Assumptions and their consequences

1. **Conditional independence** of cameras, calls and reviews given $H$. Callers who saw the
   same crash are counted through the call model, not as independent evidence.
2. **Count-based calls.** Call timing enters only through "how many by now"; the exact gaps are
   not used (except to place the first call).
3. **At most one incident per region and window**; a candidate is never split or merged later.
4. **Background calls are false calls only**; genuine calls about incidents outside the region
   that land inside it through location error are ignored ($H_0$ slightly too sharp).
5. **Average camera.** The model uses the measured average false-alarm rate; cameras that
   trigger persistently (simulated in phase 10) are a deliberate model mismatch.
6. **Scale.** With the measured detector (≈29 false alarms per camera-hour) and a camera seeing
   a real crash about once a year, $\mathrm{LR}_c - 1$ is of order $10^{-1}$ even in the top
   confidence bin and the prior for a camera's coverage is of order $10^{-5}$: a camera alarm
   alone stays far below any dispatch threshold (D-028). This is a property of the inputs, not
   of the formula; phase 10 sweeps detector quality to find the break-even point.
