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
|t_e - t_j^{\text{first}}| \le W ,
$$

where $d_G$ is the road-graph distance (direction ignored), $D_0$ = `gate.base_distance_m`,
$k$ = `gate.k_location_error`, $W$ = `gate.time_window_s` (measured from the candidate's FIRST evidence: a camera's
recurring false alarms must not chain into one endless candidate). Otherwise $e$ starts a new
candidate.
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

Calls are treated as a point process with times $t_1 < \dots < t_n$ (in the candidate). Under
$H_0$ all are background: Poisson with rate $b = \Lambda_F\, m(R)$ per second, so their density is
$b^n e^{-bT}$. Under $H_1$ the first $k$ calls are genuine and the rest background, summed over
$k = 0..n$; the common factor $e^{-bT}$ cancels:

$$
\mathrm{LR}_{\text{calls}} = \sum_{k=0}^{n} G_k\, b^{-k},
$$

where $G_k$ is the density under $H_1$ that exactly the first $k$ calls are genuine and no further
genuine call has come by now (the call process of `sim.calls`):

$$
G_0 = \mathbb{E}_s\!\left[p_u + (1-p_u)\,\bigl(1 - F_1(t_{\text{now}} - s)\bigr)\right],
\qquad
G_k = (1-p_u)\, \mathbb{E}_s\!\left[f_1(t_1 - s)\right]
\prod_{j=2}^{k} \frac{q_{j-1}}{g}\, e^{-(t_j - t_{j-1})/g}
\;\Bigl(1 - q_k\bigl(1 - e^{-(t_{\text{now}} - t_k)/g}\bigr)\Bigr),
$$

with $p_u$ = `p_unreported` of the candidate's severity class, $f_1, F_1$ the lognormal density
and CDF of the first-call delay, $N \sim \text{Poisson}(m)$ extra calls arriving one exponential gap
(mean $g$) after another, $q_j = P(N \ge j)/P(N \ge j-1)$, and $s$ the onset: pinned by a camera
(first alarm time − detector delay) or, without a camera, uniform over the window before the first
evidence. So a call at the typical delay after a camera alarm counts more than the same call
without a camera, and a candidate without calls loses belief as time passes ($G_0$ falls).
Adding a call (the latest in time) adds the non-negative term $G_{n+1} b^{-(n+1)}$ and leaves the
others unchanged, so a confirming call never lowers the posterior (tested). $G_0, G_1, G_2$ are
checked against a brute-force simulation of `sim.calls` (`tests/fusion/test_model.py`).

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
2. **Prefix assignment.** Under $H_1$ the genuine calls are taken to be the earliest ones (a
   prank before a genuine call is not considered); background calls about incidents outside the
   region are ignored.
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
