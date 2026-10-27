# C3 — Pre-positioning with a fairness constraint

Status: phase 6 (D-031). Code: `src/resqcity/optimization/preposition.py`. Parameters:
`configs/optimization.yaml` → `preposition`. Tests: `tests/optimization/test_preposition.py`
(exhaustive search on tiny instances).

Every `every_minutes`, idle units are moved between bases so that the *next* incidents are closer
to a unit. The model is MEXCLP (maximum expected covering location) because a located unit is not
always free: it is busy with probability $q$, so the second unit covering a zone still adds value.
The fairness constraint is what makes this a research object rather than an engineering tweak —
it is solved twice, with and without, and the difference is the **price of fairness** (RQ4).

## 1. Sets and parameters

| Symbol | Units | Meaning |
|---|---|---|
| $j \in J$ | — | demand zones (wards) |
| $k \in K$ | — | candidate bases (stations) |
| $u \in U$ | — | idle units of one agency available to move |
| $d_j$ | count | forecast demand of zone $j$ over the next period (phase 5) |
| $t_{k,j}$ | s | travel time from base $k$ to zone $j$ |
| $N_j = \{k : t_{k,j} \le R\}$ | — | bases that cover $j$ within the coverage radius $R$ |
| $F_j = \{k : t_{k,j} \le T_{\text{ward}}\}$ | — | bases that reach $j$ within the fairness time |
| $q$ | — | probability a located unit is busy (`busy_fraction`) |
| $\alpha$ | — | required availability per ward (`fairness.alpha`) |
| $B$ | count | relocation budget, max moves (`relocation.max_moves`) |
| $C$ | s | relocation budget, max total travel (`relocation.max_total_s`) |
| $m_{u,k}$ | s | time for unit $u$ to move to base $k$ ($0$ if it is already there) |

## 2. Decision variables

| Symbol | Domain | Meaning |
|---|---|---|
| $y_{u,k}$ | $\{0,1\}$ | unit $u$ ends the period at base $k$ |
| $n_k$ | $\mathbb{Z}_{\ge 0}$ | units stationed at base $k$ (defined by $y$) |
| $z_{j,\ell}$ | $\{0,1\}$ | zone $j$ is covered by **at least** $\ell$ units, $\ell = 1 \dots L$ |

$L = \min\big(|U|,\ \max_j \sum_{k \in N_j} \mathrm{cap}_k\big)$ — the largest cover any zone can
actually receive. Capping $L$ at the largest *single* base capacity truncates the objective and
makes the solver move units away from a zone served by several bases (found by the OR review).

## 3. Objective

$$
\max \;\; \sum_{j \in J} d_j \sum_{\ell=1}^{L} (1-q)\,q^{\,\ell-1} z_{j,\ell}
$$

The inner sum telescopes to $d_j\,(1 - q^{\,n_j})$, the expected demand of zone $j$ served by at
least one free unit, where $n_j = \sum_{k \in N_j} n_k$. Writing it with the incremental
variables $z_{j,\ell}$ keeps the model linear, which is the standard MEXCLP formulation. The
coefficients $(1-q)q^{\ell-1}$ are decreasing in $\ell$, so the solver fills first cover before
second cover without needing an ordering constraint.

## 4. Constraints

$$
\sum_{k \in K} y_{u,k} = 1 \qquad \forall u \in U \tag{P1}
$$
Every unit ends somewhere (staying put is $y_{u,k^{\text{now}}} = 1$).

$$
n_k = \sum_{u \in U} y_{u,k} \le \text{cap}_k \qquad \forall k \in K \tag{P2}
$$

$$
\sum_{\ell=1}^{L} z_{j,\ell} \;\le\; \sum_{k \in N_j} n_k \qquad \forall j \in J \tag{P3}
$$
A zone cannot claim more cover than it has.

$$
\sum_{u \in U} \sum_{k : k \ne k_u^{\text{now}}} y_{u,k} \;\le\; B,
\qquad
\sum_{u,k} m_{u,k}\, y_{u,k} \;\le\; C \tag{P4}
$$
The relocation budget: at most $B$ moves and $C$ seconds of driving in total. Without it the
model would reshuffle the whole fleet every period.

$$
\sum_{k \in F_j} n_k \;\ge\; \lceil \log(1-\alpha) / \log q \rceil \qquad \forall j \in J \tag{P5}
$$
**The fairness constraint.** Each ward must have enough units within $T_{\text{ward}}$ that at
least one is free with probability $\ge \alpha$. Since $P(\text{at least one free}) = 1 - q^{n}$,
requiring $1 - q^{n} \ge \alpha$ gives $n \ge \log(1-\alpha)/\log q$. The right-hand side is a
constant computed once from the configuration, so (P5) stays linear.

When fairness is disabled, (P5) is dropped. The **price of fairness** is the relative loss in the
objective:
$$
\text{PoF} = \frac{V^{\text{free}} - V^{\text{fair}}}{V^{\text{free}}}
$$

## 5. Feasibility

(P5) can be impossible — a ward with no base within $T_{\text{ward}}$, or too few units to meet
every ward's requirement at once. Rather than detect that case, the implementation writes (P5) as
a soft constraint **always**: a per-ward shortfall variable $g_j \ge 0$ with

$$
\sum_{k : \tau_{jk} \le T_{\text{ward}}} n_k + g_j \ge n_\alpha ,
\qquad\text{objective } \dots - M \sum_j g_j .
$$

One solve then reports *which* wards cannot be served instead of returning "infeasible". A city
that cannot cover its wards is a finding, not an error.

This is only honest if $M$ cannot buy coverage. $M$ is therefore **derived from the instance**,
not assumed. The coverage objective is largest when every $z_{j\ell} = 1$, so its upper bound is
the sum of its own integer coefficients $c_{j\ell}$:

$$
M = \sum_{j,\ell} \max(c_{j\ell},\, 0) + 1,
\qquad c_{j\ell} = \big\lfloor S\, d_j (1-q) q^{\ell-1} \big\rceil
$$

with $S$ the integer scale. One unit of shortfall then costs more than all the coverage there is
to win, which makes the soft constraint **equivalent to the hard one wherever the hard one is
satisfiable**. Two details are why the bound comes from the coefficients rather than from
$S \sum_j d_j$: each $c_{j\ell}$ is rounded, which over 58 wards and several cover levels is not
negligible slack; and summing $\ell$ only to $\ell_{\max}$ gives $1 - q^{\ell_{\max}}$ rather
than 1, so the coefficient sum is also strictly **tighter**.
`tests/optimization/test_preposition.py::test_the_soft_fairness_constraint_behaves_like_a_hard_one`
checks that equivalence against a genuinely hard-constrained solve, and
`results/optimization/preposition.json` records the penalty actually used. An earlier version
used a fixed $M = 10^4$, which an OR review flagged: a constant cannot carry that guarantee, and
the formulation claimed a detect-then-relax step the code never performed.

## 6. Assumptions and what is left out

1. **A single agency at a time.** Ambulance, police and fire are pre-positioned independently;
   the model does not co-locate them for joint response.
2. **$q$ is one number for the whole city**, not per ward or per hour. It is UNVERIFIED.
3. **Travel times are the hour-bucket estimates** of phase 2, as everywhere else.
4. **Demand comes from the phase-5 forecaster**, whose useful range is narrow (D-030): the top-5
   ward hit rate runs from 0.083 with no information to 0.237 with perfect knowledge. Pre-positioning
   therefore has limited headroom, and the price of fairness may exceed its benefit. That is a
   result to report, not a reason to change the model.
5. **No queueing.** Expected coverage is a static proxy; phase 7 simulates the real thing and
   phase 6.6 (Erlang-C) gives the analytical cross-check.
