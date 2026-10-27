# C2 — the confidence-aware dispatch policy

Status: phase 6.3 (D-031). Code: `src/resqcity/optimization/policy.py`. Parameters:
`configs/optimization.yaml` → `policy`. Tests: `tests/optimization/test_policy.py`.

C2 is **not** an optimisation model. It is the decision rule that chooses *which* incidents C1
is allowed to spend units on, and what happens to the rest. It sits between the fusion posterior
(phase 4) and the dispatch MILP (`docs/models/dispatch.md`).

## 1. Tiers

Let $p_i \in [0,1]$ be the fused posterior that incident $i$ is real, and
$0 < \tau_{\text{low}} < \tau_{\text{high}} \le 1$ the two thresholds.

| tier | condition | action |
|---|---|---|
| FULL | $p_i \ge \tau_{\text{high}}$ | every required unit is dispatched; C1 chooses which |
| VERIFY | $\tau_{\text{low}} \le p_i < \tau_{\text{high}}$ | the nearest police unit looks; the other required agencies are **pre-alerted** but do not move |
| REVIEW | $p_i < \tau_{\text{low}}$ | nothing moves; an operator is asked, with a timeout $T_{\text{review}}$ |

The rule is **monotone** by construction: raising $p_i$ never produces a weaker response
(`test_a_higher_posterior_never_yields_a_weaker_response`). Defaults are $\tau_{\text{low}} = 0.2$,
$\tau_{\text{high}} = 0.7$, $T_{\text{review}} = 120$ s, all UNVERIFIED and all swept in
phase 10.

## 2. Only FULL incidents enter the MILP

C1 is solved on the FULL subset alone. VERIFY and REVIEW incidents are deliberately excluded:
if they competed for units inside the objective, an unconfirmed candidate could displace an
incident known to be real, which is the failure confidence tiering exists to prevent.

The exclusion is why **unit exclusivity has to be enforced across tiers, not just inside the
MILP**. C1 guarantees one unit to one incident among the FULL set; the VERIFY assignments are
made afterwards, outside the solver. An OR review of this phase found the same unit being sent
both by C1 and by a later VERIFY tier. `verify_decision` therefore takes the set of units C1 has
already committed and masks their ETAs to $\infty$ before picking the nearest, and each VERIFY
assignment adds its own unit to that set:

$$
\eta^{\text{free}}_{u,i} =
\begin{cases}
\infty & u \in \text{taken} \\
\eta_{u,i} & \text{otherwise}
\end{cases}
\qquad
\text{taken} \leftarrow \text{taken} \cup \{\text{units assigned at this step}\}.
$$

`test_a_unit_committed_to_a_full_incident_cannot_also_verify_another` is the regression
test, with `test_a_spare_unit_still_verifies_when_one_is_free` to show the mask does not
simply suppress the look.

## 3. Escalation on timeout

A REVIEW candidate whose deadline expires with no operator verdict escalates to **VERIFY, not to
FULL**: something nobody has looked at should cost one police unit, not the whole requirement.
`policy.escalate_on_timeout` disables this for the phase-10 ablation.

Escalation is also the route by which a camera alarm can matter at all — see §5.

## 4. What the policy is *not* given

- **No queueing.** A VERIFY look that finds nothing releases its unit; the model does not charge
  for the trip. This flatters CCTV and is recorded as a limitation, not corrected here.
- **No operator model.** The review verdict is exogenous. C2 only records the deadline; the
  verdict itself will be drawn in phase 7 from `configs/fusion.yaml` -> `review`
  (sensitivity 0.95, specificity 0.98, duration 30 s, all UNVERIFIED). Operators are assumed
  always available, which is the same capacity blindness as L30.
- **No tier for "probably false".** A high-confidence *negative* is treated exactly like an
  unknown. With a measured camera likelihood ratio near 1 there is nothing to gain from
  distinguishing them.

## 5. What the measurements say this is worth

A camera alarm on its own multiplies the odds of a real incident by **1.01–1.04** (D-029), so a
camera-only candidate sits at a posterior of order $10^{-5}$ — far below any sensible
$\tau_{\text{low}}$, and therefore in REVIEW. **Confidence tiering does not turn CCTV into a
dispatch trigger**, and the thresholds cannot be tuned to make it one: no value of
$\tau_{\text{low}}$ above the base rate promotes an alarm whose evidence is this weak, and a
value below it promotes everything.

What remains is narrower and is the actual phase-10 hypothesis: a camera alarm already sitting in
REVIEW is **upgraded the moment a 112 call or a human review corroborates it**, which is earlier
than the call alone would have produced a dispatch, because the candidate, its location and its
requirement are already assembled. Whether that head start is worth anything is measured, not
assumed.

## 6. Parameters

| Symbol | Config key | Default | Status |
|---|---|---|---|
| $\tau_{\text{low}}$ | `policy.tau_low` | 0.2 | UNVERIFIED, swept |
| $\tau_{\text{high}}$ | `policy.tau_high` | 0.7 | UNVERIFIED, swept |
| $T_{\text{review}}$ | `policy.t_review_s` | 120 s | UNVERIFIED, swept |
| escalate on timeout | `policy.escalate_on_timeout` | true | design choice, ablated |
