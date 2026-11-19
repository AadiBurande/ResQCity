"""Statistical tests for paired policy comparisons (phase 10).

All comparisons use **common random numbers**: for a given (scenario, replication) pair the
identical incident week is generated regardless of policy, so per-replication paired differences
are exact. That eliminates inter-week variance and is the reason a paired test (rather than
independent-samples) is correct.

Statistical plan (from ``docs/plans/phase-10.md``):

1. Paired differences per replication (CRN guarantees pairing).
2. Normality check: Shapiro-Wilk on paired differences (alpha = 0.05).
3. Test: paired t-test if normal, Wilcoxon signed-rank otherwise.
4. Multiple comparisons: Holm-Bonferroni family-wise correction.
5. Effect sizes: Cohen's d (parametric) or matched-rank biserial (nonparametric) + 95%
   bootstrap CIs on the mean difference.
6. Equity (RQ4): one-sided non-inferiority test on ward-level Gini (delta = 0.02).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class PairedTestResult:
    """Result of one paired comparison."""

    metric: str
    policy_a: str
    policy_b: str
    n: int = 0

    # descriptive
    mean_a: float = float("nan")
    mean_b: float = float("nan")
    mean_diff: float = float("nan")
    std_diff: float = float("nan")

    # test
    test_name: str = ""
    statistic: float = float("nan")
    p_value: float = float("nan")
    p_adjusted: float = float("nan")

    # effect size
    effect_size: float = float("nan")
    effect_name: str = ""

    # bootstrap CI on mean difference
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    # interpretation
    normal: bool = True
    significant: bool = False
    reading: str = ""


@dataclass
class NonInferiorityResult:
    """Result of a one-sided non-inferiority test for equity (RQ4)."""

    metric: str
    policy_full: str
    policy_baseline: str
    delta: float = 0.02
    n: int = 0
    mean_full: float = float("nan")
    mean_baseline: float = float("nan")
    mean_diff: float = float("nan")
    t_statistic: float = float("nan")
    p_value: float = float("nan")
    non_inferior: bool = False
    reading: str = ""


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of *diffs*."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    gen = rng or np.random.default_rng(42)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = gen.choice(diffs, size=len(diffs), replace=True)
        means[i] = sample.mean()
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# paired test
# ---------------------------------------------------------------------------


def paired_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    metric: str,
    policy_a: str,
    policy_b: str,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Run a paired comparison (CRN guarantees pairing).

    Uses paired t-test if Shapiro-Wilk does not reject normality, else Wilcoxon signed-rank.
    """
    result = PairedTestResult(metric=metric, policy_a=policy_a, policy_b=policy_b)

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    # align lengths
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    result.n = n

    if n < 3:
        result.reading = f"Too few paired observations ({n}) to test."
        return result

    result.mean_a = float(np.mean(a))
    result.mean_b = float(np.mean(b))

    diffs = a - b
    result.mean_diff = float(np.mean(diffs))
    result.std_diff = float(np.std(diffs, ddof=1))

    # normality check on differences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n >= 8:
            _, p_shapiro = sp_stats.shapiro(diffs)
            result.normal = p_shapiro > alpha
        else:
            # too few for Shapiro-Wilk to be meaningful; default to nonparametric
            result.normal = False

    if result.normal:
        # paired t-test
        stat, p = sp_stats.ttest_rel(a, b)
        result.test_name = "paired_t"
        result.statistic = float(stat)
        result.p_value = float(p)
        # Cohen's d for paired samples
        if result.std_diff > 0:
            result.effect_size = float(result.mean_diff / result.std_diff)
            result.effect_name = "cohens_d"
    else:
        # Wilcoxon signed-rank
        nonzero = diffs[diffs != 0]
        if len(nonzero) < 1:
            result.test_name = "wilcoxon"
            result.statistic = float("nan")
            result.p_value = 1.0
            result.reading = "All differences are zero."
            return result
        stat, p = sp_stats.wilcoxon(nonzero, alternative="two-sided")
        result.test_name = "wilcoxon"
        result.statistic = float(stat)
        result.p_value = float(p)
        # matched-rank biserial r = 1 - 2W / (n*(n+1)/2)
        nn = len(nonzero)
        w_max = nn * (nn + 1) / 2
        if w_max > 0:
            result.effect_size = float(1.0 - 2.0 * stat / w_max)
            result.effect_name = "rank_biserial"

    # bootstrap CI on mean difference
    result.ci_low, result.ci_high = _bootstrap_ci(diffs, alpha=alpha)

    return result


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def holm_bonferroni(results: list[PairedTestResult], alpha: float = 0.05) -> list[PairedTestResult]:
    """Apply Holm-Bonferroni correction to a family of paired tests.

    Modifies ``p_adjusted`` and ``significant`` in place, and returns the same list sorted by
    original p-value.
    """
    # sort by raw p-value
    ordered = sorted(results, key=lambda r: r.p_value)
    m = len(ordered)

    for rank, r in enumerate(ordered):
        # Holm adjustment: p_adj = p * (m - rank)
        p_adj = min(1.0, r.p_value * (m - rank))
        # enforce monotonicity: p_adj[i] >= p_adj[i-1]
        if rank > 0:
            p_adj = max(p_adj, ordered[rank - 1].p_adjusted)
        r.p_adjusted = p_adj
        r.significant = p_adj < alpha

    # generate readings
    for r in ordered:
        direction = "lower" if r.mean_diff < 0 else "higher"
        sig = "significant" if r.significant else "not significant"
        r.reading = (
            f"{r.policy_a} vs {r.policy_b} on {r.metric}: "
            f"mean diff = {r.mean_diff:+.1f} s ({r.policy_a} is {direction}), "
            f"{r.test_name} p = {r.p_value:.4f}, "
            f"adjusted p = {r.p_adjusted:.4f} ({sig}), "
            f"{r.effect_name} = {r.effect_size:.3f}, "
            f"95% CI [{r.ci_low:.1f}, {r.ci_high:.1f}]"
        )

    return ordered


# ---------------------------------------------------------------------------
# equity non-inferiority test (RQ4)
# ---------------------------------------------------------------------------


def non_inferiority_test(
    ginis_full: np.ndarray,
    ginis_baseline: np.ndarray,
    policy_full: str = "FULL",
    policy_baseline: str = "B0",
    delta: float = 0.02,
    alpha: float = 0.05,
) -> NonInferiorityResult:
    """One-sided test: H0: Gini(FULL) - Gini(B0) >= delta vs H1: < delta.

    Non-inferiority is established if the upper bound of the one-sided CI for the difference
    is below delta.
    """
    result = NonInferiorityResult(
        metric="ward_gini",
        policy_full=policy_full,
        policy_baseline=policy_baseline,
        delta=delta,
    )

    full = np.asarray(ginis_full, dtype=float)
    base = np.asarray(ginis_baseline, dtype=float)
    n = min(len(full), len(base))
    full, base = full[:n], base[:n]
    result.n = n

    if n < 3:
        result.reading = f"Too few paired observations ({n})."
        return result

    result.mean_full = float(np.mean(full))
    result.mean_baseline = float(np.mean(base))

    diffs = full - base  # positive means FULL is MORE unequal
    result.mean_diff = float(np.mean(diffs))

    # one-sided t-test: H0: mean_diff >= delta
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    if se == 0:
        result.non_inferior = result.mean_diff < delta
        result.reading = "Zero variance in paired differences."
        return result

    t_stat = (result.mean_diff - delta) / se
    result.t_statistic = float(t_stat)
    result.p_value = float(sp_stats.t.cdf(t_stat, df=n - 1))  # left tail
    result.non_inferior = result.p_value < alpha

    verdict = "NON-INFERIOR" if result.non_inferior else "NOT ESTABLISHED"
    result.reading = (
        f"Equity ({policy_full} vs {policy_baseline}): "
        f"mean Gini diff = {result.mean_diff:+.4f}, "
        f"delta = {delta}, "
        f"t = {result.t_statistic:.3f}, p = {result.p_value:.4f} → {verdict}"
    )

    return result
