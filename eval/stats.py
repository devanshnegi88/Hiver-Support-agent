"""
Bootstrap confidence intervals and paired significance testing.

A golden set of 150-250 examples is small enough that point-estimate
differences between systems (e.g. "agent gets 0.81 F1, baseline gets 0.74")
can easily be noise. This module answers "how confident are we", not just
"what's the number" — the report should quote CIs, not bare point estimates.
"""
import numpy as np


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95,
                  seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of a metric computed per-example.
    `values` should be one score per golden-set example (e.g. 1.0/0.0 for
    correct/incorrect intent, or the judge's 1-5 score per example) — NOT
    a single pre-aggregated number, or there's nothing to resample.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def paired_bootstrap_test(values_a: list[float], values_b: list[float],
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap significance test for "is system A actually better
    than system B on this metric, or could the observed gap be noise".

    `values_a` and `values_b` must be per-example scores for the SAME
    examples in the SAME order (e.g. agent's per-example judge score vs.
    baseline's per-example judge score on the same golden-set rows) — the
    pairing is what makes this more powerful than comparing two independent
    CIs, since it cancels out per-example difficulty.

    Returns the observed mean difference, its CI, and a two-sided p-value
    (fraction of bootstrap resamples where the sign of the difference
    flips relative to the observed direction).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "paired test requires equal-length, aligned arrays"
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = diff[rng.integers(0, n, size=n)]
        boot_diffs[i] = sample.mean()

    observed = float(diff.mean())
    lo, hi = np.quantile(boot_diffs, [0.025, 0.975])
    # Two-sided p-value via the standard percentile-bootstrap construction:
    # 2x the smaller tail on either side of zero. Using min() of both tails
    # (rather than picking a tail based on the observed sign) keeps this
    # correct in the degenerate zero-variance case — e.g. two identical
    # inputs produce boot_diffs concentrated at exactly 0, which must yield
    # p=1 (no evidence of a difference), not p=0.
    p_value = 2 * min(float((boot_diffs <= 0).mean()), float((boot_diffs >= 0).mean()))
    p_value = min(1.0, p_value)

    return {
        "mean_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }
