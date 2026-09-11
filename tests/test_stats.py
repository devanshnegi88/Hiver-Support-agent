from stats import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_mean_matches_sample_mean():
    values = [1.0, 1.0, 0.0, 1.0, 0.0]
    result = bootstrap_ci(values, n_boot=500, seed=1)
    assert abs(result["mean"] - 0.6) < 1e-9


def test_bootstrap_ci_bounds_contain_mean():
    values = [1.0] * 20 + [0.0] * 5
    result = bootstrap_ci(values, n_boot=1000, seed=1)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bootstrap_ci_is_narrow_for_large_unanimous_sample():
    values = [1.0] * 200
    result = bootstrap_ci(values, n_boot=500, seed=1)
    assert result["ci_low"] == result["ci_high"] == 1.0


def test_paired_bootstrap_detects_clear_difference():
    # system A always scores 1 point higher than system B, every example —
    # this should register as significant with a tight, positive CI
    a = [5.0] * 30
    b = [4.0] * 30
    result = paired_bootstrap_test(a, b, n_boot=1000, seed=1)
    assert result["mean_diff"] == 1.0
    assert result["significant_at_0.05"] is True
    assert result["ci_low"] > 0


def test_paired_bootstrap_no_difference_not_significant():
    a = [3.0, 4.0, 2.0, 5.0, 3.0] * 6
    b = list(a)  # identical -> zero diff every time
    result = paired_bootstrap_test(a, b, n_boot=1000, seed=1)
    assert result["mean_diff"] == 0.0
    assert result["significant_at_0.05"] is False


def test_paired_bootstrap_requires_equal_length():
    import pytest
    with pytest.raises(AssertionError):
        paired_bootstrap_test([1.0, 2.0], [1.0])
