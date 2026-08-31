"""
tests/test_metrics.py -- Day 5 metric correctness.

Every metric is checked against a hand-computed value, not against another
implementation of itself, plus the NaN / zero-actual behaviour the flow targets
actually exercise.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import (
    align_pairs,
    compute_all_metrics,
    mae,
    mape,
    mase,
    mean_error,
    pooled_mase,
    rmse,
    seasonal_naive_scale,
    smape,
)


# ----------------------------------------------------------------------
# Point metrics, against hand-computed values
# ----------------------------------------------------------------------
def test_mae_hand_computed():
    # errors: 2, 2, 4  -> mean 8/3
    assert mae([10, 20, 30], [12, 18, 34]) == pytest.approx(8 / 3)


def test_rmse_hand_computed():
    # squared errors: 9, 16 -> mean 12.5 -> sqrt 3.5355...
    assert rmse([10, 20], [13, 24]) == pytest.approx(np.sqrt(12.5))


def test_rmse_penalises_large_errors_more_than_mae():
    spread = ([100, 100], [90, 110])   # errors 10, 10
    concentrated = ([100, 100], [80, 100])  # errors 20, 0
    assert mae(*spread) == mae(*concentrated)
    assert rmse(*concentrated) > rmse(*spread)


def test_mean_error_is_signed_and_directional():
    # forecast - actual: +5 and +5 -> systematic over-forecast
    assert mean_error([10, 20], [15, 25]) == pytest.approx(5.0)
    assert mean_error([10, 20], [5, 15]) == pytest.approx(-5.0)


def test_mape_hand_computed():
    # |2/10| + |4/20| = 0.2 + 0.2 -> mean 0.2 -> 20%
    assert mape([10, 20], [12, 24]) == pytest.approx(20.0)


def test_smape_hand_computed():
    # |10-12| / ((10+12)/2) = 2/11 -> 18.1818...%
    assert smape([10], [12]) == pytest.approx(2 / 11 * 100)


def test_smape_is_bounded_at_200_percent():
    # The 0-200 convention: an actual of 0 against a positive forecast saturates.
    assert smape([0], [50]) == pytest.approx(200.0)


def test_smape_treats_zero_against_zero_as_perfect_not_nan():
    """A true zero-day forecast exactly right must score 0, not 0/0 -> NaN."""
    assert smape([0, 10], [0, 10]) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Zero-day / missing-data behaviour (the flow targets actually hit these)
# ----------------------------------------------------------------------
def test_mape_excludes_zero_actuals_and_reports_the_count():
    """Zero actuals are excluded, never epsilon-padded into a finite number."""
    value, excluded = mape([0, 10], [5, 12], return_excluded=True)
    assert excluded == 1
    assert value == pytest.approx(20.0)  # only the 10 -> 12 pair scored


def test_mape_is_nan_when_every_actual_is_zero():
    assert np.isnan(mape([0, 0], [1, 2]))


def test_align_pairs_drops_pairs_with_a_missing_actual_or_forecast():
    a, f = align_pairs([1, np.nan, 3, 4], [1, 2, np.nan, 4])
    np.testing.assert_array_equal(a, [1, 4])
    np.testing.assert_array_equal(f, [1, 4])


def test_metrics_never_impute_a_missing_value():
    """A missing actual must shrink N, never be filled with a zero or a mean."""
    assert mae([10, np.nan], [12, 999]) == pytest.approx(2.0)


def test_align_pairs_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        align_pairs([1, 2, 3], [1, 2])


def test_all_metrics_are_nan_on_an_empty_overlap():
    stats = compute_all_metrics([np.nan, np.nan], [1.0, 2.0])
    assert stats["n_scored"] == 0
    for key in ("MAE", "RMSE", "MAPE", "sMAPE", "ME_bias"):
        assert np.isnan(stats[key])


# ----------------------------------------------------------------------
# MASE
# ----------------------------------------------------------------------
def test_seasonal_naive_scale_hand_computed():
    # m=2: |3-1| + |4-2| + |5-3| = 2+2+2 -> mean 2.0
    assert seasonal_naive_scale([1, 2, 3, 4, 5], m=2) == pytest.approx(2.0)


def test_seasonal_naive_scale_ignores_missing_differences():
    """Flow-target gaps are true-missing; they drop out of the scale, not zero-fill it."""
    assert seasonal_naive_scale([1, 2, np.nan, 4, 5], m=2) == pytest.approx(2.0)


def test_seasonal_naive_scale_is_nan_when_history_is_too_short():
    assert np.isnan(seasonal_naive_scale([1, 2, 3], m=5))


def test_seasonal_naive_scale_is_nan_on_a_flat_series():
    """A zero scale would make every MASE infinite; it is reported as NaN instead."""
    assert np.isnan(seasonal_naive_scale([7, 7, 7, 7, 7, 7], m=2))


def test_mase_below_one_means_the_forecast_beats_seasonal_naive():
    assert mase([100, 100], [101, 99], scale=10.0) == pytest.approx(0.1)


def test_mase_is_nan_for_an_unusable_scale():
    assert np.isnan(mase([1, 2], [1, 2], scale=0.0))
    assert np.isnan(mase([1, 2], [1, 2], scale=float("nan")))


def test_pooled_mase_scales_each_fold_by_its_own_denominator():
    """
    Two folds, equal absolute error of 10, but very different fold scales.
    Pooled MASE must be mean(10/10, 10/100) = 0.55, NOT 10/mean(10,100) = 0.18.
    """
    result = pooled_mase([100, 100], [110, 110], [10.0, 100.0])
    assert result == pytest.approx(0.55)


def test_pooled_mase_skips_rows_with_an_unusable_scale():
    assert pooled_mase([100, 100], [110, 110], [10.0, 0.0]) == pytest.approx(1.0)


def test_pooled_mase_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pooled_mase([1, 2], [1, 2], [1.0])


# ----------------------------------------------------------------------
# The bundled report
# ----------------------------------------------------------------------
def test_compute_all_metrics_reports_its_own_accounting():
    stats = compute_all_metrics(
        y_true=[10.0, 0.0, np.nan, 40.0],
        y_pred=[12.0, 1.0, 30.0, np.nan],
        scale=4.0,
    )
    assert stats["n_total"] == 4
    assert stats["n_missing_actual"] == 1
    assert stats["n_missing_pred"] == 1
    assert stats["n_scored"] == 2                       # (10,12) and (0,1)
    assert stats["n_zero_actual_excluded_from_mape"] == 1
    assert stats["MAE"] == pytest.approx(1.5)           # (2 + 1) / 2
    assert stats["MASE"] == pytest.approx(1.5 / 4.0)
