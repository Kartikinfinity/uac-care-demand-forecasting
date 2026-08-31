"""
metrics.py -- Forecast error metrics for the walk-forward harness.

Every function here is a pure function over arrays. No I/O, no model objects,
no fold logic. All functions are NaN-safe in the same way: pairs where either
the actual or the forecast is missing are dropped, and the number of pairs
actually scored is reported alongside the metric, never hidden.

Metric selection rationale (roadmap Part 6 + addendum):
  - MAE / RMSE   : reported for every model/target/horizon.
  - MAPE         : safe for Target 1 (HHS Care never reaches 0, min 1,972);
                   UNSTABLE for the flow targets, which contain true zero-days.
                   Zero actuals are excluded and the exclusion count returned.
  - sMAPE / MASE : reported alongside MAPE wherever a series has zero-days.
  - ME (bias)    : signed mean error, for systematic over/under-forecasting.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "align_pairs",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "pooled_mase",
    "mean_error",
    "seasonal_naive_scale",
    "compute_all_metrics",
]


def align_pairs(y_true, y_pred):
    """Return (y_true, y_pred) as float arrays with any NaN-containing pair dropped."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def mae(y_true, y_pred) -> float:
    a, f = align_pairs(y_true, y_pred)
    return float(np.mean(np.abs(a - f))) if a.size else float("nan")


def rmse(y_true, y_pred) -> float:
    a, f = align_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((a - f) ** 2))) if a.size else float("nan")


def mean_error(y_true, y_pred) -> float:
    """Signed mean error (forecast - actual). Positive = systematic over-forecast."""
    a, f = align_pairs(y_true, y_pred)
    return float(np.mean(f - a)) if a.size else float("nan")


def mape(y_true, y_pred, return_excluded: bool = False):
    """
    Mean absolute percentage error, in percent.

    Zero actuals are undefined for MAPE and are excluded rather than clipped or
    epsilon-padded. The count of excluded points is available so the instability
    flagged in the roadmap for zero-day series is visible, not silent.
    """
    a, f = align_pairs(y_true, y_pred)
    nonzero = a != 0
    excluded = int((~nonzero).sum())
    a, f = a[nonzero], f[nonzero]
    value = float(np.mean(np.abs((a - f) / a)) * 100.0) if a.size else float("nan")
    return (value, excluded) if return_excluded else value


def smape(y_true, y_pred) -> float:
    """
    Symmetric MAPE, in percent, on the 0-200 convention:
        mean( |a - f| / ((|a| + |f|) / 2) ) * 100

    A pair where both actual and forecast are exactly zero contributes 0 (a
    perfect forecast), rather than producing a 0/0 NaN.
    """
    a, f = align_pairs(y_true, y_pred)
    if not a.size:
        return float("nan")
    denom = (np.abs(a) + np.abs(f)) / 2.0
    ratio = np.zeros_like(denom, dtype=float)
    nz = denom != 0
    ratio[nz] = np.abs(a[nz] - f[nz]) / denom[nz]
    return float(np.mean(ratio) * 100.0)


def seasonal_naive_scale(y_train, m: int) -> float:
    """
    MASE denominator: the mean absolute error of the in-sample seasonal-naive
    forecast on the TRAINING window only (Hyndman & Koehler).

    Computed strictly from training data, so it carries no information from the
    fold's test points. NaN-affected differences are dropped (the flow targets
    are genuinely missing at gap slots and are never imputed).

    Returns NaN if fewer than one usable seasonal difference exists.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    if y.size <= m:
        return float("nan")
    diffs = np.abs(y[m:] - y[:-m])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    scale = float(np.mean(diffs))
    return scale if scale > 0 else float("nan")


def mase(y_true, y_pred, scale: float) -> float:
    """
    Mean absolute scaled error, given a pre-computed training-window `scale`
    from `seasonal_naive_scale`. MASE < 1 means the forecast beats the in-sample
    seasonal-naive benchmark.
    """
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    m = mae(y_true, y_pred)
    return float(m / scale) if np.isfinite(m) else float("nan")


def pooled_mase(y_true, y_pred, scales) -> float:
    """
    MASE pooled across folds that each carry their own training-window scale.

    Because the walk-forward harness refits at every origin, the in-sample
    seasonal-naive scale differs per fold. Dividing each absolute error by ITS
    OWN fold's scale before averaging keeps every fold on a comparable footing;
    averaging raw errors first and dividing by a single blended scale would let
    high-level early folds dominate the number.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    scales = np.asarray(scales, dtype=float).ravel()
    if not (y_true.shape == y_pred.shape == scales.shape):
        raise ValueError("y_true, y_pred and scales must be the same length")
    keep = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(scales)
        & (scales > 0)
    )
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[keep] - y_pred[keep]) / scales[keep]))


def compute_all_metrics(y_true, y_pred, scale: float | None = None) -> dict:
    """
    Every metric in one dict, plus the accounting numbers that make the metric
    interpretable: how many pairs were actually scored, and how many were
    dropped for a missing actual, a missing forecast, or a zero actual (MAPE).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n_total = int(y_true.size)
    n_missing_actual = int((~np.isfinite(y_true)).sum())
    n_missing_pred = int((np.isfinite(y_true) & ~np.isfinite(y_pred)).sum())

    mape_value, n_zero_actual = mape(y_true, y_pred, return_excluded=True)
    a, _ = align_pairs(y_true, y_pred)

    return {
        "n_scored": int(a.size),
        "n_total": n_total,
        "n_missing_actual": n_missing_actual,
        "n_missing_pred": n_missing_pred,
        "n_zero_actual_excluded_from_mape": n_zero_actual,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape_value,
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, scale) if scale is not None else float("nan"),
        "ME_bias": mean_error(y_true, y_pred),
    }
