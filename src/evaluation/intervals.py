"""
intervals.py -- Empirical prediction intervals and coverage calibration.

Addendum Section 6 fixes the approach: intervals are EMPIRICAL RESIDUAL
QUANTILES, built from out-of-sample walk-forward residuals restricted to folds
with origin on or after the 2025-02-05 cutoff -- the same cutoff as training,
deliberately a single hard boundary rather than a continuously-weighted recency
scheme (which Section 10 lists as an explicit non-goal).

Two rules this module exists to enforce:

  * "Never build a prediction interval from in-sample residuals" (invariant 5).
    Every residual here comes from a fold's TEST point, produced by a model that
    never saw it.
  * Coverage is reported "with a binomial confidence band sized to the actual
    achieved N (expected to be small -- low tens, not hundreds -- and disclosed
    as such)". A bare hit-rate at n=12 would imply precision that does not
    exist, so `coverage_with_band` always returns the band alongside it.

On the sample sizes actually available: the post-cutoff residual pools hold
12-15 observations per target/horizon. At the 95% level the 2.5th and 97.5th
percentiles of such a sample ARE its minimum and maximum, which is why
`empirical_interval` refuses to emit anything below MIN_RESIDUALS_FOR_INTERVAL
rather than returning a band with no information in it. This is a real
limitation of a 660-observation development portion split into 65 folds, and it
is disclosed rather than engineered around.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    EMPIRICAL_INTERVAL_ALPHA,
    IMBALANCE_INDEPENDENCE_THRESHOLD,
    MIN_RESIDUALS_FOR_INTERVAL,
)

__all__ = [
    "empirical_interval",
    "wilson_interval",
    "coverage_with_band",
    "imbalance_variance",
]


def empirical_interval(
    point_forecast: float,
    residuals,
    alpha: float = EMPIRICAL_INTERVAL_ALPHA,
    min_residuals: int = MIN_RESIDUALS_FOR_INTERVAL,
    lower_bound: float = None,
) -> dict:
    """
    Interval around `point_forecast` from the empirical quantiles of `residuals`.

    Residuals are defined actual - forecast, so the interval is
    [forecast + q(alpha/2), forecast + q(1 - alpha/2)]. Using the residual
    quantiles directly -- rather than a symmetric +/- multiple of their standard
    deviation -- keeps any genuine skew in the error distribution, which a
    Gaussian assumption would erase.

    `lower_bound`, when given, clips the lower end to the domain's natural floor.
    Both targets are COUNTS OF CHILDREN and cannot be negative, so an unclipped
    empirical quantile can produce a lower bound below zero -- statistically
    valid as a quantile, operationally impossible as a forecast, and misleading
    if published to a stakeholder. The unclipped value is kept as
    `lower_unclipped` so the clipping is visible rather than silent, and the
    coverage calculations deliberately do NOT clip, so calibration numbers are
    never flattered by it.

    Returns `lower`/`upper` as NaN, with `sufficient=False` and a reason, when
    the pool is too thin. An interval that is really just the sample min and max
    should not be published as a 95% band.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = int(residuals.size)

    base = {
        "n_residuals": n,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "point_forecast": float(point_forecast),
    }
    if n < min_residuals or not np.isfinite(point_forecast):
        base.update({
            "lower": float("nan"), "upper": float("nan"), "width": float("nan"),
            "sufficient": False,
            "reason": ("only %d residuals available; below the %d-residual floor, "
                       "the %.0f%% quantiles are just the sample min and max"
                       % (n, min_residuals, (1 - alpha) * 100))
            if n < min_residuals else "point forecast unavailable",
        })
        return base

    lo_q, hi_q = np.percentile(residuals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    lower = float(point_forecast + lo_q)
    upper = float(point_forecast + hi_q)
    clipped = lower_bound is not None and lower < lower_bound
    base.update({
        "lower": float(lower_bound) if clipped else lower,
        "lower_unclipped": lower,
        "clipped_at_lower_bound": bool(clipped),
        "upper": upper,
        "width": float(upper - (lower_bound if clipped else lower)),
        "residual_q_low": float(lo_q),
        "residual_q_high": float(hi_q),
        "sufficient": True,
        "reason": "",
    })
    return base


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple:
    """
    Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because at n=12-15, and with hit
    rates near 1.0, the normal approximation produces intervals that run past
    100% or collapse to zero width -- exactly the regime this project is in.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    from math import sqrt

    # Two-sided z for the requested level (avoids a scipy dependency here).
    z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(level, 1.9600)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def coverage_with_band(
    actuals,
    lowers,
    uppers,
    nominal: float,
    band_level: float = 0.95,
) -> dict:
    """
    Empirical interval coverage, always reported with its uncertainty.

    Addendum Section 6 requires the binomial confidence band to be "sized to the
    actual achieved N". `covers_nominal` says whether the nominal level falls
    inside that band -- i.e. whether the observed gap is distinguishable from
    sampling noise at this N, which at low tens it usually will not be. A gap
    that IS outside the band is a real calibration failure and belongs in
    Limitations, never silently patched by widening the interval after the fact.
    """
    actuals = np.asarray(actuals, dtype=float)
    lowers = np.asarray(lowers, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    usable = np.isfinite(actuals) & np.isfinite(lowers) & np.isfinite(uppers)
    a, lo, hi = actuals[usable], lowers[usable], uppers[usable]
    n = int(a.size)

    if n == 0:
        return {"n": 0, "n_covered": 0, "empirical_coverage": float("nan"),
                "nominal_coverage": nominal, "band_low": float("nan"),
                "band_high": float("nan"), "covers_nominal": False,
                "note": "no scorable points"}

    covered = int(((a >= lo) & (a <= hi)).sum())
    rate = covered / n
    band_low, band_high = wilson_interval(covered, n, band_level)
    return {
        "n": n,
        "n_covered": covered,
        "empirical_coverage": float(rate),
        "nominal_coverage": float(nominal),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "covers_nominal": bool(band_low <= nominal <= band_high),
        "note": ("N is small (%d); the band is correspondingly wide and this is "
                 "not a precise calibration estimate" % n) if n < 40 else "",
    }


def imbalance_variance(
    var_a: float,
    var_b: float,
    covariance: float = 0.0,
    correlation: float = None,
    independence_threshold: float = IMBALANCE_INDEPENDENCE_THRESHOLD,
) -> dict:
    """
    Variance of the derived imbalance signal A - B (Transferred Out minus
    Discharged).

    Addendum Section 6: "Combined uncertainty uses
    Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B), with Cov measured from the two
    components' actual paired out-of-sample residuals... The simplified
    independence form is used only if that measured correlation supports it."

    So the FULL form is the default and the independence simplification is a
    concession the measurement has to earn. `form_used` records which one
    applied, so a reader can see the decision rather than infer it.
    """
    full = float(var_a) + float(var_b) - 2.0 * float(covariance)
    independent = float(var_a) + float(var_b)

    use_independence = (
        correlation is not None
        and np.isfinite(correlation)
        and abs(correlation) < independence_threshold
    )
    variance = independent if use_independence else full
    variance = max(variance, 0.0)  # a variance cannot be negative
    return {
        "variance": variance,
        "std": float(np.sqrt(variance)),
        "form_used": "independence" if use_independence else "full_covariance",
        "var_a": float(var_a),
        "var_b": float(var_b),
        "covariance": float(covariance),
        "correlation": None if correlation is None else float(correlation),
        "full_form_variance": max(full, 0.0),
        "independence_form_variance": independent,
    }
