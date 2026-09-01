"""
statistical.py -- SARIMAX and ETSModel forecasters (Day 6).

Both classes expose the same tiny interface the Day-5 walk-forward harness
already drives -- `.fit(y) -> self` and `.predict(horizons) -> array` -- so the
statistical family is scored on the IDENTICAL folds as the baselines, with no
separate code path (addendum Section 2).

SINGLE FIT PER TARGET (addendum Section 2, restated at Day 6):
    "ARIMA/SARIMA and Exponential Smoothing are each fit once per target;
     forecasts at h=1/7/14 come from that single model's native multi-step
     forecast function."

So `.fit()` estimates ONE model and `.predict([1, 7, 14])` reads positions
1, 7 and 14 out of a single 14-step-ahead path. This is deliberately unlike
Random Forest / HistGradientBoosting at Day 7, which use direct multi-horizon
fitting (three models per target).

Library choices are fixed by the addendum, each for a measured reason:

  SARIMAX   -- handles missing observations natively via the Kalman filter, so
               the flow target's true-missing gap slots (invariant 2: never
               zero-filled, never interpolated) can be passed straight through.

  ETSModel  -- the state-space implementation, with missing='drop'. The classic
               `statsmodels.tsa.holtwinters.ExponentialSmoothing` does NOT
               raise on NaN input; it silently returns an all-NaN forecast.
               Verified live on the masked development series (n=706, 46 NaN):
               classic -> [nan nan nan], ETSModel(missing='drop') -> real
               values. `tests/test_statistical.py` pins this exact failure mode.

Fit failures are contained, never fatal: a fold whose fit does not converge
records NaN forecasts and a reason string, and the batch continues (addendum
Section 3: "log and continue past an individual model-fit failure within the
batch"). A NaN forecast is dropped at metric time and counted, exactly like a
baseline abstention -- it is never silently replaced with a fallback number.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    ETS_MISSING_POLICY,
    NATIVE_CI_ALPHA,
    SEASONAL_PERIOD_M,
)

__all__ = ["StatisticalForecaster", "SarimaxForecaster", "ETSForecaster"]


class StatisticalForecaster:
    """
    Shared plumbing: fit-once, forecast-a-single-path, read the requested
    horizons out of it, and capture the model's native confidence interval as a
    secondary diagnostic (addendum Section 6 -- the PRIMARY intervals are the
    empirical residual-quantile ones built at Day 9, not these).
    """

    name = "statistical"

    def __init__(self, alpha: float = NATIVE_CI_ALPHA):
        self.alpha = alpha
        self.result_ = None
        self.fit_failed_ = False       # estimation stage
        self.forecast_failed_ = False  # forecast stage -- tracked separately so
                                       # the report cannot conflate the two
        self.failure_reason_ = ""
        self.n_dropped_ = 0
        self.last_interval_ = None  # (lo, hi) arrays aligned to the horizons

    @property
    def failed_(self) -> bool:
        """Model produced no usable output for this fold, at either stage."""
        return bool(self.fit_failed_ or self.forecast_failed_)

    # -- subclass hooks -------------------------------------------------
    def _build_and_fit(self, y):
        raise NotImplementedError

    def _forecast_path(self, steps: int):
        """Return (mean, lo, hi) arrays of length `steps`."""
        raise NotImplementedError

    # -- harness interface ----------------------------------------------
    def fit(self, y):
        y = np.asarray(y, dtype=float)
        self.result_ = None
        self.fit_failed_ = False
        self.forecast_failed_ = False
        self.failure_reason_ = ""

        if np.isfinite(y).sum() < self.min_observations:
            self.fit_failed_ = True
            self.failure_reason_ = "insufficient observations (%d finite)" % int(
                np.isfinite(y).sum()
            )
            return self

        try:
            with warnings.catch_warnings():
                # Convergence chatter is captured in `failure_reason_` when it
                # actually matters; it must not flood the batch log.
                warnings.simplefilter("ignore")
                self.result_ = self._build_and_fit(y)
        except Exception as exc:  # noqa: BLE001 - contained by design
            self.fit_failed_ = True
            self.failure_reason_ = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return self

    def predict(self, horizons):
        horizons = [int(h) for h in np.atleast_1d(horizons)]
        n = len(horizons)
        if self.fit_failed_ or self.result_ is None:
            self.last_interval_ = (np.full(n, np.nan), np.full(n, np.nan))
            return np.full(n, np.nan)

        steps = max(horizons)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, lo, hi = self._forecast_path(steps)
        except Exception as exc:  # noqa: BLE001 - contained by design
            self.forecast_failed_ = True
            self.failure_reason_ = "forecast failed -- %s: %s" % (
                type(exc).__name__,
                str(exc)[:200],
            )
            self.last_interval_ = (np.full(n, np.nan), np.full(n, np.nan))
            return np.full(n, np.nan)

        idx = [h - 1 for h in horizons]  # h steps ahead == position h-1
        self.last_interval_ = (np.asarray(lo)[idx], np.asarray(hi)[idx])
        return np.asarray(mean, dtype=float)[idx]

    @property
    def min_observations(self) -> int:
        """Refuse to fit on a window too short for the specification."""
        return 2 * SEASONAL_PERIOD_M + 2


class SarimaxForecaster(StatisticalForecaster):
    """
    ARIMA/SARIMA via `SARIMAX`. Missing observations are passed through
    untouched -- the Kalman filter treats them as unobserved states, which is
    precisely why the addendum selected this class over `ARIMA`.
    """

    name = "sarima"

    def __init__(self, order, seasonal_order, alpha: float = NATIVE_CI_ALPHA):
        super().__init__(alpha=alpha)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)

    def _build_and_fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            # Left off so the optimiser is not forced onto the stationarity
            # boundary on short capped windows; the fitted parameters are
            # reported, and a diverging fit shows up as a huge error on the
            # fold rather than being silently constrained into plausibility.
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def _forecast_path(self, steps: int):
        fc = self.result_.get_forecast(steps=steps)
        ci = np.asarray(fc.conf_int(alpha=self.alpha))
        return np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1]

    @property
    def min_observations(self) -> int:
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        # Enough rows to survive the differencing and still estimate the terms.
        return d + D * m + max(3 * (p + q + P + Q), 2 * m) + 2


class ETSForecaster(StatisticalForecaster):
    """
    Exponential Smoothing via `ETSModel` (state space), `missing='drop'`.

    Dropping is applied to the ENDOG series, so a masked flow target is fitted
    on its observed values only. This compresses the effective spacing across a
    gap -- an accepted, documented consequence of the addendum's chosen policy,
    not an oversight. The alternative the addendum explicitly rejected (classic
    `ExponentialSmoothing`) returns an all-NaN forecast instead of a compressed
    but usable one.
    """

    name = "exponential_smoothing"

    def __init__(
        self,
        error="add",
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=SEASONAL_PERIOD_M,
        alpha: float = NATIVE_CI_ALPHA,
    ):
        super().__init__(alpha=alpha)
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def _build_and_fit(self, y):
        """
        The drop is performed EXPLICITLY here, before the model is constructed,
        and the resulting observed-only series is handed over on a clean
        contiguous RangeIndex.

        This is not a stylistic preference -- it is required to make the
        addendum's mandated policy actually work. Measured on statsmodels
        0.15.0:
          * numpy input          -> `.forecast()` works, but `.get_prediction()`
                                    raises AttributeError, so no intervals.
          * Series carrying NaN  -> `missing='drop'` leaves a non-contiguous
            with missing='drop'     index and even `.forecast()` raises
                                    "No supported index is available."
          * observed-only Series -> both work.

        Dropping ourselves yields the same estimation sample `missing='drop'`
        would produce (same observations, same order); `missing` is still passed
        through as a declaration of intent.

        DOCUMENTED CONSEQUENCE: dropping compresses the series across a gap, so
        the m=5 seasonal cycle is counted in OBSERVED positions rather than true
        period-positions. For the flow target that misaligns the seasonal index
        at each of its gap slots. This is inherent to the addendum's chosen ETS
        policy, not a defect introduced here -- and it is precisely why the
        addendum pairs ETS with SARIMAX, which absorbs missing observations
        natively through the Kalman filter and keeps the alignment intact.
        """
        import pandas as pd
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        y = np.asarray(y, dtype=float)
        observed = y[np.isfinite(y)]
        self.n_dropped_ = int(y.size - observed.size)
        series = pd.Series(observed, index=pd.RangeIndex(observed.size), dtype=float)

        return ETSModel(
            series,
            error=self.error,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods if self.seasonal else None,
            missing=ETS_MISSING_POLICY,
        ).fit(disp=False)

    def _forecast_path(self, steps: int):
        start = self.result_.nobs
        pred = self.result_.get_prediction(start=start, end=start + steps - 1)
        mean = np.asarray(pred.predicted_mean, dtype=float)
        try:
            ci = np.asarray(pred.pred_int(alpha=self.alpha), dtype=float)
            lo, hi = ci[:, 0], ci[:, 1]
        except Exception:  # noqa: BLE001
            # Interval machinery is unavailable for some specifications. The
            # point forecast still stands; the interval is reported as ABSENT
            # rather than fabricated from a normal approximation.
            lo = np.full(steps, np.nan)
            hi = np.full(steps, np.nan)
        return mean, lo, hi

    @property
    def min_observations(self) -> int:
        if self.seasonal:
            return 2 * self.seasonal_periods + 4
        return 6
