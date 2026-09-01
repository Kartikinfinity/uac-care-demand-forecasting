"""
ml.py -- Random Forest and Gradient Boosting forecasters (Day 7).

DIRECT MULTI-HORIZON FITTING (addendum Section 2):
    "Random Forest and Gradient Boosting use direct multi-horizon fitting
     (3 models per target)."

So unlike SARIMAX/ETSModel -- which fit ONCE and read h=1/7/14 off a single
recursive forecast path -- these families fit a SEPARATE estimator per horizon.
Each one learns the mapping

        X[t]  ->  y[t + lead]

directly, so a 14-period-ahead forecast never compounds 14 one-step errors.

Training-pair construction is where a direct multi-horizon model leaks if you
are careless. For a fold whose training window is [start .. cutoff], the pairs
are:

        { (X[t], y[t + lead])  :  start <= t  and  t + lead <= cutoff }

The `t + lead <= cutoff` half is the load-bearing constraint. Without it a pair
would carry a LABEL from after the training cutoff -- the model would be trained
on the future it is about to be asked to predict. Note this costs `lead` rows
off the end of every window, so h=14 always trains on fewer pairs than h=1.

The forecast itself uses X[cutoff] -- the last feature row inside the window,
which by construction is built only from data at or before the cutoff.

Missing-data policy differs BY DESIGN between the two families (addendum
Sections 2 and 4):

  RandomForestRegressor        -- feature rows containing NaN are DROPPED, and
                                  the count is logged.
  HistGradientBoostingRegressor -- keeps them; it handles NaN natively.

Recorded honestly: scikit-learn gained native NaN support for tree ensembles in
1.4, so on the installed 1.9.0 RandomForestRegressor no longer *requires* the
drop. The addendum was written when it did. The rule is retained as the
governing methodological choice rather than quietly dropped, which also
preserves the deliberate contrast between the two families.

Neither family has native prediction intervals. Per addendum Section 6 the
intervals are EMPIRICAL residual quantiles built at Day 9 from out-of-sample
walk-forward residuals; Day 7's job is to persist those residuals, not to build
the intervals here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    HIST_GRADIENT_BOOSTING_PARAMS,
    ML_MIN_TRAINING_PAIRS,
    RANDOM_FOREST_ABSTAINS_ON_NAN_PREDICTION_ROW,
    RANDOM_FOREST_DROPS_NAN_ROWS,
    RANDOM_FOREST_PARAMS,
)

__all__ = [
    "DirectMultiHorizonForecaster",
    "RandomForestForecaster",
    "HistGradientBoostingForecaster",
    "select_feature_columns",
]


def select_feature_columns(features: pd.DataFrame, exclude: list) -> list:
    """
    The engineered predictors only: lags, rolling statistics, calendar effects.

    Raw contemporaneous columns are excluded because the roadmap (Part 3) is
    explicit that the upstream series are "usable only in lagged/rolling form",
    and flags same-row use as the high-leakage failure mode. `lag_1_*` already
    carries the most recent level, so nothing informative is lost.

    `rolling_*_nobs_*` columns ARE included: they record how many real
    observations back a rolling statistic, which is genuine information about
    data quality at that row rather than a leak.
    """
    keep = []
    for col in features.columns:
        if col in exclude:
            continue
        if col.startswith(("lag_", "rolling_")):
            keep.append(col)
        elif col in ("day_of_week", "month", "is_near_holiday"):
            keep.append(col)
    return keep


class DirectMultiHorizonForecaster:
    """
    Shared plumbing for the two ML families.

    `.fit(y, X)` stores the aligned training window; `.predict(leads)` fits ONE
    estimator per requested lead and returns one forecast each. Estimation is
    deferred to `predict` because the required leads are a property of the fold
    (they exceed the nominal horizon when the window was pulled back off an
    interpolated origin), and the addendum's "3 models per target" contract is
    about one estimator per horizon -- which is exactly what this produces.
    """

    name = "ml"
    requires_features = True
    drops_nan_rows = False
    abstains_on_nan_prediction_row = False

    def __init__(self, feature_columns=None, min_pairs: int = ML_MIN_TRAINING_PAIRS):
        self.feature_columns = feature_columns
        self.min_pairs = min_pairs
        self.models_ = {}
        self.fit_failed_ = False
        self.forecast_failed_ = False
        self.failure_reason_ = ""
        self.n_rows_dropped_ = 0
        self.n_pairs_ = {}
        self.last_interval_ = None  # no native intervals; Day 9 builds them
        self._y = None
        self._X = None

    @property
    def failed_(self) -> bool:
        return bool(self.fit_failed_ or self.forecast_failed_)

    def _make_estimator(self):
        raise NotImplementedError

    def fit(self, y, X=None):
        if X is None:
            raise ValueError("%s requires a feature matrix" % type(self).__name__)
        self.models_ = {}
        self.fit_failed_ = False
        self.forecast_failed_ = False
        self.failure_reason_ = ""
        self.n_rows_dropped_ = 0
        self.n_pairs_ = {}

        y = np.asarray(y, dtype=float)
        cols = self.feature_columns or list(X.columns)
        X = X[cols]
        if len(X) != len(y):
            raise ValueError("feature/target length mismatch: %d vs %d" % (len(X), len(y)))
        self._y = y
        self._X = X.to_numpy(dtype=float)
        return self

    def _training_pairs(self, lead: int):
        """
        (X[t], y[t+lead]) pairs whose LABEL lands at or before the training
        cutoff. Anything else would train on the future.
        """
        n = len(self._y)
        last_t = n - 1 - lead  # inclusive; t+lead must stay inside the window
        if last_t < 0:
            return None, None, 0
        X = self._X[: last_t + 1]
        target = self._y[lead : lead + last_t + 1]

        usable = np.isfinite(target)
        if self.drops_nan_rows:
            usable &= np.isfinite(X).all(axis=1)
        dropped = int((~usable).sum())
        return X[usable], target[usable], dropped

    def predict(self, horizons):
        leads = [int(h) for h in np.atleast_1d(horizons)]
        out = np.full(len(leads), np.nan)
        if self.fit_failed_ or self._y is None:
            return out

        reasons = []
        for i, lead in enumerate(leads):
            X_tr, y_tr, dropped = self._training_pairs(lead)
            self.n_rows_dropped_ += dropped
            if X_tr is None or len(y_tr) < self.min_pairs:
                reasons.append("lead %d: only %d usable pairs (min %d)"
                               % (lead, 0 if y_tr is None else len(y_tr), self.min_pairs))
                self.n_pairs_[lead] = 0 if y_tr is None else int(len(y_tr))
                continue
            self.n_pairs_[lead] = int(len(y_tr))

            X_pred = self._X[-1:]  # the feature row AT the training cutoff
            if self.abstains_on_nan_prediction_row and not np.isfinite(X_pred).all():
                reasons.append("lead %d: prediction row contains NaN and this "
                               "family drops NaN rows" % lead)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self._make_estimator().fit(X_tr, y_tr)
                    out[i] = float(model.predict(X_pred)[0])
                self.models_[lead] = model
            except Exception as exc:  # noqa: BLE001 - contained by design
                reasons.append("lead %d: %s: %s" % (lead, type(exc).__name__, str(exc)[:120]))

        if reasons:
            self.forecast_failed_ = not np.isfinite(out).any()
            self.failure_reason_ = "; ".join(reasons)[:400]
        return out

    def feature_importances(self, lead: int):
        model = self.models_.get(lead)
        if model is None:
            return None
        return getattr(model, "feature_importances_", None)


class RandomForestForecaster(DirectMultiHorizonForecaster):
    """`RandomForestRegressor`, NaN-affected feature rows dropped and counted."""

    name = "random_forest"
    drops_nan_rows = RANDOM_FOREST_DROPS_NAN_ROWS
    abstains_on_nan_prediction_row = RANDOM_FOREST_ABSTAINS_ON_NAN_PREDICTION_ROW

    def __init__(self, feature_columns=None, params: dict = None, **kwargs):
        super().__init__(feature_columns=feature_columns, **kwargs)
        self.params = dict(params or RANDOM_FOREST_PARAMS)

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**self.params)


class HistGradientBoostingForecaster(DirectMultiHorizonForecaster):
    """
    `HistGradientBoostingRegressor` -- the histogram class, NOT the classic
    `GradientBoostingRegressor`. The addendum selects it for native NaN support
    at zero new dependencies, so its feature rows are kept intact.
    """

    name = "gradient_boosting"
    drops_nan_rows = False

    def __init__(self, feature_columns=None, params: dict = None, **kwargs):
        super().__init__(feature_columns=feature_columns, **kwargs)
        self.params = dict(params or HIST_GRADIENT_BOOSTING_PARAMS)

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(**self.params)
