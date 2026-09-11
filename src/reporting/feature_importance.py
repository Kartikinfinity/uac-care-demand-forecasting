"""
feature_importance.py -- Extract tree-ensemble feature importances (Day 13).

Reads the ML models persisted at Day 7 and writes a flat artifact, so the paper
can describe what the tree models keyed on without loading a pickle. Kept
separate from the fitting code because it is a REPORTING concern: nothing in the
forecasting pipeline consumes it.

THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS
----------------------------------------------
Addendum Day 13: "caveats any RF/GBR feature importances for collinearity."

The feature set is built from lags and rolling windows of the same five series,
so `lag_1_x`, `lag_7_x` and `rolling_7_mean_x` are strongly correlated with each
other by construction. Impurity-based importance splits credit arbitrarily among
correlated predictors: whichever one a tree happens to split on first absorbs the
gain, and its near-duplicates look unimportant. Two runs on slightly different
data can therefore reorder them without anything meaningful having changed.

These values indicate roughly which SERIES the model used. They do not support a
claim that one particular lag matters more than another, and they are not
evidence of a causal driver. The paper states that alongside the table rather
than in a footnote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    FEATURE_IMPORTANCE_PATH,
    MODELS_DIR,
)

__all__ = ["extract_importances", "write_importances"]

COLLINEARITY_CAVEAT = (
    "Impurity-based importance distributes credit arbitrarily among correlated "
    "predictors. The feature set is lags and rolling windows of the same five "
    "series, so these values indicate which SERIES the model used, not which "
    "specific lag matters most, and they are not evidence of a causal driver."
)


def extract_importances() -> pd.DataFrame:
    """
    One row per (target, family, horizon, feature) with its importance.

    Only Random Forest exposes `feature_importances_`;
    HistGradientBoostingRegressor does not, so it is reported as unavailable
    rather than substituted with a different measure that would not be
    comparable.
    """
    import joblib

    rows = []
    for path in sorted(MODELS_DIR.glob("ml_*.pkl")):
        blob = joblib.load(path)
        estimator = blob.get("estimator")
        columns = blob.get("feature_columns") or []
        if estimator is None:
            continue
        values = getattr(estimator, "feature_importances_", None)
        if values is None or len(values) != len(columns):
            continue
        for feature, importance in zip(columns, np.asarray(values, dtype=float)):
            rows.append({
                "target": blob["target"],
                "family": blob["family"],
                "horizon": int(blob["horizon"]),
                "feature": feature,
                "importance": float(importance),
                "n_training_pairs": blob.get("n_training_pairs"),
            })
    return pd.DataFrame(rows)


def write_importances(path: Path = FEATURE_IMPORTANCE_PATH) -> pd.DataFrame:
    frame = extract_importances()
    if len(frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return frame


def top_features(frame: pd.DataFrame, target: str, horizon: int, n: int = 8):
    sub = frame[(frame["target"] == target) & (frame["horizon"] == horizon)]
    return sub.sort_values("importance", ascending=False).head(n)


if __name__ == "__main__":
    result = write_importances()
    if not len(result):
        print("No tree-ensemble importances available. Run "
              "`python -m src.evaluation.run_ml` first.")
    else:
        print("Wrote %s (%d rows, %d model/horizon combinations)"
              % (FEATURE_IMPORTANCE_PATH.name, len(result),
                 result.groupby(["target", "family", "horizon"]).ngroups))
        print("\nCAVEAT: " + COLLINEARITY_CAVEAT)
