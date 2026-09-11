"""
comparison_matrix.py -- Complete the model comparison matrix (Day 13).

WHY THIS EXISTS
---------------
`forecasts/full_model_comparison.csv` holds seven models. The project evaluated
eight. The ensemble is built at Day 8, from forecasts the Day-7 run had already
produced, so it was never written into the Day-7 comparison artifact -- and yet
it is the numerical leader in at least one cell (see `models/model_registry.json`,
`numerical_leader`).

The addendum's Day 13 requirement is that the paper "reports the FULL comparison
matrix, not just the winning path". A matrix missing the candidate that led one of
the cells is precisely the omission that requirement is aimed at. So this module
scores the ensemble and appends it, producing `forecasts/comparison_matrix.csv`.

WHAT IT DOES NOT DO
-------------------
It does not recompute anything for the other seven models. Their rows are carried
across unchanged, and a test asserts that byte-for-byte.

That matters because of how common support works. If the ensemble were folded into
a fresh `common_support_mask` over eight models, any point where the ensemble
abstains would drop out of the comparison for *every* model, shifting all seven
sets of numbers -- and the paper would then disagree with the dashboard and the
registry for no reason a reader could see.

Instead the ensemble is scored on the support set the existing seven were scored
on: the points flagged `common_support` in `ml_predictions.csv`, which is the
full seven-model pool. If the ensemble cannot produce a forecast at some of those
points, its `n_scored` comes out lower and the table shows that, which is the
honest presentation -- the comparison is restricted, never silently rebased.

Run:  python -m src.reporting.comparison_matrix
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    ENSEMBLE_PREDICTIONS_PATH, FORECASTS_DIR, FULL_COMPARISON_PATH,
    ML_PREDICTIONS_PATH, PROJECT_ROOT, TRAINING_CAP_DATE,
)
from src.evaluation.walk_forward import aggregate_metrics  # noqa: E402

COMPLETE_COMPARISON_PATH = FORECASTS_DIR / "comparison_matrix.csv"

# The keys that identify one scored test point, matching common_support_mask.
POINT_KEYS = ["target", "window_rule", "fold_id", "horizon"]

__all__ = ["build_complete_matrix", "COMPLETE_COMPARISON_PATH"]


def _support_points(seven_model_preds: pd.DataFrame) -> pd.DataFrame:
    """The test points the existing seven-model comparison was scored on."""
    supported = seven_model_preds[seven_model_preds["common_support"].astype(bool)]
    return supported[POINT_KEYS].drop_duplicates()


def build_complete_matrix() -> pd.DataFrame:
    existing = pd.read_csv(FULL_COMPARISON_PATH)
    seven = pd.read_csv(ML_PREDICTIONS_PATH)
    ensemble = pd.read_csv(ENSEMBLE_PREDICTIONS_PATH)

    # Restrict the ensemble to the existing support set, then adopt that set as
    # its common_support flag -- deliberately overriding the ensemble file's own
    # single-model flag, which was computed against nothing but itself.
    points = _support_points(seven)
    ensemble = ensemble.merge(points.assign(_supported=True), on=POINT_KEYS, how="left")
    # notna() rather than fillna(False): an unmatched left join leaves NaN in an
    # object column, and fillna-then-astype(bool) is a deprecated downcast path.
    ensemble["common_support"] = ensemble["_supported"].notna()
    ensemble = ensemble.drop(columns=["_supported"])

    post = ensemble[ensemble["origin_post_cutoff"].astype(bool)]
    scopes = {
        "all_dev_folds": ensemble,
        "common_support": ensemble[ensemble["common_support"]],
        "post_cutoff_origins": post,
        "post_cutoff_common_support": post[post["common_support"]],
    }
    rows = pd.concat(
        [aggregate_metrics(frame).assign(fold_scope=name)
         for name, frame in scopes.items()],
        ignore_index=True,
    )
    rows = rows[["fold_scope"] + [c for c in rows.columns if c != "fold_scope"]]

    complete = pd.concat([existing, rows], ignore_index=True)
    complete = complete[existing.columns.tolist()]
    COMPLETE_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(COMPLETE_COMPARISON_PATH, index=False)
    return complete


if __name__ == "__main__":
    matrix = build_complete_matrix()
    print("Wrote %s" % COMPLETE_COMPARISON_PATH.relative_to(PROJECT_ROOT))
    print("  models : %d  (%s)" % (matrix["model"].nunique(),
                                   ", ".join(sorted(matrix["model"].unique()))))
    print("  rows   : %d" % len(matrix))
    governing = matrix[(matrix["fold_scope"] == "post_cutoff_common_support")
                       & (matrix["window_rule"] == "capped")]
    print("\nEnsemble on the governing scope (training cap %s):" % TRAINING_CAP_DATE)
    print(governing[governing["model"] == "ensemble"][
        ["target", "horizon", "n_scored", "MAE", "RMSE", "MASE"]].to_string(index=False))
