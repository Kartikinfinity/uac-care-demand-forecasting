"""
tests/test_ml.py -- Day 7 machine-learning models.

The load-bearing tests here protect the direct-multi-horizon training-pair
construction, which is where this family leaks if you are careless: a pair whose
LABEL falls after the training cutoff would train the model on the future it is
about to predict.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    COL_DATE,
    DATA_PROCESSED_DIR,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    MASTER_SERIES_PATH,
    ML_MIN_TRAINING_PAIRS,
    NUMERIC_COLS,
    RANDOM_FOREST_DROPS_NAN_ROWS,
    RANDOM_FOREST_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    RANDOM_SEED,
    TARGET_1,
    TARGET_2,
)
from src.models.ml import (
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    select_feature_columns,
)
from src.evaluation.walk_forward import (
    generate_folds,
    resolve_split_boundaries,
    run_walk_forward,
)
from src.evaluation.run_baselines import resolve_cutoff_pos


@pytest.fixture(scope="module")
def master():
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    return df


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(DATA_PROCESSED_DIR / "features_target1.parquet")


@pytest.fixture(scope="module")
def columns(features):
    return select_feature_columns(
        features, exclude=[COL_DATE, "parsed_date", "is_imputed"] + NUMERIC_COLS + ["net_flow"]
    )


@pytest.fixture(scope="module")
def dev_end(master):
    _, end = resolve_split_boundaries(master["is_imputed"], FINAL_TEST_WINDOW)
    return end


# ----------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------
def test_raw_contemporaneous_columns_are_excluded_from_the_ml_feature_set(columns):
    """
    Roadmap Part 3: the upstream series are "usable only in lagged/rolling form",
    and same-row use is flagged as the high-leakage failure mode.
    """
    for raw in NUMERIC_COLS + ["net_flow", "parsed_date", COL_DATE, "is_imputed"]:
        assert raw not in columns


def test_feature_set_contains_the_documented_families(columns):
    assert any(c.startswith("lag_") for c in columns)
    assert any(c.startswith("rolling_") and "_mean_" in c for c in columns)
    assert any(c.startswith("rolling_") and "_var_" in c for c in columns)
    assert {"day_of_week", "month", "is_near_holiday"} <= set(columns)


def test_features_align_row_for_row_with_the_master_series(master, features):
    assert len(features) == len(master)
    pd.testing.assert_series_equal(
        pd.to_datetime(features["parsed_date"]), master["parsed_date"], check_names=False
    )


# ----------------------------------------------------------------------
# THE LEAKAGE-CRITICAL CONTRACT: training-pair construction
# ----------------------------------------------------------------------
@pytest.mark.parametrize("lead", [1, 7, 14])
def test_no_training_pair_carries_a_label_from_after_the_cutoff(lead):
    """
    A pair is (X[t], y[t+lead]). Every label must land at or before the training
    cutoff -- otherwise the model trains on the future it will be asked about.
    """
    n = 100
    y = np.arange(n, dtype=float)          # y[i] == i, so a label reveals its index
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    model = HistGradientBoostingForecaster(feature_columns=["f"]).fit(y, X)

    X_tr, y_tr, _ = model._training_pairs(lead)
    assert len(y_tr) == n - lead
    assert y_tr.max() <= n - 1, "a label came from beyond the training window"
    # Each label is exactly `lead` ahead of its feature row.
    np.testing.assert_allclose(y_tr, X_tr[:, 0] + lead)


def test_longer_horizons_train_on_strictly_fewer_pairs():
    """Direct multi-horizon costs `lead` rows off the end of every window."""
    n = 100
    y = np.arange(n, dtype=float)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    model = HistGradientBoostingForecaster(feature_columns=["f"]).fit(y, X)
    counts = [len(model._training_pairs(lead)[1]) for lead in (1, 7, 14)]
    assert counts[0] > counts[1] > counts[2]


def test_prediction_uses_the_feature_row_at_the_training_cutoff():
    """The forecast origin row must be the LAST row of the training window."""
    n = 80
    y = np.arange(n, dtype=float)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    model = HistGradientBoostingForecaster(feature_columns=["f"]).fit(y, X)
    np.testing.assert_allclose(model._X[-1], [n - 1])


def test_appending_future_rows_cannot_change_an_earlier_forecast(master, features, columns, dev_end):
    """
    The end-to-end property: a historical fold's ML forecast must be identical
    whether or not later data exists in the frame.
    """
    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    fold = folds[45]
    cutoff = resolve_cutoff_pos(master["parsed_date"])
    factories = {"gb": lambda: HistGradientBoostingForecaster(feature_columns=columns)}

    full = run_walk_forward(master, TARGET_1, factories, [fold], cutoff,
                            features=features, window_rules=["full"])
    truncated_df = master.iloc[: fold.max_test_pos + 1].copy()
    truncated_X = features.iloc[: fold.max_test_pos + 1].copy()
    short = run_walk_forward(truncated_df, TARGET_1, factories, [fold], cutoff,
                             features=truncated_X, window_rules=["full"])

    np.testing.assert_allclose(
        full.sort_values("horizon")["y_pred"].to_numpy(),
        short.sort_values("horizon")["y_pred"].to_numpy(),
    )


def test_harness_slices_features_with_the_same_bounds_as_the_target(master, features, columns, dev_end):
    """The feature matrix must inherit every leakage bound the target has."""
    seen = {}

    class Spy(HistGradientBoostingForecaster):
        def fit(self, y, X=None):
            seen["n_y"] = len(y)
            seen["n_X"] = len(X)
            seen["last_date"] = pd.to_datetime(X["parsed_date"].iloc[-1])
            return super().fit(y, X)

    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    fold = folds[30]
    run_walk_forward(master, TARGET_1, {"spy": lambda: Spy(feature_columns=columns)},
                     [fold], resolve_cutoff_pos(master["parsed_date"]),
                     features=features, window_rules=["full"])

    assert seen["n_y"] == seen["n_X"] == fold.train_cutoff_pos + 1
    assert seen["last_date"] == master["parsed_date"].iloc[fold.train_cutoff_pos]


def test_harness_rejects_a_misaligned_feature_matrix(master, features, columns, dev_end):
    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    with pytest.raises(ValueError, match="align row-for-row"):
        run_walk_forward(master, TARGET_1,
                         {"gb": lambda: HistGradientBoostingForecaster(feature_columns=columns)},
                         [folds[10]], resolve_cutoff_pos(master["parsed_date"]),
                         features=features.iloc[:-5], window_rules=["full"])


def test_a_feature_requiring_model_errors_if_no_features_are_supplied(master, columns, dev_end):
    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    with pytest.raises(ValueError, match="requires a feature matrix"):
        run_walk_forward(master, TARGET_1,
                         {"gb": lambda: HistGradientBoostingForecaster(feature_columns=columns)},
                         [folds[10]], resolve_cutoff_pos(master["parsed_date"]),
                         window_rules=["full"])


# ----------------------------------------------------------------------
# Direct multi-horizon: one estimator per horizon
# ----------------------------------------------------------------------
def test_one_estimator_is_fitted_per_horizon(master, features, columns, dev_end):
    """Addendum: "direct multi-horizon fitting (3 models per target)"."""
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    model = HistGradientBoostingForecaster(feature_columns=columns).fit(
        y, features.iloc[: dev_end + 1]
    )
    model.predict(FORECAST_HORIZONS)
    assert sorted(model.models_) == sorted(FORECAST_HORIZONS)
    assert len({id(m) for m in model.models_.values()}) == len(FORECAST_HORIZONS)


def test_each_horizon_gets_a_different_fitted_estimator(master, features, columns, dev_end):
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    model = RandomForestForecaster(feature_columns=columns).fit(y, features.iloc[: dev_end + 1])
    preds = model.predict(FORECAST_HORIZONS)
    assert np.isfinite(preds).all()
    assert len(set(np.round(preds, 6))) > 1, "all horizons produced the same number"


# ----------------------------------------------------------------------
# Missing-data policy (asymmetric BY DESIGN, addendum Sections 2 and 4)
# ----------------------------------------------------------------------
def test_random_forest_drops_nan_feature_rows_and_counts_them():
    n = 200
    y = np.arange(n, dtype=float)
    f = np.arange(n, dtype=float)
    f[10:40] = np.nan
    X = pd.DataFrame({"f": f})

    rf = RandomForestForecaster(feature_columns=["f"]).fit(y, X)
    assert rf.drops_nan_rows is RANDOM_FOREST_DROPS_NAN_ROWS is True
    _, y_rf, dropped_rf = rf._training_pairs(1)
    assert dropped_rf == 30
    assert len(y_rf) == n - 1 - 30


def test_gradient_boosting_keeps_nan_rows():
    """HistGradientBoostingRegressor was chosen for native NaN support."""
    n = 200
    y = np.arange(n, dtype=float)
    f = np.arange(n, dtype=float)
    f[10:40] = np.nan
    X = pd.DataFrame({"f": f})

    gb = HistGradientBoostingForecaster(feature_columns=["f"]).fit(y, X)
    assert gb.drops_nan_rows is False
    _, y_gb, dropped_gb = gb._training_pairs(1)
    assert dropped_gb == 0
    assert len(y_gb) == n - 1


def test_a_missing_label_is_dropped_by_both_families():
    """Invariant 2: a true-missing flow value is never used as a training label."""
    n = 200
    y = np.arange(n, dtype=float)
    y[50:60] = np.nan
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    for model in (RandomForestForecaster(feature_columns=["f"]),
                  HistGradientBoostingForecaster(feature_columns=["f"])):
        model.fit(y, X)
        _, labels, _ = model._training_pairs(1)
        assert np.isfinite(labels).all()


def test_gradient_boosting_really_accepts_nan_at_fit_time():
    """Pins the library behaviour the addendum's choice depends on."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = HistGradientBoostingRegressor(max_iter=5).fit(X, y)
    assert np.isfinite(model.predict(X)).all()


# ----------------------------------------------------------------------
# Abstention and determinism
# ----------------------------------------------------------------------
def test_too_few_training_pairs_abstains_rather_than_fitting():
    n = ML_MIN_TRAINING_PAIRS + 2
    y = np.arange(n, dtype=float)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    model = HistGradientBoostingForecaster(feature_columns=["f"]).fit(y, X)
    preds = model.predict([14])
    assert np.isnan(preds).all()
    assert "usable pairs" in model.failure_reason_


def test_an_abstention_is_nan_never_a_fallback_number():
    y = np.arange(5, dtype=float)
    X = pd.DataFrame({"f": np.arange(5, dtype=float)})
    model = RandomForestForecaster(feature_columns=["f"]).fit(y, X)
    preds = model.predict(FORECAST_HORIZONS)
    assert np.isnan(preds).all()
    assert not np.any(preds == 0)


def test_fits_are_reproducible_under_the_project_seed(master, features, columns, dev_end):
    """Addendum Section 3: one global RANDOM_SEED on every stochastic fit."""
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    X = features.iloc[: dev_end + 1]
    assert RANDOM_FOREST_PARAMS["random_state"] == RANDOM_SEED
    assert HIST_GRADIENT_BOOSTING_PARAMS["random_state"] == RANDOM_SEED

    first = RandomForestForecaster(feature_columns=columns).fit(y, X).predict(FORECAST_HORIZONS)
    second = RandomForestForecaster(feature_columns=columns).fit(y, X).predict(FORECAST_HORIZONS)
    np.testing.assert_allclose(first, second)


def test_early_stopping_is_disabled_so_no_random_split_of_a_time_series_occurs():
    assert HIST_GRADIENT_BOOSTING_PARAMS["early_stopping"] is False


# ----------------------------------------------------------------------
# Residual persistence (addendum Day 7 -> Day 9 input)
# ----------------------------------------------------------------------
def test_residuals_are_only_built_from_scorable_points(master, features, columns, dev_end):
    from src.evaluation.run_ml import build_residuals

    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    preds = run_walk_forward(
        master, TARGET_2,
        {"gb": lambda: HistGradientBoostingForecaster(feature_columns=columns)},
        folds[:20], resolve_cutoff_pos(master["parsed_date"]),
        features=features, window_rules=["full"],
    )
    residuals = build_residuals(preds)
    assert len(residuals) > 0
    assert residuals["y_true"].notna().all()
    assert residuals["y_pred"].notna().all()
    np.testing.assert_allclose(
        residuals["residual"].to_numpy(),
        (residuals["y_true"] - residuals["y_pred"]).to_numpy(),
    )
    assert "origin_date" in residuals.columns
    assert "origin_post_cutoff" in residuals.columns
