"""
tests/test_statistical.py -- Day 6 statistical models.

The load-bearing test here is the addendum's explicitly required one
(Section 3): "SARIMAX/ETSModel missing-data behavior (regression test against
the exact silent-failure mode found)". Everything else protects the
single-fit-per-target contract, the harness interface, and the containment of
fit failures.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    ETS_MISSING_POLICY,
    ETS_SPECS,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    MASTER_SERIES_PATH,
    NATIVE_CI_ALPHA,
    SARIMA_ORDERS,
    SEASONAL_PERIOD_M,
    TARGET_1,
    TARGET_2,
)
from src.models.statistical import ETSForecaster, SarimaxForecaster
from src.evaluation.walk_forward import generate_folds, resolve_split_boundaries


@pytest.fixture(scope="module")
def master():
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    return df


@pytest.fixture(scope="module")
def dev_end(master):
    _, end = resolve_split_boundaries(master["is_imputed"], FINAL_TEST_WINDOW)
    return end


@pytest.fixture(scope="module")
def masked_flow(master, dev_end):
    """The real masked flow series -- the exact input the addendum warns about."""
    return master[TARGET_2].astype(float).to_numpy()[: dev_end + 1]


# ----------------------------------------------------------------------
# THE REQUIRED REGRESSION TEST (addendum Section 3)
# ----------------------------------------------------------------------
def test_classic_exponential_smoothing_still_fails_silently_on_nan_input(masked_flow):
    """
    Pins the EXACT silent-failure mode the addendum documents as its reason for
    mandating ETSModel over the classic class:

        "confirmed necessary: classic `ExponentialSmoothing` silently returns
         an all-NaN forecast on NaN input"

    It does not raise. It returns NaNs. A pipeline that trusted it would publish
    empty forecasts without a single error. If a future statsmodels release ever
    fixes this, this test fails and the choice can be revisited deliberately
    rather than by accident.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    assert np.isnan(masked_flow).sum() > 0, "fixture no longer contains missing values"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = ExponentialSmoothing(
            masked_flow, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD_M
        ).fit().forecast(max(FORECAST_HORIZONS))

    assert np.all(np.isnan(np.asarray(forecast, dtype=float))), (
        "classic ExponentialSmoothing no longer fails silently -- revisit the "
        "addendum's library choice deliberately"
    )


def test_our_ets_path_produces_real_forecasts_on_the_same_masked_input(masked_flow):
    """The other half of the regression: our configured path must NOT be all-NaN."""
    model = ETSForecaster(**ETS_SPECS[TARGET_2]).fit(masked_flow)
    preds = model.predict(FORECAST_HORIZONS)

    assert not model.failed_, model.failure_reason_
    assert np.isfinite(preds).all(), "ETS path regressed to producing NaN forecasts"
    assert model.n_dropped_ == int(np.isnan(masked_flow).sum())


def test_ets_missing_policy_is_the_one_the_addendum_mandates():
    assert ETS_MISSING_POLICY == "drop"


def test_sarimax_absorbs_missing_values_without_dropping_them(masked_flow):
    """
    SARIMAX was chosen for native missing-data handling. It must accept the
    masked series as-is -- no imputation, no zero-fill, no dropping.
    """
    model = SarimaxForecaster(**SARIMA_ORDERS[TARGET_2]).fit(masked_flow)
    preds = model.predict(FORECAST_HORIZONS)

    assert not model.failed_, model.failure_reason_
    assert np.isfinite(preds).all()
    # The Kalman filter keeps every period-position; nothing was removed.
    assert model.result_.nobs == masked_flow.size


# ----------------------------------------------------------------------
# Single-fit-per-target contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory", [
    lambda: SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]),
    lambda: ETSForecaster(**ETS_SPECS[TARGET_1]),
])
def test_all_horizons_come_from_one_fitted_model(master, dev_end, factory):
    """
    Addendum Section 2: these families are "fit once per target; forecasts at
    h=1/7/14 come from that single model's native multi-step forecast function".
    Calling predict twice must not refit, and must be identical.
    """
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    model = factory().fit(y)
    first = model.predict(FORECAST_HORIZONS)
    fitted = model.result_
    second = model.predict(FORECAST_HORIZONS)

    assert model.result_ is fitted, "predict() refitted the model"
    np.testing.assert_allclose(first, second)


def test_horizons_are_read_off_the_multi_step_path_at_the_right_positions(master, dev_end):
    """h steps ahead must be position h-1 of the forecast path, not h."""
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    model = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(y)
    path = np.asarray(model.result_.get_forecast(steps=max(FORECAST_HORIZONS)).predicted_mean)
    preds = model.predict(FORECAST_HORIZONS)
    for i, h in enumerate(FORECAST_HORIZONS):
        assert preds[i] == pytest.approx(path[h - 1])


def test_predict_order_follows_the_requested_horizon_order(master, dev_end):
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    model = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(y)
    forward = model.predict([1, 7, 14])
    reverse = model.predict([14, 7, 1])
    np.testing.assert_allclose(forward, reverse[::-1])


# ----------------------------------------------------------------------
# Failure containment (addendum Section 3: log and continue)
# ----------------------------------------------------------------------
def test_a_window_too_short_to_fit_abstains_rather_than_raising():
    model = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(np.arange(5, dtype=float))
    preds = model.predict(FORECAST_HORIZONS)

    assert model.fit_failed_ is True
    assert "insufficient observations" in model.failure_reason_
    assert np.isnan(preds).all()


def test_an_all_missing_window_abstains_rather_than_raising():
    model = ETSForecaster(**ETS_SPECS[TARGET_2]).fit(np.full(100, np.nan))
    preds = model.predict(FORECAST_HORIZONS)

    assert model.failed_ is True
    assert np.isnan(preds).all()


def test_a_failed_model_never_substitutes_a_fallback_number():
    """An abstention must be NaN -- never a zero, a mean, or a last value."""
    model = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(np.arange(4, dtype=float))
    preds = model.predict(FORECAST_HORIZONS)
    assert np.isnan(preds).all()
    assert not np.any(preds == 0)


def test_fit_and_forecast_failures_are_tracked_separately(master, dev_end):
    """
    The report distinguishes "the model could not be estimated" from "it was
    estimated but could not forecast". Conflating them would misreport
    reliability.
    """
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    healthy = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(y)
    healthy.predict(FORECAST_HORIZONS)
    assert healthy.fit_failed_ is False
    assert healthy.forecast_failed_ is False
    assert healthy.failed_ is False

    broken = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]).fit(np.arange(3, dtype=float))
    assert broken.fit_failed_ is True and broken.failed_ is True


# ----------------------------------------------------------------------
# Native intervals (secondary diagnostic only)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("target", [TARGET_1, TARGET_2])
def test_native_intervals_are_produced_and_ordered(master, dev_end, target):
    y = master[target].astype(float).to_numpy()[: dev_end + 1]
    for model in (SarimaxForecaster(**SARIMA_ORDERS[target]),
                  ETSForecaster(**ETS_SPECS[target])):
        model.fit(y)
        preds = model.predict(FORECAST_HORIZONS)
        lo, hi = model.last_interval_
        assert np.isfinite(lo).all() and np.isfinite(hi).all(), model.name
        assert (lo <= preds).all() and (preds <= hi).all(), model.name
        assert (np.diff(hi - lo) >= 0).all(), "intervals must widen with horizon"


def test_interval_width_responds_to_the_nominal_level(master, dev_end):
    y = master[TARGET_1].astype(float).to_numpy()[: dev_end + 1]
    tight = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1], alpha=0.5).fit(y)
    tight.predict(FORECAST_HORIZONS)
    wide = SarimaxForecaster(**SARIMA_ORDERS[TARGET_1], alpha=NATIVE_CI_ALPHA).fit(y)
    wide.predict(FORECAST_HORIZONS)
    tl, th = tight.last_interval_
    wl, wh = wide.last_interval_
    assert ((wh - wl) > (th - tl)).all(), "a 95% interval must exceed a 50% one"


# ----------------------------------------------------------------------
# Harness compatibility and leakage
# ----------------------------------------------------------------------
def test_forecasters_satisfy_the_harness_interface():
    for model in (SarimaxForecaster(**SARIMA_ORDERS[TARGET_1]),
                  ETSForecaster(**ETS_SPECS[TARGET_1])):
        assert callable(model.fit) and callable(model.predict)


def test_statistical_models_never_see_data_past_the_training_cutoff(master, dev_end):
    """
    Same leakage guarantee the baselines get: the array handed to `.fit` is
    exactly y[:train_cutoff+1], and appending later data cannot change a
    historical fold's forecast.
    """
    from src.evaluation.run_baselines import resolve_cutoff_pos
    from src.evaluation.walk_forward import run_walk_forward

    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end)
    fold = folds[40]
    cutoff = resolve_cutoff_pos(master["parsed_date"])
    factories = {"sarima": lambda: SarimaxForecaster(**SARIMA_ORDERS[TARGET_1])}

    full = run_walk_forward(master, TARGET_1, factories, [fold], cutoff, window_rules=["full"])
    truncated = master.iloc[: fold.max_test_pos + 1].copy()
    short = run_walk_forward(truncated, TARGET_1, factories, [fold], cutoff, window_rules=["full"])

    np.testing.assert_allclose(
        full.sort_values("horizon")["y_pred"].to_numpy(),
        short.sort_values("horizon")["y_pred"].to_numpy(),
        rtol=1e-9,
    )


def test_specifications_match_the_day3_evidence():
    """d=1 and m=5 were fixed at Day 3; D=1/Q=1 came from the seasonal ACF."""
    for target in (TARGET_1, TARGET_2):
        order = SARIMA_ORDERS[target]["order"]
        seasonal = SARIMA_ORDERS[target]["seasonal_order"]
        assert order[1] == 1, "d must be 1 per the Day-3 ADF/KPSS result"
        assert seasonal[1] == 1, "D=1 per the seasonal-ACF diagnostic"
        assert seasonal[2] == 1, "Q=1 per the post-differencing negative seasonal ACF"
        assert seasonal[3] == SEASONAL_PERIOD_M == 5
        assert ETS_SPECS[target]["seasonal_periods"] == SEASONAL_PERIOD_M
        assert ETS_SPECS[target]["error"] == "add"
        assert ETS_SPECS[target]["seasonal"] == "add"


# ----------------------------------------------------------------------
# Baseline-beating gate (roadmap Part 6) -- Day 8 will build on this
# ----------------------------------------------------------------------
def _gate_frame(sarima_mae, naive_mae, snaive_mae):
    return pd.DataFrame([
        {"target": "T", "window_rule": "full", "horizon": 1, "model": "naive", "MAE": naive_mae},
        {"target": "T", "window_rule": "full", "horizon": 1, "model": "seasonal_naive", "MAE": snaive_mae},
        {"target": "T", "window_rule": "full", "horizon": 1, "model": "sarima", "MAE": sarima_mae},
    ])


def test_gate_requires_beating_both_baselines_not_just_one():
    from src.evaluation.run_statistical import evaluate_baseline_gate

    # Beats seasonal_naive (50) but loses to naive (10) -> must FAIL.
    beats_one = evaluate_baseline_gate(_gate_frame(sarima_mae=30.0, naive_mae=10.0, snaive_mae=50.0))
    assert bool(beats_one["beats_both_baselines"].iloc[0]) is False

    # Beats both -> passes.
    beats_both = evaluate_baseline_gate(_gate_frame(sarima_mae=8.0, naive_mae=10.0, snaive_mae=50.0))
    assert bool(beats_both["beats_both_baselines"].iloc[0]) is True

    # Beats neither -> fails.
    beats_none = evaluate_baseline_gate(_gate_frame(sarima_mae=99.0, naive_mae=10.0, snaive_mae=50.0))
    assert bool(beats_none["beats_both_baselines"].iloc[0]) is False


def test_gate_ties_do_not_count_as_beating():
    """Equal MAE is not an improvement; the gate must be strict."""
    from src.evaluation.run_statistical import evaluate_baseline_gate

    tie = evaluate_baseline_gate(_gate_frame(sarima_mae=10.0, naive_mae=10.0, snaive_mae=50.0))
    assert bool(tie["beats_both_baselines"].iloc[0]) is False


def test_gate_skill_is_measured_against_the_better_baseline():
    from src.evaluation.run_statistical import evaluate_baseline_gate

    row = evaluate_baseline_gate(_gate_frame(sarima_mae=5.0, naive_mae=10.0, snaive_mae=50.0)).iloc[0]
    assert row["skill_vs_best_baseline_pct"] == pytest.approx(50.0)  # vs 10, not vs 50


def test_persisted_artifacts_carry_spec_provenance_and_summary():
    """A model artifact must be traceable back to the data it was fitted on."""
    import joblib

    from src.config import MODELS_DIR

    for name in ("stat_target1_sarima", "stat_target2_exponential_smoothing"):
        path = MODELS_DIR / ("%s.pkl" % name)
        if not path.exists():
            pytest.skip("run `python -m src.evaluation.run_statistical` first")
        blob = joblib.load(path)
        assert blob["spec"], name
        assert blob["fit_failed"] is False, name
        assert np.isfinite(blob["aic"]) and np.isfinite(blob["bic"]), name
        assert blob["provenance"]["raw_csv_sha256"], name
        assert "development portion" in blob["fitted_on"]
