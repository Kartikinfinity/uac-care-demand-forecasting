"""
tests/test_day8.py -- Champion registry, ensemble, imbalance signal, early warning.

The load-bearing tests here are the two that protect honesty rather than
arithmetic: the early-warning threshold must never see past its own origin, and
the capacity card must never emit a bare number that could read as an official
capacity figure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    BASELINE_MODELS,
    CAPACITY_TIER_LABELS,
    EARLY_WARNING_PERCENTILE,
    EARLY_WARNING_TIERS,
    EARLY_WARNING_TRAILING_WINDOW,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    IMBALANCE_BASELINE_MODELS,
    IMBALANCE_COMPONENT,
    IMBALANCE_INDEPENDENCE_THRESHOLD,
    MASTER_SERIES_PATH,
    MODEL_REGISTRY_PATH,
    SELECTION_SCOPE,
    SELECTION_WINDOW_RULE,
    TARGET_1,
    TARGET_2,
)
from src.signals.early_warning import (
    PROXY_DISCLAIMER,
    capacity_tier,
    classify_tier,
    evaluate_origin,
    trailing_threshold,
)


# ======================================================================
# EARLY WARNING -- the leakage-critical invariant
# ======================================================================
def test_threshold_never_sees_data_past_its_own_origin():
    """
    Invariant: "never compute the early-warning threshold using data past the
    historical origin being evaluated". A threshold built with hindsight makes
    every backtested lead time meaningless.
    """
    series = np.arange(200, dtype=float)
    early = trailing_threshold(series, origin_pos=100)
    # Appending an enormous future spike must change nothing at origin 100.
    tampered = series.copy()
    tampered[101:] = 10_000.0
    assert trailing_threshold(tampered, origin_pos=100) == early


def test_threshold_uses_only_the_trailing_window():
    series = np.concatenate([np.full(100, 1000.0), np.full(100, 1.0)])
    # At origin 199 the window holds only the recent low values.
    assert trailing_threshold(series, 199, window=50) == pytest.approx(1.0)
    # At origin 99 it holds only the old high ones.
    assert trailing_threshold(series, 99, window=50) == pytest.approx(1000.0)


def test_threshold_tracks_a_regime_shift_rather_than_full_history():
    """
    Why a window exists at all: HHS Care falls ~5.8x, so a full-history
    percentile would sit at old levels and could never fire in the new regime.
    """
    df = pd.read_parquet(MASTER_SERIES_PATH)
    values = df[TARGET_1].astype(float).to_numpy()
    observed = (~df["is_imputed"]).to_numpy()
    last = len(values) - 1
    windowed = trailing_threshold(values, last, observed)
    full_history = trailing_threshold(values, last, observed, window=len(values))
    assert windowed < full_history / 2, "the window is not tracking the recent regime"


def test_threshold_excludes_interpolated_values():
    """A percentile over invented values would set the alarm from a straight line."""
    values = np.concatenate([np.full(50, 10.0), np.full(10, 9999.0)])
    observed = np.concatenate([np.ones(50, dtype=bool), np.zeros(10, dtype=bool)])
    assert trailing_threshold(values, 59, observed) == pytest.approx(10.0)
    assert trailing_threshold(values, 59, None) > 10.0


def test_threshold_is_nan_when_nothing_observed_rather_than_guessing():
    values = np.full(30, np.nan)
    assert np.isnan(trailing_threshold(values, 29))


def test_threshold_rejects_an_out_of_range_origin():
    with pytest.raises(IndexError):
        trailing_threshold(np.arange(10, dtype=float), 10)


def test_percentile_and_window_are_the_frozen_conventions():
    assert EARLY_WARNING_PERCENTILE == 90
    # Deliberately tied to an already-frozen length so no new free parameter enters.
    assert EARLY_WARNING_TRAILING_WINDOW == FINAL_TEST_WINDOW


# ======================================================================
# EARLY WARNING -- tiers
# ======================================================================
def test_tier_mapping_matches_the_addendum():
    """Watch from h=14, Warning from h=7, Alert from h=1."""
    assert EARLY_WARNING_TIERS == {14: "Watch", 7: "Warning", 1: "Alert"}


def test_only_crossing_horizons_fire():
    out = classify_tier({1: 50.0, 7: 150.0, 14: 200.0}, threshold=100.0)
    assert out["fired"] == {7: "Warning", 14: "Watch"}
    assert 1 not in out["fired"]


def test_highest_tier_is_the_shortest_crossing_horizon():
    """Alert outranks Warning outranks Watch -- nearer forecasts are more confident."""
    out = classify_tier({1: 150.0, 7: 150.0, 14: 150.0}, threshold=100.0)
    assert out["highest_tier"] == "Alert"
    assert out["earliest_tier"] == "Watch"


def test_a_watch_only_firing_is_labelled_lower_confidence():
    out = classify_tier({1: 50.0, 7: 50.0, 14: 150.0}, threshold=100.0)
    assert out["highest_tier"] == "Watch"
    assert "lower-confidence" in out["confidence_note"]


def test_nothing_fires_below_the_threshold():
    out = classify_tier({1: 10.0, 7: 20.0, 14: 30.0}, threshold=100.0)
    assert out["fired"] == {} and out["highest_tier"] is None


def test_an_unavailable_threshold_fires_nothing_rather_than_everything():
    out = classify_tier({1: 1e9}, threshold=float("nan"))
    assert out["fired"] == {} and out["highest_tier"] is None


def test_evaluate_origin_always_carries_the_proxy_disclaimer():
    """The signal must never travel without its caveat."""
    result = evaluate_origin(np.arange(100, dtype=float), 50, {1: 1e6, 7: 0.0, 14: 0.0})
    assert result["is_proxy"] is True
    assert "no official capacity threshold" in result["disclaimer"].lower()
    assert PROXY_DISCLAIMER == result["disclaimer"]


# ======================================================================
# CAPACITY CARD -- qualitative only
# ======================================================================
def test_capacity_card_returns_a_label_never_a_number():
    for forecast in (10.0, 95.0, 150.0, float("nan")):
        label = capacity_tier(forecast, threshold=100.0)
        assert label in CAPACITY_TIER_LABELS
        assert isinstance(label, str)


def test_capacity_tiers_are_ordered_as_expected():
    assert capacity_tier(150.0, 100.0) == "High"
    assert capacity_tier(95.0, 100.0) == "Elevated"
    assert capacity_tier(10.0, 100.0) == "Normal"


def test_capacity_card_degrades_to_normal_on_a_missing_threshold():
    assert capacity_tier(150.0, float("nan")) == "Normal"
    assert capacity_tier(float("nan"), 100.0) == "Normal"


# ======================================================================
# REGISTRY, ENSEMBLE, IMBALANCE -- against the real Day-8 artifacts
# ======================================================================
@pytest.fixture(scope="module")
def registry():
    if not MODEL_REGISTRY_PATH.exists():
        pytest.skip("run `python -m src.evaluation.run_selection` first")
    from src.models.registry import read_registry
    return read_registry()


def test_registry_covers_every_target_and_horizon(registry):
    from src.models.registry import champion_for
    for target in (TARGET_1, TARGET_2):
        for horizon in FORECAST_HORIZONS:
            entry = champion_for(registry, target, horizon)
            assert entry is not None, "%s h=%d missing" % (target, horizon)
            assert entry["champion"]


def test_every_registry_entry_carries_its_evidence_trail(registry):
    """The Day-8 checkpoint: every claim traces to a metric row."""
    for entry in registry["entries"]:
        assert entry["reason"]
        assert entry["full_dev_ranking"], "no ranking table behind the decision"
        assert entry["bias_diagnostics"], "no bias evidence"
        assert entry["stability_diagnostics"], "no stability evidence"
        assert entry["selection_scope"] == SELECTION_SCOPE
        assert entry["window_rule"] == SELECTION_WINDOW_RULE
        assert "rankings_agree" in entry


def test_registry_records_both_rankings_not_just_the_governing_one(registry):
    """Addendum Section 5 requires both to be reported."""
    for entry in registry["entries"]:
        assert entry["champion_full_dev"]
        assert entry["champion"]
        if entry["rankings_agree"]:
            assert entry["champion"] == entry["champion_full_dev"]
        else:
            assert entry["champion"] != entry["champion_full_dev"]
            assert "recent-regime" in entry["governing_scope_note"]


def test_registry_is_traceable_to_a_data_version(registry):
    assert registry["provenance"] is not None
    assert registry["provenance"]["raw_csv_sha256"]
    assert registry["selection_rule"]["baselines_eligible"] is True


def test_a_baseline_champion_is_recorded_as_a_result_not_a_failure(registry):
    """
    The Day-7 open item, now settled on real data: baselines win, and the
    registry preserves that rather than substituting a complex model.
    """
    champions = [e["champion"] for e in registry["entries"]]
    assert any(c in BASELINE_MODELS for c in champions)
    for entry in registry["entries"]:
        if entry["champion"] in BASELINE_MODELS:
            assert entry["reason"], "a baseline champion must still carry its reason"


def test_ensemble_is_a_plain_average_of_its_two_components():
    from src.config import ENSEMBLE_PREDICTIONS_PATH, ML_PREDICTIONS_PATH

    if not ENSEMBLE_PREDICTIONS_PATH.exists():
        pytest.skip("run `python -m src.evaluation.run_selection` first")
    ens = pd.read_csv(ENSEMBLE_PREDICTIONS_PATH)
    preds = pd.read_csv(ML_PREDICTIONS_PATH)
    assert len(ens) > 0

    row = ens.dropna(subset=["y_pred"]).iloc[0]
    stat, ml = [s.strip() for s in row["ensemble_components"].split("+")]
    key = ["target", "window_rule", "fold_id", "horizon"]
    sel = {k: row[k] for k in key}
    def _pred(model):
        m = preds[(preds["model"] == model)]
        for k, v in sel.items():
            m = m[m[k] == v]
        return float(m["y_pred"].iloc[0])
    assert row["y_pred"] == pytest.approx((_pred(stat) + _pred(ml)) / 2.0)


def test_ensemble_abstains_when_a_component_abstains():
    """It must never fall back to the surviving component -- that is a different model."""
    from src.config import ENSEMBLE_PREDICTIONS_PATH, ML_PREDICTIONS_PATH

    if not ENSEMBLE_PREDICTIONS_PATH.exists():
        pytest.skip("run `python -m src.evaluation.run_selection` first")
    ens = pd.read_csv(ENSEMBLE_PREDICTIONS_PATH)
    preds = pd.read_csv(ML_PREDICTIONS_PATH)
    key = ["target", "window_rule", "fold_id", "horizon"]
    missing = ens[ens["y_pred"].isna()]
    for _, row in missing.head(5).iterrows():
        stat, ml = [s.strip() for s in row["ensemble_components"].split("+")]
        sub = preds[preds["model"].isin([stat, ml])]
        for k in key:
            sub = sub[sub[k] == row[k]]
        assert sub["y_pred"].isna().any(), "ensemble is NaN but both components forecast"


def test_imbalance_correlation_is_measured_from_paired_oos_residuals():
    from src.config import IMBALANCE_CORRELATION_PATH

    if not IMBALANCE_CORRELATION_PATH.exists():
        pytest.skip("run `python -m src.evaluation.run_selection` first")
    corr = pd.read_csv(IMBALANCE_CORRELATION_PATH)
    assert len(corr) > 0
    assert set(corr["transferred_model"]) <= set(IMBALANCE_BASELINE_MODELS), (
        "Transferred Out must use the baseline treatment, not a champion track"
    )
    assert (corr["n_paired"] > 0).all()
    assert corr["correlation"].abs().le(1.0).all()
    # The independence flag must follow the measured number, not a prior.
    expected = corr["correlation"].abs() < IMBALANCE_INDEPENDENCE_THRESHOLD
    assert (corr["independence_admissible"] == expected).all()


def test_transferred_out_is_never_promoted_to_a_third_target(registry):
    """Addendum Section 6: it stays a derived-signal component."""
    targets = {e["target"] for e in registry["entries"]}
    assert IMBALANCE_COMPONENT not in targets
    assert targets == {TARGET_1, TARGET_2}
