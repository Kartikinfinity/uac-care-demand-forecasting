"""
tests/test_selection.py -- The champion-selection rule.

Closes the Day-7 open item "three of six recent-regime cells are won by a
baseline". These tests pin the rule so the decision cannot drift at Day 8:
a baseline can win, ties break toward simplicity, and the gate is strict.
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
    MODEL_COMPLEXITY_ORDER,
    PRACTICAL_EQUIVALENCE_SEED,
    RANDOM_SEED,
)
from src.evaluation.selection import (
    complexity_rank,
    is_practically_equivalent,
    paired_absolute_errors,
    paired_bootstrap_difference,
    passes_baseline_gate,
    select_champion,
)


def _cell(preds: dict, actuals):
    """Build a one-cell prediction frame from {model: [forecasts]}."""
    rows = []
    for model, yhat in preds.items():
        for i, (a, p) in enumerate(zip(actuals, yhat)):
            rows.append({"model": model, "fold_id": i, "horizon": 1,
                         "y_true": a, "y_pred": p})
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame):
    g = frame.assign(ae=(frame.y_true - frame.y_pred).abs())
    return g.groupby("model", as_index=False)["ae"].mean().rename(columns={"ae": "MAE"})


# ----------------------------------------------------------------------
# Complexity ordering -- measured, and total
# ----------------------------------------------------------------------
def test_complexity_ordering_puts_baselines_first_and_ml_last():
    for baseline in BASELINE_MODELS:
        assert complexity_rank(baseline) < complexity_rank("sarima")
        assert complexity_rank(baseline) < complexity_rank("random_forest")
    assert complexity_rank("sarima") < complexity_rank("random_forest")
    assert complexity_rank("exponential_smoothing") < complexity_rank("gradient_boosting")


def test_complexity_ordering_covers_every_evaluated_family():
    for model in BASELINE_MODELS + ["sarima", "exponential_smoothing",
                                    "random_forest", "gradient_boosting"]:
        assert model in MODEL_COMPLEXITY_ORDER


def test_unknown_model_ranks_last_rather_than_raising():
    assert complexity_rank("something_new") == len(MODEL_COMPLEXITY_ORDER)


# ----------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------
def test_bootstrap_detects_a_real_and_consistent_difference():
    a = np.full(40, 10.0)   # model A always off by 10
    b = np.full(40, 1.0)    # model B always off by 1
    stats = paired_bootstrap_difference(a, b)
    assert stats["distinguishable"] is True
    assert stats["mean_difference"] == pytest.approx(9.0)
    assert stats["ci_low"] > 0


def test_bootstrap_does_not_manufacture_a_winner_from_noise():
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(10, 5, 14))
    b = np.abs(rng.normal(10, 5, 14))
    stats = paired_bootstrap_difference(a, b)
    assert stats["distinguishable"] is False
    assert stats["ci_low"] < 0 < stats["ci_high"]


def test_bootstrap_is_reproducible_under_the_project_seed():
    rng = np.random.default_rng(1)
    a, b = np.abs(rng.normal(5, 2, 20)), np.abs(rng.normal(5, 2, 20))
    assert PRACTICAL_EQUIVALENCE_SEED == RANDOM_SEED
    first = paired_bootstrap_difference(a, b)
    second = paired_bootstrap_difference(a, b)
    assert first == second


def test_bootstrap_is_empty_safe_and_length_checked():
    empty = paired_bootstrap_difference([], [])
    assert empty["n"] == 0 and empty["distinguishable"] is False
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1.0, 2.0], [1.0])


def test_small_samples_widen_the_interval_rather_than_narrowing_it():
    """The whole point: the margin must adapt to the N actually achieved."""
    rng = np.random.default_rng(7)
    big_a, big_b = np.abs(rng.normal(10, 4, 200)), np.abs(rng.normal(11, 4, 200))
    small_a, small_b = big_a[:12], big_b[:12]
    wide = paired_bootstrap_difference(small_a, small_b)
    narrow = paired_bootstrap_difference(big_a, big_b)
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_pairing_requires_the_same_test_points():
    frame = _cell({"a": [1.0, 2.0], "b": [1.0, 2.0]}, [1.0, 2.0])
    frame.loc[frame.model == "b", "y_true"] = [9.0, 9.0]
    with pytest.raises(ValueError, match="same test points"):
        paired_absolute_errors(frame, "a", "b")


def test_pairing_keeps_only_points_both_models_scored():
    frame = _cell({"a": [1.0, 2.0, 3.0], "b": [1.0, np.nan, 3.0]}, [1.0, 2.0, 3.0])
    ea, eb = paired_absolute_errors(frame, "a", "b")
    assert len(ea) == len(eb) == 2


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------
def test_gate_requires_beating_both_reference_baselines():
    refs = {"naive": 10.0, "seasonal_naive": 50.0}
    assert passes_baseline_gate({"MAE": 8.0}, refs) is True
    assert passes_baseline_gate({"MAE": 30.0}, refs) is False   # beats only one
    assert passes_baseline_gate({"MAE": 99.0}, refs) is False


def test_gate_is_strict_so_a_tie_is_not_an_improvement():
    assert passes_baseline_gate({"MAE": 10.0}, {"naive": 10.0, "seasonal_naive": 50.0}) is False


def test_gate_rejects_a_nan_candidate():
    assert passes_baseline_gate({"MAE": float("nan")}, {"naive": 10.0, "seasonal_naive": 5.0}) is False


# ----------------------------------------------------------------------
# THE DAY-7 OPEN ITEM: a baseline must be allowed to win
# ----------------------------------------------------------------------
def test_a_baseline_wins_and_is_preserved_when_nothing_clears_the_gate():
    """
    The decision Day 7 left open. When no complex model beats both baselines,
    the baseline IS the champion -- not a placeholder to be overridden.
    """
    actuals = [10.0] * 20
    frame = _cell({
        "naive": [10.0] * 20,                       # perfect
        "seasonal_naive": [12.0] * 20,
        "sarima": [20.0] * 20,                      # much worse
        "gradient_boosting": [25.0] * 20,
    }, actuals)
    result = select_champion(frame, _metrics(frame))
    assert result["champion"] == "naive"
    assert result["gate_cleared_by"] == []
    assert "best baseline is the champion" in result["reason"]


def test_a_complex_model_wins_when_it_is_genuinely_and_measurably_better():
    actuals = list(np.linspace(10, 30, 30))
    frame = _cell({
        "naive": [a + 12.0 for a in actuals],
        "seasonal_naive": [a + 14.0 for a in actuals],
        "moving_average": [a + 13.0 for a in actuals],
        "sarima": [a + 0.5 for a in actuals],
    }, actuals)
    result = select_champion(frame, _metrics(frame))
    assert result["champion"] == "sarima"
    assert "sarima" in result["gate_cleared_by"]


def test_a_tie_breaks_toward_the_simpler_model_not_the_lowest_number():
    """
    Addendum Section 5: "within that margin, the simpler/more stable candidate
    wins over the numerically lowest one."
    """
    # The tie must be NOISY, not a deterministic scaling. A paired bootstrap
    # will (correctly) detect even a 2% edge if it is perfectly consistent
    # across every observation -- pairing removes the shared variation and the
    # constant edge survives. So the two candidates get INDEPENDENT error
    # draws: gradient_boosting ends up numerically ahead, but the per-
    # observation differences are noise and the interval spans zero.
    rng = np.random.default_rng(1)
    actuals = list(np.linspace(80, 120, 40))
    err_sarima = rng.normal(0, 3, 40)
    err_gb = rng.normal(0, 3, 40)
    frame = _cell({
        "naive": [a + 20 for a in actuals],
        "seasonal_naive": [a + 22 for a in actuals],
        "sarima": [a + e for a, e in zip(actuals, err_sarima)],
        "gradient_boosting": [a + e for a, e in zip(actuals, err_gb)],
    }, actuals)
    metrics = _metrics(frame)
    leader = metrics.sort_values("MAE").iloc[0]["model"]
    result = select_champion(frame, metrics)

    assert leader == "gradient_boosting", "fixture no longer exercises the tie"
    assert not paired_bootstrap_difference(
        *paired_absolute_errors(frame, "sarima", "gradient_boosting")
    )["distinguishable"], "fixture is no longer a practical tie"
    assert result["champion"] == "sarima", "tie did not break toward simplicity"
    assert result["numerical_leader"] == "gradient_boosting"
    assert complexity_rank(result["champion"]) < complexity_rank(result["numerical_leader"])


def test_champion_result_always_carries_its_evidence_trail():
    actuals = list(np.linspace(50, 80, 25))
    frame = _cell({
        "naive": [a + 9 for a in actuals],
        "seasonal_naive": [a + 11 for a in actuals],
        "sarima": [a + 1 for a in actuals],
    }, actuals)
    result = select_champion(frame, _metrics(frame))
    assert result["reason"]
    assert "tied_with_best" in result
    assert "champion_mae" in result


def test_no_candidate_with_a_finite_score_returns_a_clear_reason():
    frame = _cell({"naive": [np.nan] * 5, "seasonal_naive": [np.nan] * 5}, [1.0] * 5)
    result = select_champion(frame, _metrics(frame))
    assert result["champion"] is None
    assert "finite" in result["reason"]


# ----------------------------------------------------------------------
# The rule reproduces the measured Day-7 finding on real artifacts
# ----------------------------------------------------------------------
def test_rule_reproduces_the_measured_recent_regime_finding():
    """
    On the real recent-regime pools, the single distinguishable
    baseline-vs-complex difference favours the baseline (Discharged, h=14).
    """
    from src.config import FULL_COMPARISON_PATH, ML_PREDICTIONS_PATH, TARGET_2

    if not (ML_PREDICTIONS_PATH.exists() and FULL_COMPARISON_PATH.exists()):
        pytest.skip("run `python -m src.evaluation.run_ml` first")

    preds = pd.read_csv(ML_PREDICTIONS_PATH)
    cell = preds[
        (preds["target"] == TARGET_2)
        & (preds["horizon"] == 14)
        & (preds["window_rule"] == "capped")
        & preds["origin_post_cutoff"]
        & preds["common_support"]
        & preds["y_true_is_observed"]
    ]
    ea, eb = paired_absolute_errors(cell, "exponential_smoothing", "seasonal_naive")
    stats = paired_bootstrap_difference(ea, eb)
    assert stats["n"] > 0
    assert stats["distinguishable"] is True
    assert stats["mean_difference"] > 0, "the baseline should have the lower error here"
    assert not is_practically_equivalent(ea, eb)


def test_a_small_but_perfectly_consistent_edge_is_still_detected():
    """
    The counterpart to the tie test, and the reason the tie fixture uses
    independent noise. Pairing removes shared variation, so a tiny edge that
    holds on EVERY observation is real signal and must not be dismissed as a
    tie -- otherwise the rule would discard genuine, reliable improvements.
    """
    rng = np.random.default_rng(3)
    base = np.abs(rng.normal(0, 3, 40))
    stats = paired_bootstrap_difference(base, base * 0.98)
    assert stats["distinguishable"] is True
    assert stats["ci_low"] > 0
