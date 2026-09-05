"""
tests/test_day9.py -- Intervals, coverage calibration, and batch generation.

The load-bearing tests protect three honesty properties:
  * intervals are built only from OUT-OF-SAMPLE residuals (invariant 5);
  * coverage is never reported without its binomial band;
  * the holdout is used once, confirmatorily, and never feeds its own interval.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    EMPIRICAL_INTERVAL_ALPHA,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    FORECAST_PROVENANCE_PATH,
    FORWARD_FORECASTS_PATH,
    HOLDOUT_EVALUATION_PATH,
    IMBALANCE_FORECAST_PATH,
    IMBALANCE_INDEPENDENCE_THRESHOLD,
    INTERVAL_COVERAGE_PATH,
    KPI_SUMMARY_PATH,
    MASTER_SERIES_PATH,
    MIN_RESIDUALS_FOR_INTERVAL,
    SELECTION_WINDOW_RULE,
    TARGET_1,
    TARGET_2,
)
from src.evaluation.intervals import (
    coverage_with_band,
    empirical_interval,
    imbalance_variance,
    wilson_interval,
)


# ======================================================================
# EMPIRICAL INTERVALS
# ======================================================================
def test_interval_is_the_residual_quantiles_around_the_point_forecast():
    residuals = np.arange(-50, 51, dtype=float)      # symmetric, n=101
    out = empirical_interval(100.0, residuals, alpha=0.05)
    lo, hi = np.percentile(residuals, [2.5, 97.5])
    assert out["lower"] == pytest.approx(100.0 + lo)
    assert out["upper"] == pytest.approx(100.0 + hi)
    assert out["sufficient"] is True


def test_interval_preserves_skew_rather_than_assuming_symmetry():
    """
    Using residual quantiles directly, not mean +/- k*sd, keeps genuine skew.
    A right-skewed error distribution must produce a right-skewed interval.
    """
    residuals = np.concatenate([np.full(90, -1.0), np.linspace(5, 100, 10)])
    out = empirical_interval(0.0, residuals, alpha=0.05)
    assert abs(out["upper"]) > abs(out["lower"]) * 2, "skew was flattened"


def test_interval_is_withheld_when_the_residual_pool_is_too_thin():
    """At n<10 the 95% quantiles are the sample min/max -- not an interval estimate."""
    out = empirical_interval(100.0, np.arange(5, dtype=float))
    assert out["sufficient"] is False
    assert np.isnan(out["lower"]) and np.isnan(out["upper"])
    assert "min and max" in out["reason"]


def test_the_residual_floor_boundary_is_exact():
    just_enough = empirical_interval(0.0, np.arange(MIN_RESIDUALS_FOR_INTERVAL, dtype=float))
    one_short = empirical_interval(0.0, np.arange(MIN_RESIDUALS_FOR_INTERVAL - 1, dtype=float))
    assert just_enough["sufficient"] is True
    assert one_short["sufficient"] is False


def test_interval_widens_with_a_more_dispersed_residual_pool():
    tight = empirical_interval(0.0, np.random.default_rng(0).normal(0, 1, 200))
    wide = empirical_interval(0.0, np.random.default_rng(0).normal(0, 10, 200))
    assert wide["width"] > tight["width"] * 5


def test_lower_bound_clipping_is_applied_and_visible():
    """
    Counts of children cannot be negative. Clipping must happen AND be
    recorded -- a silently clipped interval hides that the model wanted to
    predict something impossible.
    """
    out = empirical_interval(5.0, np.arange(-40, 41, dtype=float), lower_bound=0.0)
    assert out["lower"] == 0.0
    assert out["lower_unclipped"] < 0
    assert out["clipped_at_lower_bound"] is True
    assert out["width"] == pytest.approx(out["upper"] - 0.0)


def test_no_clipping_flag_when_the_interval_is_already_valid():
    out = empirical_interval(1000.0, np.arange(-40, 41, dtype=float), lower_bound=0.0)
    assert out["clipped_at_lower_bound"] is False
    assert out["lower"] == out["lower_unclipped"]


def test_interval_handles_a_missing_point_forecast():
    out = empirical_interval(float("nan"), np.arange(50, dtype=float))
    assert out["sufficient"] is False


# ======================================================================
# COVERAGE + BINOMIAL BAND
# ======================================================================
def test_wilson_band_stays_inside_zero_to_one_at_perfect_coverage():
    """
    Why Wilson and not the normal approximation: at n=12 with 12 hits the
    normal interval runs past 1.0 and has zero width.
    """
    lo, hi = wilson_interval(12, 12)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo < 1.0, "a perfect run at n=12 must not imply certainty"


def test_wilson_band_narrows_as_n_grows():
    small = wilson_interval(9, 12)
    large = wilson_interval(750, 1000)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 5


def test_coverage_reports_the_band_alongside_the_rate():
    actuals = np.array([1.0, 2.0, 3.0, 100.0])
    lowers = np.zeros(4)
    uppers = np.full(4, 10.0)
    out = coverage_with_band(actuals, lowers, uppers, nominal=0.95)
    assert out["n"] == 4 and out["n_covered"] == 3
    assert out["empirical_coverage"] == pytest.approx(0.75)
    assert out["band_low"] < out["band_high"]
    assert "small" in out["note"]


def test_coverage_flags_a_gap_that_is_outside_the_band():
    """A real calibration failure must be detectable, not smoothed away."""
    actuals = np.arange(100, dtype=float)
    lowers = np.zeros(100)
    uppers = np.full(100, 49.0)          # covers ~half
    out = coverage_with_band(actuals, lowers, uppers, nominal=0.95)
    assert out["covers_nominal"] is False


def test_coverage_is_empty_safe():
    out = coverage_with_band([], [], [], nominal=0.95)
    assert out["n"] == 0 and np.isnan(out["empirical_coverage"])


# ======================================================================
# IMBALANCE VARIANCE
# ======================================================================
def test_full_covariance_form_is_the_default():
    """The independence simplification must be earned, not assumed."""
    out = imbalance_variance(10.0, 20.0, covariance=4.0, correlation=0.9)
    assert out["form_used"] == "full_covariance"
    assert out["variance"] == pytest.approx(10 + 20 - 2 * 4)


def test_independence_form_applies_only_below_the_measured_threshold():
    below = imbalance_variance(10.0, 20.0, covariance=1.0,
                               correlation=IMBALANCE_INDEPENDENCE_THRESHOLD / 2)
    above = imbalance_variance(10.0, 20.0, covariance=1.0,
                               correlation=IMBALANCE_INDEPENDENCE_THRESHOLD + 0.1)
    assert below["form_used"] == "independence"
    assert below["variance"] == pytest.approx(30.0)
    assert above["form_used"] == "full_covariance"


def test_unmeasured_correlation_falls_back_to_the_full_form():
    out = imbalance_variance(10.0, 20.0, covariance=3.0, correlation=None)
    assert out["form_used"] == "full_covariance"


def test_variance_is_never_negative():
    out = imbalance_variance(1.0, 1.0, covariance=100.0, correlation=0.99)
    assert out["variance"] >= 0.0 and np.isfinite(out["std"])


# ======================================================================
# GENERATED ARTIFACTS
# ======================================================================
@pytest.fixture(scope="module")
def artifacts():
    needed = [FORWARD_FORECASTS_PATH, INTERVAL_COVERAGE_PATH, HOLDOUT_EVALUATION_PATH,
              IMBALANCE_FORECAST_PATH, KPI_SUMMARY_PATH, FORECAST_PROVENANCE_PATH]
    if not all(p.exists() for p in needed):
        pytest.skip("run `python -m src.forecast.generate` first")
    return {
        "forward": pd.read_csv(FORWARD_FORECASTS_PATH),
        "coverage": pd.read_csv(INTERVAL_COVERAGE_PATH),
        "holdout": pd.read_csv(HOLDOUT_EVALUATION_PATH),
        "imbalance": pd.read_csv(IMBALANCE_FORECAST_PATH),
        "kpis": pd.read_csv(KPI_SUMMARY_PATH),
        "provenance": json.loads(FORECAST_PROVENANCE_PATH.read_text(encoding="utf-8")),
    }


def test_a_forward_forecast_exists_for_every_model_target_and_horizon(artifacts):
    """
    Every candidate, not just champions -- the documented "model toggle" needs
    them, and the app may never fit anything in a page callback.
    """
    fwd = artifacts["forward"]
    for target in (TARGET_1, TARGET_2):
        sub = fwd[fwd["target"] == target]
        assert set(sub["horizon"]) == set(FORECAST_HORIZONS)
        for horizon in FORECAST_HORIZONS:
            models = set(sub[sub["horizon"] == horizon]["model"])
            assert {"naive", "seasonal_naive", "moving_average", "sarima",
                    "exponential_smoothing", "random_forest", "gradient_boosting",
                    "ensemble"} <= models


def test_exactly_one_champion_is_flagged_per_target_and_horizon(artifacts):
    fwd = artifacts["forward"]
    counts = fwd.groupby(["target", "horizon"])["is_champion"].sum()
    assert (counts == 1).all(), "champion flag is not unique per cell"


def test_the_flagged_champion_matches_the_registry(artifacts):
    from src.models.registry import read_registry

    registry = read_registry()
    fwd = artifacts["forward"]
    for entry in registry["entries"]:
        row = fwd[(fwd["target"] == entry["target"])
                  & (fwd["horizon"] == entry["horizon"])
                  & fwd["is_champion"]]
        assert len(row) == 1
        assert row.iloc[0]["model"] == entry["champion"]


def test_the_ensemble_also_gets_an_interval(artifacts):
    """
    REGRESSION. The ensemble was built at Day 8, after Day 7 persisted its
    residuals, so it initially had an empty pool and was the one candidate the
    dashboard could show a point forecast for but no band.
    """
    ens = artifacts["forward"][artifacts["forward"]["model"] == "ensemble"]
    assert len(ens) > 0
    assert (ens["n_residuals"] > 0).all(), "ensemble residual pool is empty"


def test_no_published_interval_bound_is_negative(artifacts):
    """These are counts of children; a negative bound must never reach a reader."""
    fwd = artifacts["forward"]
    published = fwd[fwd["interval_sufficient"]]
    assert (published["lower"] >= 0).all()
    assert (published["upper"] >= 0).all()


def test_point_forecast_lies_inside_its_own_interval(artifacts):
    fwd = artifacts["forward"][artifacts["forward"]["interval_sufficient"]]
    assert (fwd["lower"] <= fwd["point_forecast"]).all()
    assert (fwd["point_forecast"] <= fwd["upper"]).all()


def test_intervals_widen_with_horizon_for_the_stock_target(artifacts):
    """Uncertainty must grow with distance; a flat band would signal a bug."""
    fwd = artifacts["forward"]
    care = fwd[(fwd["target"] == TARGET_1) & fwd["interval_sufficient"]].sort_values("horizon")
    widths = care["interval_width"].to_numpy()
    assert widths[0] < widths[-1], "h=14 is not wider than h=1"


def test_every_coverage_row_carries_a_binomial_band(artifacts):
    cov = artifacts["coverage"]
    assert len(cov) > 0
    for col in ("n", "empirical_coverage", "nominal_coverage", "band_low",
                "band_high", "covers_nominal"):
        assert col in cov.columns
    assert (cov["band_low"] <= cov["empirical_coverage"]).all()
    assert (cov["empirical_coverage"] <= cov["band_high"]).all()


def test_small_sample_coverage_is_disclosed_as_such(artifacts):
    cov = artifacts["coverage"]
    small = cov[cov["n"] < 40]
    assert len(small) > 0, "fixture no longer exercises the small-N disclosure"
    assert small["note"].str.contains("small", case=False).all()


def test_holdout_is_used_once_and_recorded_as_confirmatory(artifacts):
    prov = artifacts["provenance"]
    assert prov["holdout_touched"] is True
    assert "confirmatory" in prov["holdout_use"]
    assert "never fed back" in prov["holdout_use"]


def test_holdout_points_all_fall_inside_the_reserved_window(artifacts):
    """Nothing labelled a holdout evaluation may come from the development portion."""
    from src.evaluation.walk_forward import resolve_split_boundaries

    df = pd.read_parquet(MASTER_SERIES_PATH)
    holdout_start, _ = resolve_split_boundaries(df["is_imputed"], FINAL_TEST_WINDOW)
    assert (artifacts["holdout"]["test_pos"] >= holdout_start).all()


def test_holdout_intervals_come_from_the_development_residual_pool(artifacts):
    """
    Rebuilding an interval from holdout residuals would leak the holdout into
    its own evaluation. The published holdout interval must match the one built
    from the development pool for the same target/horizon/champion.
    """
    from src.config import ML_RESIDUALS_PATH

    residuals = pd.read_csv(ML_RESIDUALS_PATH)
    holdout = artifacts["holdout"]
    row = holdout.dropna(subset=["lower"]).iloc[0]
    pool = residuals[
        (residuals["target"] == row["target"]) & (residuals["model"] == row["champion"])
        & (residuals["horizon"] == row["horizon"])
        & (residuals["window_rule"] == SELECTION_WINDOW_RULE)
        & residuals["origin_post_cutoff"]
    ]["residual"].to_numpy()
    expected = empirical_interval(row["point_forecast"], pool, alpha=EMPIRICAL_INTERVAL_ALPHA)
    assert row["lower"] == pytest.approx(expected["lower"])
    assert row["upper"] == pytest.approx(expected["upper"])


def test_imbalance_components_share_the_same_residual_scope(artifacts):
    """
    REGRESSION. The Transferred variance was originally read from the Day-8
    all-folds table while Discharged came from the post-cutoff pool. Mixing the
    2023-24 regime with the current one inflated the combined uncertainty about
    eight-fold and swamped a single-digit net-pressure signal.
    """
    imb = artifacts["imbalance"]
    assert (imb["n_paired_residuals"] > 0).all()
    # Post-cutoff pools are low tens, not the ~350 an all-folds pairing gives.
    assert (imb["n_paired_residuals"] < 40).all()
    # Both component variances must be the same order of magnitude now.
    ratio = imb["var_transferred"] / imb["var_discharged"]
    assert ratio.between(0.02, 50).all(), "components still span different regimes"


def test_imbalance_uses_the_measured_correlation_to_pick_its_form(artifacts):
    imb = artifacts["imbalance"]
    for _, row in imb.iterrows():
        expected = ("independence"
                    if abs(row["measured_correlation"]) < IMBALANCE_INDEPENDENCE_THRESHOLD
                    else "full_covariance")
        assert row["uncertainty_form"] == expected


def test_kpis_carry_their_formulas_and_the_proxy_disclaimer(artifacts):
    kpis = artifacts["kpis"]
    assert len(kpis) == 2
    for col in ("capacity_tier_formula", "surge_lead_time_formula", "stability_formula"):
        assert kpis[col].notna().all() and (kpis[col].str.len() > 20).all()
    assert kpis["threshold_is_proxy"].all()
    assert kpis["proxy_disclaimer"].str.contains("no official capacity threshold",
                                                 case=False).all()


def test_stability_index_documents_that_it_is_a_substitution(artifacts):
    assert artifacts["kpis"]["stability_formula"].str.contains("SUBSTITUTED").all()


def test_false_positive_and_negative_rates_are_present_not_optional(artifacts):
    """Addendum Section 8 makes these required metrics."""
    kpis = artifacts["kpis"]
    assert "false_positive_rate" in kpis.columns
    assert "false_negative_rate" in kpis.columns
    assert kpis["n_backtest_origins"].notna().all()


def test_provenance_ties_every_artifact_to_one_data_version(artifacts):
    prov = artifacts["provenance"]
    from src.data.validate import read_provenance

    data = read_provenance()
    assert prov["raw_csv_sha256"] == data["raw_csv_sha256"]
    assert prov["master_series_sha256"] == data["master_series_sha256"]
    assert prov["data_as_of"] == data["data_as_of"]
    assert len(prov["winning_configuration"]) == 2 * len(FORECAST_HORIZONS)
    assert prov["refresh_policy"].startswith("manual only")


def test_provenance_records_the_early_warning_signal_as_a_proxy(artifacts):
    ew = artifacts["provenance"]["early_warning"]
    assert ew["is_proxy"] is True
    assert "no official capacity threshold" in ew["disclaimer"].lower()
    assert ew["percentile"] == 90
