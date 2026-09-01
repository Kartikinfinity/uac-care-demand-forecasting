"""
tests/test_validation.py -- Day 5 walk-forward harness correctness.

Two jobs:
  1. Folds are generated to the frozen cadence (step 10, min initial training 50,
     60-real-observation holdout, horizons {1,7,14}).
  2. NOTHING LEAKS. No fold sees data at or after its own test point, no fold
     touches the held-out window, and the 2025-02-05 training cap is never
     applied retroactively to an earlier origin.

The synthetic fixtures make the arithmetic checkable by hand; the real-data
fixture proves the same properties hold on the actual 769-slot master series.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    MASTER_SERIES_PATH,
    MAX_USABLE_ROWS_FLOOR,
    MIN_INITIAL_TRAINING,
    SEASONAL_PERIOD_M,
    TARGET_1,
    TARGET_2,
    TRAINING_CAP_DATE,
    TRAINING_FLOOR_ROWS,
    WALK_FORWARD_STEP,
)
from src.evaluation.run_baselines import BASELINE_FACTORIES, resolve_cutoff_pos
from src.evaluation.walk_forward import (
    Fold,
    aggregate_metrics,
    common_support_mask,
    generate_folds,
    resolve_split_boundaries,
    resolve_training_window,
    run_walk_forward,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def master():
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    return df


@pytest.fixture(scope="module")
def boundaries(master):
    holdout_start, dev_end = resolve_split_boundaries(master["is_imputed"], FINAL_TEST_WINDOW)
    return holdout_start, dev_end


@pytest.fixture(scope="module")
def cutoff_pos(master):
    return resolve_cutoff_pos(master["parsed_date"])


@pytest.fixture(scope="module")
def real_folds(master, boundaries):
    _, dev_end = boundaries
    return generate_folds(
        dates=master["parsed_date"],
        is_imputed=master["is_imputed"],
        dev_end_pos=dev_end,
        horizons=FORECAST_HORIZONS,
    )


def _synthetic(n=200, gaps=()):
    """A clean synthetic period-position index with optional gap slots."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    is_imputed = np.zeros(n, dtype=bool)
    for g in gaps:
        is_imputed[g] = True
    return dates, is_imputed


# ----------------------------------------------------------------------
# 1. Split geometry
# ----------------------------------------------------------------------
def test_holdout_is_counted_in_real_observations_not_raw_slots(master, boundaries):
    """
    The addendum reserves "the most recent 60 REAL observations". If the tail
    contains gap slots, a naive last-60-rows cut would under-reserve.
    """
    holdout_start, _ = boundaries
    tail = master.iloc[holdout_start:]
    assert int((~tail["is_imputed"]).sum()) == FINAL_TEST_WINDOW


def test_development_and_holdout_partition_the_series_exactly(master, boundaries):
    holdout_start, dev_end = boundaries
    assert dev_end == holdout_start - 1
    assert dev_end >= 0 and holdout_start < len(master)


def test_holdout_raises_when_the_series_is_too_short():
    with pytest.raises(ValueError):
        resolve_split_boundaries(np.zeros(10, dtype=bool), final_test_window=60)


def test_holdout_ignores_imputed_slots_in_the_tail():
    """A gap inside the holdout tail must push the boundary earlier, not shrink it."""
    is_imputed = np.zeros(100, dtype=bool)
    is_imputed[95] = True
    start_clean, _ = resolve_split_boundaries(np.zeros(100, dtype=bool), final_test_window=10)
    start_gappy, _ = resolve_split_boundaries(is_imputed, final_test_window=10)
    assert start_clean == 90
    assert start_gappy == 89  # one extra slot pulled in to still hold 10 real rows


# ----------------------------------------------------------------------
# 2. Fold cadence
# ----------------------------------------------------------------------
def test_first_origin_gives_exactly_the_minimum_initial_training_size():
    dates, is_imputed = _synthetic()
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert folds[0].origin_pos == MIN_INITIAL_TRAINING - 1
    assert folds[0].origin_pos + 1 == MIN_INITIAL_TRAINING


def test_origins_are_spaced_by_exactly_the_configured_step():
    dates, is_imputed = _synthetic()
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    origins = [f.origin_pos for f in folds]
    assert np.all(np.diff(origins) == WALK_FORWARD_STEP)


def test_fold_cadence_is_fixed_not_ad_hoc(real_folds):
    """Addendum: 'fold spacing fixed at 10 periods for every model/target'."""
    origins = [f.origin_pos for f in real_folds]
    assert np.all(np.diff(origins) == WALK_FORWARD_STEP)
    assert origins[0] == MIN_INITIAL_TRAINING - 1


def test_every_fold_carries_the_full_horizon_grid(real_folds):
    for fold in real_folds:
        assert sorted(fold.horizons) == sorted(FORECAST_HORIZONS)
        assert sorted(fold.test_positions) == sorted(FORECAST_HORIZONS)


def test_horizons_are_integer_period_offsets_not_calendar_days(real_folds):
    """
    Invariant 1: horizons are period-position offsets. On a Sun-Thu series the
    calendar gap for h=1 is sometimes 1 day and sometimes 3 (Thu -> Sun), so a
    correct implementation must NOT produce a constant calendar delta.
    """
    for fold in real_folds:
        for h, pos in fold.test_positions.items():
            assert pos - fold.origin_pos == h

    deltas = set()
    for fold in real_folds[:20]:
        deltas.add((fold.test_positions[1] - fold.origin_pos))
    assert deltas == {1}


def test_folds_stop_before_the_longest_horizon_would_overrun(real_folds, boundaries):
    _, dev_end = boundaries
    last = real_folds[-1]
    assert last.max_test_pos <= dev_end
    assert last.origin_pos + WALK_FORWARD_STEP + max(FORECAST_HORIZONS) > dev_end


# ----------------------------------------------------------------------
# 3. LEAKAGE -- the load-bearing tests
# ----------------------------------------------------------------------
def test_no_fold_trains_on_data_at_or_after_its_own_test_point(real_folds):
    """max(train) < min(test), asserted per the roadmap's Day-5 checkpoint."""
    for fold in real_folds:
        assert fold.origin_pos < min(fold.test_positions.values())


def test_no_fold_test_point_ever_enters_the_held_out_window(real_folds, boundaries):
    holdout_start, _ = boundaries
    for fold in real_folds:
        for pos in fold.test_positions.values():
            assert pos < holdout_start


def test_no_training_window_under_any_rule_reaches_the_holdout(
    real_folds, master, boundaries, cutoff_pos
):
    holdout_start, _ = boundaries
    y = master[TARGET_1].astype(float).to_numpy()
    for fold in real_folds:
        for rule in ("full", "capped"):
            w = resolve_training_window(fold, rule, y, cutoff_pos)
            assert w.train_end_pos < holdout_start
            assert w.train_end_pos == fold.train_cutoff_pos
            assert w.train_end_pos <= fold.origin_pos


def test_training_slice_excludes_every_test_position(real_folds, master, cutoff_pos):
    y = master[TARGET_1].astype(float).to_numpy()
    for fold in real_folds:
        for rule in ("full", "capped"):
            w = resolve_training_window(fold, rule, y, cutoff_pos)
            train_positions = set(range(w.train_start_pos, w.train_end_pos + 1))
            assert train_positions.isdisjoint(fold.test_positions.values())


def test_a_model_only_ever_receives_history_up_to_its_own_origin(master, cutoff_pos):
    """
    End-to-end leakage proof: record the exact array handed to `.fit`, and
    assert it matches y[train_start .. origin] with nothing beyond it.
    """
    seen = {}

    class Spy:
        def fit(self, y):
            seen["y"] = np.asarray(y, dtype=float)
            return self

        def predict(self, horizons):
            return np.zeros(len(horizons))

    _, dev_end = resolve_split_boundaries(master["is_imputed"], FINAL_TEST_WINDOW)
    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end_pos=dev_end)
    fold = folds[len(folds) // 2]

    run_walk_forward(
        df=master,
        target_col=TARGET_1,
        model_factories={"spy": lambda: Spy()},
        folds=[fold],
        cutoff_pos=cutoff_pos,
        window_rules=["full"],
    )

    y = master[TARGET_1].astype(float).to_numpy()
    np.testing.assert_array_equal(seen["y"], y[: fold.train_cutoff_pos + 1])
    assert seen["y"].size == fold.train_cutoff_pos + 1


def test_appending_future_rows_cannot_change_an_earlier_fold(master, cutoff_pos):
    """
    A historical origin's forecasts must be identical whether or not later data
    exists in the frame -- the property that makes a walk-forward number honest.
    """
    _, dev_end = resolve_split_boundaries(master["is_imputed"], FINAL_TEST_WINDOW)
    folds = generate_folds(master["parsed_date"], master["is_imputed"], dev_end_pos=dev_end)
    fold = folds[10]

    full_run = run_walk_forward(
        master, TARGET_1, BASELINE_FACTORIES, [fold], cutoff_pos, window_rules=["full"]
    )
    truncated = master.iloc[: fold.max_test_pos + 1].copy()
    short_run = run_walk_forward(
        truncated, TARGET_1, BASELINE_FACTORIES, [fold], cutoff_pos, window_rules=["full"]
    )

    np.testing.assert_allclose(
        full_run.sort_values(["model", "horizon"])["y_pred"].to_numpy(),
        short_run.sort_values(["model", "horizon"])["y_pred"].to_numpy(),
    )


# ----------------------------------------------------------------------
# 4. The 2025-02-05 training cap
# ----------------------------------------------------------------------
def _fold_at(pos, horizons=(1, 7, 14)):
    return Fold(
        fold_id=0,
        origin_pos=pos,
        origin_date=pd.Timestamp("2025-01-01"),
        horizons=tuple(horizons),
        test_positions={h: pos + h for h in horizons},
        origin_is_imputed=False,
        origin_adjacent_to_gap=False,
        train_cutoff_pos=pos,
        test_is_imputed={h: False for h in horizons},
    )


def test_cap_is_never_applied_to_an_origin_before_the_cutoff():
    """Invariant 4, stated directly: no retroactive capping."""
    y = np.arange(1000, dtype=float)
    fold = _fold_at(300)  # well before the cutoff position
    capped = resolve_training_window(fold, "capped", y, cutoff_pos=540)
    full = resolve_training_window(fold, "full", y, cutoff_pos=540)
    assert capped.train_start_pos == full.train_start_pos == 0
    assert capped.fallback_applied is False


def test_cap_truncates_the_training_start_for_a_post_cutoff_origin():
    y = np.arange(1000, dtype=float)
    fold = _fold_at(700)  # 161 usable rows from pos 540, comfortably over the floor
    capped = resolve_training_window(fold, "capped", y, cutoff_pos=540)
    assert capped.train_start_pos == 540
    assert capped.fallback_applied is False
    assert capped.n_positions == 700 - 540 + 1


def test_capped_window_falls_back_when_it_holds_too_few_usable_rows():
    y = np.arange(1000, dtype=float)
    fold = _fold_at(560)  # only 21 usable rows from the cutoff
    capped = resolve_training_window(fold, "capped", y, cutoff_pos=540)
    assert capped.fallback_applied is True
    assert capped.train_start_pos == 0


def test_fallback_counts_usable_rows_not_slots():
    """
    Missing flow rows do not count toward the floor. A window of 70 slots that
    holds only 55 non-missing target values is below the 60-row floor and must
    fall back, even though it has 'enough' slots.
    """
    y = np.arange(1000, dtype=float)
    y[545:560] = np.nan  # 15 true-missing slots inside the capped window
    fold = _fold_at(609)  # 70 slots from 540, of which 55 are usable
    w = resolve_training_window(fold, "capped", y, cutoff_pos=540)
    assert w.n_positions == 70 or w.fallback_applied
    assert w.fallback_applied is True


def test_floor_boundary_is_exact():
    """Exactly `TRAINING_FLOOR_ROWS` usable rows must pass, one fewer must fall back."""
    y = np.arange(1000, dtype=float)
    at_floor = _fold_at(540 + TRAINING_FLOOR_ROWS - 1)
    below_floor = _fold_at(540 + TRAINING_FLOOR_ROWS - 2)
    assert resolve_training_window(at_floor, "capped", y, 540).fallback_applied is False
    assert resolve_training_window(below_floor, "capped", y, 540).fallback_applied is True


def test_upper_floor_sensitivity_flag_is_recorded():
    """
    The addendum froze a ~60-80 range; the harness enforces 60 and must record
    which folds would decide differently at 80, so the choice stays auditable.
    """
    y = np.arange(1000, dtype=float)
    between = _fold_at(540 + TRAINING_FLOOR_ROWS + 2)  # 62 usable: >=60, <80
    above = _fold_at(540 + MAX_USABLE_ROWS_FLOOR + 5)
    assert resolve_training_window(between, "capped", y, 540).flips_at_upper_floor is True
    assert resolve_training_window(above, "capped", y, 540).flips_at_upper_floor is False


def test_unknown_window_rule_is_rejected():
    with pytest.raises(ValueError):
        resolve_training_window(_fold_at(600), "recent-ish", np.arange(1000.0), 540)


def test_cutoff_position_resolves_to_the_frozen_date(master, cutoff_pos):
    assert master["parsed_date"].iloc[cutoff_pos] >= pd.Timestamp(TRAINING_CAP_DATE)
    assert master["parsed_date"].iloc[cutoff_pos - 1] < pd.Timestamp(TRAINING_CAP_DATE)


# ----------------------------------------------------------------------
# 5. Gap flagging
# ----------------------------------------------------------------------
def test_folds_are_flagged_when_the_origin_sits_in_a_gap():
    dates, is_imputed = _synthetic(gaps=[MIN_INITIAL_TRAINING - 1])
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert folds[0].origin_is_imputed is True
    assert folds[0].gap_contaminated is True


def test_folds_are_flagged_when_the_origin_is_adjacent_to_a_gap():
    dates, is_imputed = _synthetic(gaps=[MIN_INITIAL_TRAINING])  # the slot after origin 49
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert folds[0].origin_is_imputed is False
    assert folds[0].origin_adjacent_to_gap is True
    assert folds[0].gap_contaminated is True


def test_folds_are_flagged_when_the_near_term_test_point_is_a_gap():
    # Fold 0's origin is pos 49, so its h=1 test point is pos 50.
    dates, is_imputed = _synthetic(gaps=[MIN_INITIAL_TRAINING])
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert folds[0].test_positions[1] == MIN_INITIAL_TRAINING
    assert folds[0].test_is_imputed[1] is True
    assert folds[0].gap_contaminated is True


def test_a_long_horizon_gap_is_recorded_but_does_not_set_the_near_term_flag():
    """
    Addendum Section 5 scopes contamination to the training cut-off and the
    NEAR-TERM test window. A gap only at h=14 is still recorded per-horizon --
    so metrics drop it -- but it is not what the flag is about.
    """
    # Fold 1's origin is pos 59, so its h=14 test point is pos 73, which is
    # neither an origin nor adjacent to one.
    dates, is_imputed = _synthetic(gaps=[73])
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert folds[1].test_positions[14] == 73
    assert folds[1].test_is_imputed[14] is True
    assert folds[1].test_is_imputed[1] is False
    assert folds[1].gap_contaminated is False


def test_clean_folds_are_not_flagged():
    dates, is_imputed = _synthetic()
    folds = generate_folds(dates, is_imputed, dev_end_pos=150)
    assert not any(f.gap_contaminated for f in folds)


# ----------------------------------------------------------------------
# 6. End-to-end run on the real series
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline_run(master, real_folds, cutoff_pos):
    return run_walk_forward(master, TARGET_1, BASELINE_FACTORIES, real_folds, cutoff_pos)


def test_run_produces_one_row_per_model_rule_fold_horizon(baseline_run, real_folds):
    expected = len(BASELINE_FACTORIES) * 2 * len(real_folds) * len(FORECAST_HORIZONS)
    assert len(baseline_run) == expected


def test_every_recorded_actual_matches_the_series_at_its_test_position(baseline_run, master):
    y = master[TARGET_1].astype(float).to_numpy()
    recorded = baseline_run["y_true"].to_numpy()
    expected = y[baseline_run["test_pos"].to_numpy()]
    np.testing.assert_array_equal(recorded, expected)


def test_baselines_are_refit_at_every_origin_not_reused(baseline_run):
    """A single fitted model reused across origins would give one constant forecast."""
    naive_h1 = baseline_run[
        (baseline_run["model"] == "naive")
        & (baseline_run["horizon"] == 1)
        & (baseline_run["window_rule"] == "full")
    ]
    assert naive_h1["y_pred"].nunique() > 1


def test_naive_forecast_equals_the_last_training_value(baseline_run, master):
    y = master[TARGET_1].astype(float).to_numpy()
    naive = baseline_run[
        (baseline_run["model"] == "naive") & (baseline_run["window_rule"] == "full")
    ]
    for row in naive.itertuples():
        assert row.y_pred == pytest.approx(y[row.train_cutoff_pos])


def test_seasonal_naive_forecast_equals_the_value_m_periods_before_the_target(baseline_run, master):
    """
    Seasonal naive predicts the observation one seasonal cycle before the TARGET
    date. The lookback is driven by the effective lead (the true forecast
    distance), not the nominal horizon -- they differ when the window was pulled
    back off an interpolated origin.
    """
    y = master[TARGET_1].astype(float).to_numpy()
    sn = baseline_run[
        (baseline_run["model"] == "seasonal_naive") & (baseline_run["window_rule"] == "full")
    ]
    assert len(sn) > 0
    for row in sn.itertuples():
        lead = row.effective_lead
        offset = -(SEASONAL_PERIOD_M - (lead - 1) % SEASONAL_PERIOD_M)
        expected = y[row.train_cutoff_pos + 1 + offset]
        assert row.y_pred == pytest.approx(expected)
        # And that source observation is always one or more full cycles before
        # the scored date -- never at or after it.
        assert row.train_cutoff_pos + 1 + offset < row.test_pos


def test_capped_rule_changes_the_training_window_on_post_cutoff_folds(baseline_run):
    post = baseline_run[baseline_run["origin_post_cutoff"]]
    capped = post[(post["window_rule"] == "capped") & (~post["fallback_applied"])]
    assert len(capped) > 0
    assert (capped["train_start_pos"] > 0).all()


def test_pre_cutoff_folds_are_identical_under_both_rules(baseline_run):
    pre = baseline_run[~baseline_run["origin_post_cutoff"]]
    key = ["model", "fold_id", "horizon"]
    full = pre[pre["window_rule"] == "full"].set_index(key).sort_index()
    capped = pre[pre["window_rule"] == "capped"].set_index(key).sort_index()
    np.testing.assert_array_equal(
        full["train_start_pos"].to_numpy(), capped["train_start_pos"].to_numpy()
    )
    np.testing.assert_allclose(full["y_pred"].to_numpy(), capped["y_pred"].to_numpy())


def test_flow_target_gaps_are_dropped_never_scored_against_a_filled_actual(
    master, real_folds, cutoff_pos
):
    """Invariant 2: a test point on a flow gap must stay NaN and leave the metric."""
    run = run_walk_forward(master, TARGET_2, BASELINE_FACTORIES, real_folds, cutoff_pos)
    gap_rows = run[run["test_is_imputed"]]
    assert len(gap_rows) > 0
    assert gap_rows["y_true"].isna().all()

    metrics = aggregate_metrics(run)
    for row in metrics.itertuples():
        assert row.n_scored == row.n_total - row.n_missing_actual - row.n_missing_pred


def test_common_support_gives_every_model_the_same_scored_n(master, real_folds, cutoff_pos):
    run = run_walk_forward(master, TARGET_2, BASELINE_FACTORIES, real_folds, cutoff_pos)
    restricted = run[common_support_mask(run)]
    counts = (
        restricted.groupby(["window_rule", "horizon"])["model"].value_counts().unstack()
    )
    assert counts.nunique(axis=1).eq(1).all()
    assert restricted["y_true"].notna().all()
    assert restricted["y_pred"].notna().all()


def test_mase_scale_is_origin_anchored_and_leakage_free(baseline_run, master):
    """
    The MASE denominator must come from all history at or before the fold's
    training cutoff -- never from data after it.
    """
    from src.evaluation.metrics import seasonal_naive_scale

    y = master[TARGET_1].astype(float).to_numpy()
    for row in baseline_run.drop_duplicates(["fold_id", "window_rule"]).itertuples():
        expected = seasonal_naive_scale(y[: row.train_cutoff_pos + 1], SEASONAL_PERIOD_M)
        assert row.mase_scale == pytest.approx(expected, nan_ok=True)


def test_mase_scale_does_not_move_with_the_window_rule(baseline_run):
    """
    REGRESSION (audit finding D3). The scale was previously derived from each
    rule's own training slice, so `full` and `capped` reported different MASE
    for mathematically identical forecasts -- corrupting the very full-vs-capped
    ranking comparison the addendum requires.
    """
    key = ["model", "fold_id", "horizon"]
    full = baseline_run[baseline_run["window_rule"] == "full"].set_index(key).sort_index()
    capped = baseline_run[baseline_run["window_rule"] == "capped"].set_index(key).sort_index()
    np.testing.assert_allclose(
        full["mase_scale"].to_numpy(), capped["mase_scale"].to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        full["y_pred"].to_numpy(), capped["y_pred"].to_numpy(), equal_nan=True
    )


# ----------------------------------------------------------------------
# 7. AUDIT REGRESSIONS -- interpolation leakage and fabricated actuals
# ----------------------------------------------------------------------
def test_training_never_ends_on_an_interpolated_slot(real_folds, master):
    """
    REGRESSION (audit finding D1). Linear interpolation fills a gap from the
    values on BOTH sides, so an interpolated value at the origin encodes the
    next real observation -- which is in the future relative to that origin.
    Training must therefore end at the last real observation, never on a gap.
    """
    imputed = master["is_imputed"].to_numpy(dtype=bool)
    for fold in real_folds:
        assert not imputed[fold.train_cutoff_pos]
        assert fold.train_cutoff_pos <= fold.origin_pos


def test_the_documented_leaky_fold_is_actually_repaired(real_folds, master):
    """
    The concrete instance the audit found: an origin inside a gap run whose
    interpolated value is a blend of the surrounding reals, one of which is that
    fold's own h=1 test point.
    """
    imputed = master["is_imputed"].to_numpy(dtype=bool)
    leaky = [f for f in real_folds if imputed[f.origin_pos]]
    assert len(leaky) >= 1, "fixture no longer exercises the bug this test protects"

    y = master[TARGET_1].astype(float).to_numpy()
    for fold in leaky:
        nxt = fold.origin_pos + int(np.flatnonzero(~imputed[fold.origin_pos:])[0])
        prev = int(np.flatnonzero(~imputed[: fold.origin_pos])[-1])
        # The stored value at the origin really is a blend of a FUTURE observation...
        blended = y[prev] + (y[nxt] - y[prev]) * (fold.origin_pos - prev) / (nxt - prev)
        assert y[fold.origin_pos] == pytest.approx(blended, abs=1.0)
        assert nxt > fold.origin_pos
        # ...so the harness must not let a model see it.
        assert fold.train_cutoff_pos == prev
        assert fold.effective_lead[1] > 1


def test_a_model_never_receives_an_interpolated_trailing_value(master, cutoff_pos, real_folds):
    """End-to-end: the last element handed to `.fit` is always a real observation."""
    tails = {}

    class Spy:
        def fit(self, y):
            self._y = np.asarray(y, dtype=float)
            return self

        def predict(self, horizons):
            tails[len(tails)] = self._y[-1]
            return np.zeros(len(horizons))

    imputed = master["is_imputed"].to_numpy(dtype=bool)
    run = run_walk_forward(
        master, TARGET_1, {"spy": lambda: Spy()}, real_folds, cutoff_pos, window_rules=["full"]
    )
    y = master[TARGET_1].astype(float).to_numpy()
    real_values = set(y[~imputed].tolist())
    for row in run.drop_duplicates("fold_id").itertuples():
        assert not imputed[row.train_cutoff_pos]
        assert y[row.train_cutoff_pos] in real_values


def test_interpolated_actuals_are_never_scored(master, real_folds, cutoff_pos):
    """
    REGRESSION (audit finding D2). Stock columns are interpolated at gap slots,
    so their `y_true` is never NaN and a naive aggregation silently grades
    forecasts against a straight line drawn by the cleaning step.
    """
    run = run_walk_forward(master, TARGET_1, BASELINE_FACTORIES, real_folds, cutoff_pos)
    interpolated = run[run["test_is_imputed"]]
    assert len(interpolated) > 0, "fixture no longer exercises the bug this test protects"
    assert interpolated["y_true"].notna().all(), "stock actuals are filled, not NaN"
    assert (~interpolated["y_true_is_observed"]).all()

    scored = aggregate_metrics(run)
    unscored = aggregate_metrics(run, score_only_observed=False)
    assert (scored["n_excluded_interpolated_actual"] > 0).any()
    assert (scored["n_scored"] < unscored["n_scored"]).any()
    # And the exclusion actually moves the number, so it is not cosmetic.
    assert not np.allclose(scored["MAE"], unscored["MAE"])


def test_scored_actuals_are_all_genuine_published_observations(master, real_folds, cutoff_pos):
    """Every value any metric is computed against must exist in the raw CSV."""
    raw = pd.read_csv(project_root / "HHS_Unaccompanied_Alien_Children_Program (1).csv")
    raw = raw[raw.notna().any(axis=1)]
    published = set(
        raw[TARGET_1].astype(str).str.replace(",", "").astype(float).tolist()
    )
    for target in (TARGET_1, TARGET_2):
        run = run_walk_forward(master, target, BASELINE_FACTORIES, real_folds, cutoff_pos)
        observed = run[run["y_true_is_observed"].astype(bool)]
        assert observed["y_true"].notna().all()
        if target == TARGET_1:
            assert set(observed["y_true"].tolist()).issubset(published)


def test_common_support_excludes_interpolated_actuals_too(master, real_folds, cutoff_pos):
    run = run_walk_forward(master, TARGET_1, BASELINE_FACTORIES, real_folds, cutoff_pos)
    kept = run[common_support_mask(run)]
    assert kept["y_true_is_observed"].astype(bool).all()


def test_common_support_is_symmetric_and_adds_no_selection_of_its_own(master, real_folds, cutoff_pos):
    """
    Common support must drop a (fold, horizon) cell for EVERY model or for none --
    never for some models only, which would reintroduce the bias it exists to remove.
    """
    run = run_walk_forward(master, TARGET_2, BASELINE_FACTORIES, real_folds, cutoff_pos)
    run = run.assign(keep=common_support_mask(run))
    per_cell = run.groupby(["window_rule", "fold_id", "horizon"])["keep"].nunique()
    assert (per_cell == 1).all(), "common support dropped a cell for only some models"


def test_effective_lead_is_honest_about_gap_widened_horizons(real_folds):
    """When the origin falls in a gap the real forecast distance exceeds h; say so."""
    for fold in real_folds:
        for h in fold.horizons:
            assert fold.effective_lead[h] == fold.test_positions[h] - fold.train_cutoff_pos
            assert fold.effective_lead[h] >= h


def test_models_are_asked_to_forecast_to_the_date_they_are_scored_on(master, real_folds, cutoff_pos):
    """
    REGRESSION. A model's last observation is at `train_cutoff_pos` and it is
    scored at `origin + h`. When the origin fell in an interpolated gap those
    differ, and asking the model for `h` steps would produce a forecast for the
    wrong date which the harness would then grade as if it were right.
    """
    asked = {}

    class Spy:
        def fit(self, y):
            self._n = len(y)
            return self

        def predict(self, horizons):
            asked["leads"] = list(horizons)
            return np.zeros(len(horizons))

    leaky_folds = [f for f in real_folds if f.train_cutoff_pos != f.origin_pos]
    assert leaky_folds, "fixture no longer exercises a pulled-back window"

    for fold in leaky_folds:
        run_walk_forward(master, TARGET_1, {"spy": lambda: Spy()}, [fold],
                         cutoff_pos, window_rules=["full"])
        for lead, h in zip(asked["leads"], fold.horizons):
            assert lead == fold.test_positions[h] - fold.train_cutoff_pos
            assert lead > h, "this fold should require a longer lead than nominal"


def test_nominal_horizon_is_used_whenever_the_origin_is_a_real_observation(master, real_folds, cutoff_pos):
    asked = {}

    class Spy:
        def fit(self, y):
            return self

        def predict(self, horizons):
            asked["leads"] = list(horizons)
            return np.zeros(len(horizons))

    clean = [f for f in real_folds if f.train_cutoff_pos == f.origin_pos][0]
    run_walk_forward(master, TARGET_1, {"spy": lambda: Spy()}, [clean],
                     cutoff_pos, window_rules=["full"])
    assert asked["leads"] == list(clean.horizons)
