"""
walk_forward.py -- The single walk-forward validation harness.

This is the one harness every subsequent day reuses: Day 6 (SARIMAX / ETSModel),
Day 7 (RandomForest / HistGradientBoosting) and Day 8 (ensemble + champion
selection) all run through `run_walk_forward` on the *identical* folds produced
here, so every candidate is compared on exactly the same evidence.

Protocol implemented (PRE_BUILD_TECHNICAL_ADDENDUM Section 5, frozen):

  * Final held-out test window  : the most recent FINAL_TEST_WINDOW (60) REAL
                                  observations. Reserved, never folded over.
  * Development portion         : everything before it.
  * Folds                       : expanding-window, chronological, origins
                                  spaced WALK_FORWARD_STEP (10) period-positions
                                  apart, first origin at MIN_INITIAL_TRAINING
                                  (50) positions of history.
  * Horizons                    : FORECAST_HORIZONS {1, 7, 14}, as integer
                                  period-position offsets -- never calendar-day
                                  arithmetic (invariant 1).
  * Window rules                : every fold is evaluated twice, under the
                                  full-expanding rule and under the
                                  2025-02-05-capped rule. The two are identical
                                  by construction for folds whose origin
                                  precedes the cutoff -- the cap is never
                                  applied retroactively (invariant 4).
  * Fallback floor              : a capped training window holding fewer than
                                  TRAINING_FLOOR_ROWS usable (non-missing)
                                  target rows falls back to the full expanding
                                  window for that fold/target.

Leakage guarantees, asserted at construction time and again in tests:
  * every fold satisfies max(train_position) < min(test_position);
  * no test position ever lands in the held-out window;
  * a model only ever sees y[train_start .. train_cutoff] inclusive.

  * INTERPOLATION LEAKAGE. Stock columns are linearly interpolated at gap slots
    as a one-time global step before folding (addendum Section 2). Linear
    interpolation fills a gap from the surrounding valid values, so an
    interpolated value at position p encodes the next REAL observation after p.
    If a fold's origin lands on a gap slot, y[origin] is therefore a function of
    an observation that occurs AFTER the origin -- future information, inside
    the training window.

    Measured on the real series: fold origin 619 (2025-05-26) sits in a gap run,
    and y[619] is a linear blend of y[617] and y[620] -- where position 620 is
    that same fold's h=1 test point. A naive forecast from that origin is
    literally a function of the value it is scored against.

    The harness therefore ends every training window at `train_cutoff_pos`, the
    last REAL observation at or before the origin, never at the origin itself.
    This is also what a real-time forecaster actually has: on a date with no
    published report, you forecast from the last published figure.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    MAX_USABLE_ROWS_FLOOR,
    MIN_INITIAL_TRAINING,
    SEASONAL_PERIOD_M,
    TRAINING_FLOOR_ROWS,
    WALK_FORWARD_STEP,
    WINDOW_RULES,
)
from src.evaluation.metrics import (  # noqa: E402
    compute_all_metrics,
    pooled_mase,
    seasonal_naive_scale,
)

__all__ = [
    "Fold",
    "TrainingWindow",
    "resolve_split_boundaries",
    "generate_folds",
    "resolve_training_window",
    "run_walk_forward",
    "common_support_mask",
    "aggregate_metrics",
]


# ----------------------------------------------------------------------
# Fold construction
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One rolling forecast origin and the test positions it is scored on."""

    fold_id: int
    origin_pos: int
    origin_date: pd.Timestamp
    horizons: tuple
    test_positions: dict
    origin_is_imputed: bool
    origin_adjacent_to_gap: bool
    # Last REAL observation at or before the origin. Training always ends here,
    # never at an interpolated origin -- see the interpolation-leakage note in
    # the module docstring. Equal to origin_pos for every fold with a real origin.
    train_cutoff_pos: int = -1
    test_is_imputed: dict = field(default_factory=dict)

    @property
    def max_test_pos(self) -> int:
        return max(self.test_positions.values())

    @property
    def effective_lead(self) -> dict:
        """
        Real forecast distance per horizon: test position minus the last actually
        observed training position. Equals the nominal horizon whenever the origin
        is a real observation, and is larger when the origin fell in a gap.
        """
        return {h: p - self.train_cutoff_pos for h, p in self.test_positions.items()}

    @property
    def gap_contaminated(self) -> bool:
        """
        True when the fold's training cut-off, or its near-term (shortest-horizon)
        test point, sits in or next to an interpolated gap. Addendum Section 5
        requires these folds to be flagged and CHECKED at Day 5 rather than
        assumed negligible.
        """
        near_h = min(self.horizons)
        return bool(
            self.origin_is_imputed
            or self.origin_adjacent_to_gap
            or self.test_is_imputed.get(near_h, False)
        )


@dataclass(frozen=True)
class TrainingWindow:
    """The slice of history a model may see at one fold, under one window rule."""

    rule: str
    train_start_pos: int
    train_end_pos: int  # inclusive: the origin itself
    n_positions: int
    n_usable_rows: int
    fallback_applied: bool
    flips_at_upper_floor: bool


def resolve_split_boundaries(is_imputed, final_test_window: int = FINAL_TEST_WINDOW):
    """
    Split the period-position index into development and final-holdout portions.

    The holdout is defined in REAL observations (addendum: "the most recent 60
    real observations"), so imputed gap slots inside the tail cannot silently
    shrink it. Returns (holdout_start_pos, dev_end_pos), each the inclusive
    bound of its own side.
    """
    is_imputed = np.asarray(is_imputed, dtype=bool)
    real_positions = np.flatnonzero(~is_imputed)
    if real_positions.size < final_test_window:
        raise ValueError(
            "only %d real observations, cannot reserve a %d-observation holdout"
            % (real_positions.size, final_test_window)
        )
    holdout_start_pos = int(real_positions[-final_test_window])
    return holdout_start_pos, holdout_start_pos - 1


def generate_folds(
    dates,
    is_imputed,
    dev_end_pos: int,
    horizons: Iterable = FORECAST_HORIZONS,
    min_initial_training: int = MIN_INITIAL_TRAINING,
    step: int = WALK_FORWARD_STEP,
):
    """
    Expanding-window, rolling-origin folds over the development portion.

    Origins are placed on the period-position index at a FIXED cadence -- the
    first origin after `min_initial_training` positions of history, then every
    `step` positions -- and a fold is kept only if its longest horizon still
    lands inside the development portion, so the holdout stays untouched.
    """
    dates = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    is_imputed = np.asarray(is_imputed, dtype=bool)
    horizons = tuple(sorted(int(h) for h in horizons))
    max_h = max(horizons)

    folds = []
    origin = min_initial_training - 1  # 0-indexed: 50 positions of history == pos 49
    fold_id = 0
    while origin + max_h <= dev_end_pos:
        test_positions = {h: origin + h for h in horizons}
        prev_gap = bool(is_imputed[origin - 1]) if origin - 1 >= 0 else False
        next_gap = bool(is_imputed[origin + 1]) if origin + 1 < is_imputed.size else False

        real_at_or_before = np.flatnonzero(~is_imputed[: origin + 1])
        if real_at_or_before.size == 0:
            raise ValueError(
                "fold origin %d has no real observation at or before it" % origin
            )
        train_cutoff = int(real_at_or_before[-1])

        folds.append(
            Fold(
                fold_id=fold_id,
                origin_pos=origin,
                origin_date=dates.iloc[origin],
                horizons=horizons,
                test_positions=test_positions,
                origin_is_imputed=bool(is_imputed[origin]),
                origin_adjacent_to_gap=prev_gap or next_gap,
                train_cutoff_pos=train_cutoff,
                test_is_imputed={h: bool(is_imputed[p]) for h, p in test_positions.items()},
            )
        )
        fold_id += 1
        origin += step

    # Leakage guarantees -- asserted here so a malformed fold can never reach a model.
    for fold in folds:
        assert fold.origin_pos < min(fold.test_positions.values()), (
            "fold %d: training data is not strictly before its test data" % fold.fold_id
        )
        assert fold.max_test_pos <= dev_end_pos, (
            "fold %d: test position %d enters the held-out window"
            % (fold.fold_id, fold.max_test_pos)
        )
        assert fold.train_cutoff_pos <= fold.origin_pos, (
            "fold %d: training cutoff is after its own origin" % fold.fold_id
        )
        assert not is_imputed[fold.train_cutoff_pos], (
            "fold %d: training ends on an interpolated slot, which encodes a "
            "future observation" % fold.fold_id
        )
    return folds


def resolve_training_window(
    fold: Fold,
    rule: str,
    y,
    cutoff_pos: int,
    floor: int = TRAINING_FLOOR_ROWS,
    upper_floor: int = MAX_USABLE_ROWS_FLOOR,
) -> TrainingWindow:
    """
    The training slice for one fold under one window rule.

    `full`   -- expanding window from the very start of the series.
    `capped` -- starts no earlier than the 2025-02-05 regime cutoff, but ONLY
                for folds whose origin is on or after that cutoff. For an
                earlier origin the cap is not applied at all (invariant 4), so
                the capped window is identical to the full one.

    A capped window holding fewer than `floor` usable (non-missing) target rows
    falls back to the full expanding window. `flips_at_upper_floor` records
    whether that decision would differ at the top of the addendum's ~60-80 range,
    so the choice made inside that range stays auditable instead of silent.

    Every window ends at `fold.train_cutoff_pos` -- the last REAL observation at
    or before the origin -- NOT at the origin itself. An interpolated value at
    the origin is a linear blend of its neighbours and therefore encodes the
    next real observation, which lies after the origin; training on it would
    leak the future. See the module docstring for the measured instance.
    """
    if rule not in WINDOW_RULES:
        raise ValueError("unknown window rule %r; expected one of %s" % (rule, WINDOW_RULES))

    y = np.asarray(y, dtype=float)
    end = fold.train_cutoff_pos  # inclusive; never an interpolated slot

    if rule == "full" or fold.origin_pos < cutoff_pos:
        usable = int(np.isfinite(y[0 : end + 1]).sum())
        return TrainingWindow(
            rule=rule,
            train_start_pos=0,
            train_end_pos=end,
            n_positions=end + 1,
            n_usable_rows=usable,
            fallback_applied=False,
            flips_at_upper_floor=False,
        )

    capped_usable = int(np.isfinite(y[cutoff_pos : end + 1]).sum())
    fallback = capped_usable < floor
    flips = (capped_usable >= floor) and (capped_usable < upper_floor)

    if fallback:
        full_usable = int(np.isfinite(y[0 : end + 1]).sum())
        return TrainingWindow(
            rule=rule,
            train_start_pos=0,
            train_end_pos=end,
            n_positions=end + 1,
            n_usable_rows=full_usable,
            fallback_applied=True,
            flips_at_upper_floor=flips,
        )

    return TrainingWindow(
        rule=rule,
        train_start_pos=cutoff_pos,
        train_end_pos=end,
        n_positions=end - cutoff_pos + 1,
        n_usable_rows=capped_usable,
        fallback_applied=False,
        flips_at_upper_floor=flips,
    )


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def run_walk_forward(
    df: pd.DataFrame,
    target_col: str,
    model_factories: dict,
    folds: Sequence,
    cutoff_pos: int,
    date_col: str = "parsed_date",
    seasonal_m: int = SEASONAL_PERIOD_M,
    window_rules: Sequence = WINDOW_RULES,
) -> pd.DataFrame:
    """
    Fit every model at every fold origin under every window rule, and return one
    tidy row per (model, window_rule, fold, horizon).

    `model_factories` maps a model name to a zero-argument callable returning a
    FRESH estimator exposing `.fit(y) -> self` and `.predict(horizons) -> array`.
    A fresh instance per fold is what makes this a genuine refit-at-each-origin
    harness rather than one model reused across origins.

    Nothing is imputed, clipped or filled here. A missing actual or a missing
    forecast is carried through as NaN and dropped at metric time, with the
    count reported.
    """
    y = df[target_col].astype(float).to_numpy()
    dates = pd.to_datetime(df[date_col]).reset_index(drop=True)

    is_imputed = df["is_imputed"].to_numpy(dtype=bool)

    # MASE denominator, computed ONCE per fold from all history available at the
    # fold's training cutoff, and deliberately INDEPENDENT of the window rule.
    #
    # Deriving it from each rule's own training slice made MASE incomparable
    # across rules: on the real series the baselines' point forecasts are
    # identical under `full` and `capped`, yet their MASE differed purely because
    # the denominator moved (e.g. HHS Care naive h=1 post-cutoff: MAE 8.73 under
    # both rules, MASE 0.032 vs 0.070). Addendum Section 5 requires the full and
    # capped rankings to be COMPARED, so a denominator that shifts with the rule
    # would corrupt exactly the comparison it feeds. A fold-level scale keeps
    # MASE a normaliser for series difficulty at that origin -- shared by every
    # candidate -- and uses only data at or before the cutoff, so it cannot leak.
    fold_scales = {
        fold.fold_id: seasonal_naive_scale(y[: fold.train_cutoff_pos + 1], seasonal_m)
        for fold in folds
    }

    rows = []
    for rule in window_rules:
        for fold in folds:
            window = resolve_training_window(fold, rule, y, cutoff_pos)
            y_train = y[window.train_start_pos : window.train_end_pos + 1]
            scale = fold_scales[fold.fold_id]

            for model_name, factory in model_factories.items():
                model = factory()
                with warnings.catch_warnings():
                    # An all-missing tail in a flow target legitimately yields a
                    # NaN forecast; that is recorded, not silenced into a number.
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    model.fit(y_train)
                    preds = np.asarray(model.predict(list(fold.horizons)), dtype=float)

                for h, y_hat in zip(fold.horizons, preds):
                    test_pos = fold.test_positions[h]
                    test_imputed = fold.test_is_imputed[h]
                    test_adjacent = (
                        (test_pos - 1 >= 0 and bool(is_imputed[test_pos - 1]))
                        or (test_pos + 1 < is_imputed.size and bool(is_imputed[test_pos + 1]))
                    )
                    rows.append(
                        {
                            "target": target_col,
                            "model": model_name,
                            "window_rule": rule,
                            "fold_id": fold.fold_id,
                            "origin_pos": fold.origin_pos,
                            "origin_date": fold.origin_date,
                            "origin_post_cutoff": fold.origin_pos >= cutoff_pos,
                            "horizon": h,
                            "effective_lead": test_pos - fold.train_cutoff_pos,
                            "test_pos": test_pos,
                            "test_date": dates.iloc[test_pos],
                            "y_true": y[test_pos],
                            # An interpolated slot carries a value that was
                            # invented by `clean.py`, not observed. Scoring a
                            # forecast against it measures how well the model
                            # reproduces a straight line, so it is excluded from
                            # every metric (see `aggregate_metrics`).
                            "y_true_is_observed": (not test_imputed) and bool(np.isfinite(y[test_pos])),
                            "y_pred": y_hat,
                            "train_start_pos": window.train_start_pos,
                            "train_cutoff_pos": window.train_end_pos,
                            "train_n_positions": window.n_positions,
                            "train_n_usable": window.n_usable_rows,
                            "fallback_applied": window.fallback_applied,
                            "flips_at_upper_floor": window.flips_at_upper_floor,
                            "mase_scale": scale,
                            "origin_is_imputed": fold.origin_is_imputed,
                            "origin_adjacent_to_gap": fold.origin_adjacent_to_gap,
                            "test_is_imputed": test_imputed,
                            "test_adjacent_to_gap": test_adjacent,
                            # Fold-level flag, matching the addendum's wording
                            # ("training cut-off or NEAR-TERM test window").
                            "gap_contaminated": fold.gap_contaminated,
                            # Row-level flag: is THIS horizon's scored point
                            # touched by interpolation on either side?
                            "row_gap_adjacent": bool(
                                fold.origin_is_imputed
                                or fold.origin_adjacent_to_gap
                                or test_imputed
                                or test_adjacent
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def common_support_mask(
    predictions: pd.DataFrame,
    key_cols: Sequence = ("target", "window_rule", "fold_id", "horizon"),
) -> pd.Series:
    """
    Boolean mask keeping only the test points that EVERY model in the table can
    be scored on -- the actual exists and no model returned a NaN forecast.

    This exists because models legitimately abstain at different points. The
    seasonal-naive baseline, for instance, looks back exactly m periods, and on
    a flow target that lookback sometimes lands on a true-missing gap slot, so
    it produces fewer forecasts than naive on the same folds. Comparing raw MAE
    across models scored on different subsets is not a like-for-like comparison,
    and the Day-8 champion rule needs a like-for-like one. Per-model N is still
    reported separately -- this restricts the comparison, it does not hide the
    abstentions.
    """
    key_cols = list(key_cols)
    scorable = (
        predictions["y_true"].notna()
        & predictions["y_pred"].notna()
        & predictions["y_true_is_observed"].astype(bool)
    )
    return (
        predictions.assign(_scorable=scorable)
        .groupby(key_cols, dropna=False)["_scorable"]
        .transform("all")
        .astype(bool)
    )


def aggregate_metrics(
    predictions: pd.DataFrame,
    group_cols: Sequence = ("target", "model", "window_rule", "horizon"),
    score_only_observed: bool = True,
) -> pd.DataFrame:
    """
    Collapse the tidy prediction table into the metrics matrix.

    `score_only_observed` (default True) drops test points whose actual is an
    interpolated value rather than a published observation. This matters only
    for the STOCK targets: `clean.py` interpolates them at gap slots, so their
    `y_true` is never NaN and a naive aggregation silently grades forecasts
    against a straight line the cleaning step drew. On the real series 10 of the
    195 test points per target land on interpolated slots, and including them
    moved HHS Care's h=7 naive MAE by -4.4%. The flow targets are unaffected
    because their gaps stay true-missing (invariant 2) and drop out as NaN.

    MASE is pooled using each fold's own scale (see `metrics.pooled_mase`);
    every other metric comes from `metrics.compute_all_metrics`, which reports
    the number of pairs actually scored alongside each value.
    """
    group_cols = list(group_cols)
    out = []
    for keys, grp in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))

        observed = grp["y_true_is_observed"].astype(bool)
        n_interp = int((~observed & grp["y_true"].notna()).sum())
        scored = grp[observed] if score_only_observed else grp

        stats = compute_all_metrics(scored["y_true"].to_numpy(), scored["y_pred"].to_numpy())
        stats["MASE"] = pooled_mase(
            scored["y_true"].to_numpy(),
            scored["y_pred"].to_numpy(),
            scored["mase_scale"].to_numpy(),
        )
        record.update(stats)
        record["n_excluded_interpolated_actual"] = n_interp
        record["n_folds"] = int(grp["fold_id"].nunique())
        record["n_gap_contaminated_folds"] = int(
            grp.loc[grp["gap_contaminated"], "fold_id"].nunique()
        )
        out.append(record)
    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)
