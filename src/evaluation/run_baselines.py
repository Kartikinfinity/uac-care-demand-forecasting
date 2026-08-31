"""
run_baselines.py -- Day 5 driver.

Runs the three Day-4 baselines (naive persistence, seasonal-naive m=5, moving
average w=7) through the Day-5 walk-forward harness, for both targets, on the
EXACT folds every later model will be scored on, and writes:

  forecasts/walk_forward_folds.csv    fold manifest (origins, test positions, flags)
  forecasts/baseline_predictions.csv  one row per model/rule/fold/horizon
  forecasts/baseline_metrics.csv      the metrics matrix
  docs/day5_baseline_metrics.md       the human-readable logged baseline table

This is the floor every statistical and ML model must beat (roadmap Part 6,
"baseline-beating gate"). Nothing here is fitted at dashboard-serve time; this
is an offline batch step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    BASELINE_METRICS_PATH,
    BASELINE_METRICS_REPORT_PATH,
    BASELINE_PREDICTIONS_PATH,
    FINAL_TEST_WINDOW,
    FOLD_MANIFEST_PATH,
    FORECAST_HORIZONS,
    MASTER_SERIES_PATH,
    MAX_USABLE_ROWS_FLOOR,
    MIN_INITIAL_TRAINING,
    MOVING_AVERAGE_WINDOW,
    SEASONAL_PERIOD_M,
    TARGET_1,
    TARGET_2,
    TRAINING_CAP_DATE,
    TRAINING_FLOOR_ROWS,
    WALK_FORWARD_STEP,
)
from src.models.baselines import (  # noqa: E402
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalNaiveBaseline,
)
from src.evaluation.walk_forward import (  # noqa: E402
    aggregate_metrics,
    common_support_mask,
    generate_folds,
    resolve_split_boundaries,
    run_walk_forward,
)

BASELINE_FACTORIES = {
    "naive": lambda: NaiveBaseline(),
    "seasonal_naive": lambda: SeasonalNaiveBaseline(m=SEASONAL_PERIOD_M),
    "moving_average": lambda: MovingAverageBaseline(w=MOVING_AVERAGE_WINDOW),
}

TARGETS = [TARGET_1, TARGET_2]

METRIC_COLS = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "ME_bias"]


def resolve_cutoff_pos(dates: pd.Series, cap_date: str = TRAINING_CAP_DATE) -> int:
    """First period-position on or after the frozen 2025-02-05 regime cutoff."""
    matches = np.flatnonzero(dates.to_numpy() >= np.datetime64(cap_date))
    if matches.size == 0:
        raise ValueError("training cap date %s is after the end of the series" % cap_date)
    return int(matches[0])


def build_fold_manifest(folds) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": f.fold_id,
                "origin_pos": f.origin_pos,
                "origin_date": f.origin_date,
                "train_cutoff_pos": f.train_cutoff_pos,
                **{"test_pos_h%d" % h: p for h, p in f.test_positions.items()},
                **{"effective_lead_h%d" % h: v for h, v in f.effective_lead.items()},
                "origin_is_imputed": f.origin_is_imputed,
                "origin_adjacent_to_gap": f.origin_adjacent_to_gap,
                **{"test_is_imputed_h%d" % h: v for h, v in f.test_is_imputed.items()},
                "gap_contaminated": f.gap_contaminated,
            }
            for f in folds
        ]
    )


def _md_table(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a GitHub-flavoured markdown table.

    Written locally rather than via `DataFrame.to_markdown`, which needs the
    optional `tabulate` package -- the addendum commits to a minimal, pinned
    dependency set for the deployed app, and a report formatter is not worth a
    new runtime dependency.
    """
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + body)


def _fmt(df: pd.DataFrame, cols) -> str:
    """Metrics table rounded to a readable, non-misleading precision."""
    out = df.copy()
    for c in METRIC_COLS:
        if c in out.columns:
            out[c] = out[c].astype(float).round(3)
    return _md_table(out[cols])


def main() -> None:
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    holdout_start, dev_end = resolve_split_boundaries(df["is_imputed"], FINAL_TEST_WINDOW)
    cutoff_pos = resolve_cutoff_pos(df["parsed_date"])

    folds = generate_folds(
        dates=df["parsed_date"],
        is_imputed=df["is_imputed"],
        dev_end_pos=dev_end,
        horizons=FORECAST_HORIZONS,
    )

    manifest = build_fold_manifest(folds)
    FOLD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(FOLD_MANIFEST_PATH, index=False)

    preds = pd.concat(
        [
            run_walk_forward(
                df=df,
                target_col=target,
                model_factories=BASELINE_FACTORIES,
                folds=folds,
                cutoff_pos=cutoff_pos,
            )
            for target in TARGETS
        ],
        ignore_index=True,
    )
    preds["common_support"] = common_support_mask(preds)
    preds.to_csv(BASELINE_PREDICTIONS_PATH, index=False)

    metrics_all = aggregate_metrics(preds)
    metrics_all.insert(0, "fold_scope", "all_dev_folds")

    post = preds[preds["origin_post_cutoff"]]
    metrics_recent = aggregate_metrics(post)
    metrics_recent.insert(0, "fold_scope", "post_cutoff_origins")

    metrics_common = aggregate_metrics(preds[preds["common_support"]])
    metrics_common.insert(0, "fold_scope", "common_support")

    # The recent-regime ranking GOVERNS champion selection at Day 8 (addendum
    # Section 5), so it needs the like-for-like restriction at least as much as
    # the all-fold view does. Without it the three baselines were scored on 15,
    # 15 and 12-14 points respectively for the flow target.
    metrics_recent_common = aggregate_metrics(post[post["common_support"]])
    metrics_recent_common.insert(0, "fold_scope", "post_cutoff_common_support")

    metrics = pd.concat(
        [metrics_all, metrics_recent, metrics_common, metrics_recent_common],
        ignore_index=True,
    )
    metrics.to_csv(BASELINE_METRICS_PATH, index=False)

    _write_report(df, preds, metrics, folds, manifest, holdout_start, dev_end, cutoff_pos)

    print("Folds: %d | dev_end_pos=%d | holdout_start_pos=%d | cutoff_pos=%d"
          % (len(folds), dev_end, holdout_start, cutoff_pos))
    print("Predictions rows: %d" % len(preds))
    print("Wrote %s" % BASELINE_METRICS_REPORT_PATH)


def _write_report(df, preds, metrics, folds, manifest, holdout_start, dev_end, cutoff_pos) -> None:
    d = df["parsed_date"]
    n_real = int((~df["is_imputed"]).sum())
    post_folds = [f for f in folds if f.origin_pos >= cutoff_pos]

    lines = []
    lines.append("# Day 5 -- Baseline Metrics & Walk-Forward Harness")
    lines.append("")
    lines.append("**Generated by** `src/evaluation/run_baselines.py`. ")
    lines.append("Every number below is produced by running the pipeline -- none is asserted.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Split geometry")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append("| Period-positions in master series | %d |" % len(df))
    lines.append("| Real (non-imputed) observations | %d |" % n_real)
    lines.append("| Final held-out test window | last %d **real** observations |" % FINAL_TEST_WINDOW)
    lines.append("| Holdout start | pos %d (%s) |" % (holdout_start, d.iloc[holdout_start].date()))
    lines.append("| Holdout end | pos %d (%s) |" % (len(df) - 1, d.iloc[-1].date()))
    lines.append("| Development portion | pos 0 (%s) .. pos %d (%s) |"
                 % (d.iloc[0].date(), dev_end, d.iloc[dev_end].date()))
    lines.append("| Real observations in development portion | %d |"
                 % int((~df["is_imputed"].to_numpy()[: dev_end + 1]).sum()))
    lines.append("| Training-cap position (%s) | pos %d (%s) |"
                 % (TRAINING_CAP_DATE, cutoff_pos, d.iloc[cutoff_pos].date()))
    lines.append("")
    lines.append("## 2. Fold cadence")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Minimum initial training size | %d periods |" % MIN_INITIAL_TRAINING)
    lines.append("| Step between fold origins | %d periods |" % WALK_FORWARD_STEP)
    lines.append("| Horizons | %s periods |" % FORECAST_HORIZONS)
    lines.append("| Folds generated | %d |" % len(folds))
    lines.append("| First origin | pos %d (%s) |"
                 % (folds[0].origin_pos, folds[0].origin_date.date()))
    lines.append("| Last origin | pos %d (%s) |"
                 % (folds[-1].origin_pos, folds[-1].origin_date.date()))
    lines.append("| Folds with origin on/after the cutoff | %d |" % len(post_folds))
    lines.append("")
    lines.append("Every fold satisfies `max(train_position) < min(test_position)` and no test ")
    lines.append("position reaches the held-out window -- both asserted in `generate_folds` and ")
    lines.append("re-tested in `tests/test_validation.py`.")
    lines.append("")
    lines.append("## 3. Training-cap fallback (the ~60-80 usable-row floor)")
    lines.append("")
    lines.append("Enforced floor: **%d** usable rows (lower bound of the addendum's frozen range; "
                 "upper bound %d reported as a sensitivity)." % (TRAINING_FLOOR_ROWS, MAX_USABLE_ROWS_FLOOR))
    lines.append("")
    cap = preds[(preds["window_rule"] == "capped") & preds["origin_post_cutoff"]]
    lines.append("| Target | Post-cutoff folds | Fell back to full window | Genuinely capped | Would flip at floor=%d |"
                 % MAX_USABLE_ROWS_FLOOR)
    lines.append("|---|---|---|---|---|")
    for target, g in cap.groupby("target"):
        n_folds = g["fold_id"].nunique()
        n_fb = g.loc[g["fallback_applied"], "fold_id"].nunique()
        n_flip = g.loc[g["flips_at_upper_floor"], "fold_id"].nunique()
        lines.append("| %s | %d | %d | %d | %d |" % (target, n_folds, n_fb, n_folds - n_fb, n_flip))
    lines.append("")
    lines.append("Folds whose origin *precedes* the cutoff are untouched by the cap "
                 "(invariant 4): their capped window is identical to their full window.")
    lines.append("")
    lines.append("**Read this before comparing the two window rules below.** All three "
                 "baselines depend only on the *tail* of the training window -- the last "
                 "value, the last m=%d values, the last w=%d values. Truncating the *start* "
                 "of the window therefore cannot change their point forecasts, so MAE, RMSE, "
                 "MAPE, sMAPE and bias are identical under `full` and `capped` **by "
                 "construction, not by coincidence**. Only MASE differs, because its "
                 "denominator is an in-sample seasonal-naive scale measured over the whole "
                 "training window, which the cap does change. The cap is still exercised and "
                 "logged here so that Days 6-7 (SARIMAX, ETSModel, RandomForest, "
                 "HistGradientBoosting -- all of which fit on the entire window) inherit a "
                 "harness that has already been proven to apply it correctly."
                 % (SEASONAL_PERIOD_M, MOVING_AVERAGE_WINDOW))
    lines.append("")
    lines.append("## 4. Baseline metrics -- all development folds")
    lines.append("")
    cols = ["target", "model", "window_rule", "horizon", "n_scored"] + METRIC_COLS
    for target in TARGETS:
        sub = metrics[(metrics["fold_scope"] == "all_dev_folds") & (metrics["target"] == target)]
        lines.append("### %s" % target)
        lines.append("")
        lines.append(_fmt(sub, cols))
        lines.append("")
    lines.append("## 5. Baseline metrics -- folds with origin on/after %s" % TRAINING_CAP_DATE)
    lines.append("")
    lines.append("Addendum Section 5: where the two rankings disagree, **this** "
                 "recent-regime ranking governs champion selection at Day 8.")
    lines.append("")
    for target in TARGETS:
        sub = metrics[(metrics["fold_scope"] == "post_cutoff_origins") & (metrics["target"] == target)]
        lines.append("### %s" % target)
        lines.append("")
        lines.append(_fmt(sub, cols))
        lines.append("")
    lines.append("## 6. Like-for-like comparison (common support)")
    lines.append("")
    n_dropped = int((~preds["common_support"]).sum())
    lines.append("Models abstain at different points: the seasonal-naive lookback sometimes lands "
                 "on a true-missing flow-gap slot, so it returns fewer forecasts than naive on the "
                 "same folds (see Section 8). Comparing raw MAE across models scored on different "
                 "subsets is not like-for-like, and the Day-8 champion rule needs a like-for-like "
                 "one. The table below restricts every model to the test points **all** baselines "
                 "could be scored on -- %d of %d prediction rows dropped. The abstentions are "
                 "reported, not hidden." % (n_dropped, len(preds)))
    lines.append("")
    for target in TARGETS:
        sub = metrics[(metrics["fold_scope"] == "common_support") & (metrics["target"] == target)]
        lines.append("### %s -- all development folds, common support" % target)
        lines.append("")
        lines.append(_fmt(sub, cols))
        lines.append("")
    lines.append("The same restriction applied to the **recent-regime** scope that governs Day-8 "
                 "champion selection:")
    lines.append("")
    for target in TARGETS:
        sub = metrics[
            (metrics["fold_scope"] == "post_cutoff_common_support")
            & (metrics["target"] == target)
        ]
        lines.append("### %s -- post-%s origins, common support" % (target, TRAINING_CAP_DATE))
        lines.append("")
        lines.append(_fmt(sub, cols))
        lines.append("")

    lines.append("## 7. Interpolation handling and contamination check")
    lines.append("")
    lines.append("Addendum Section 5 requires folds whose training cut-off or near-term test point "
                 "falls in or next to an interpolated gap to be flagged **and checked**, not "
                 "assumed negligible. Checking found two concrete problems, both now fixed in the "
                 "harness rather than merely reported:")
    lines.append("")
    truncated = manifest[manifest["origin_pos"] != manifest["train_cutoff_pos"]]
    lines.append("**(a) Interpolated origins leaked the future.** Stock columns are linearly "
                 "interpolated at gap slots, so an interpolated value encodes the *next real "
                 "observation after it* -- which, at a fold origin, lies after the origin. "
                 "Training windows now end at `train_cutoff_pos`, the last REAL observation at or "
                 "before the origin, never at an interpolated origin. Folds whose window was "
                 "pulled back as a result: **%d of %d**." % (len(truncated), len(manifest)))
    for r in truncated.itertuples():
        lines.append("  - fold %d: origin pos %d (%s) -> training now ends at pos %d, so its "
                     "nominal h=1/7/14 are really %d/%d/%d periods ahead of the last observation."
                     % (r.fold_id, r.origin_pos, pd.Timestamp(r.origin_date).date(),
                        r.train_cutoff_pos,
                        r.test_pos_h1 - r.train_cutoff_pos,
                        r.test_pos_h7 - r.train_cutoff_pos,
                        r.test_pos_h14 - r.train_cutoff_pos))
    lines.append("")
    lines.append("**(b) Interpolated values were being scored as if observed.** A test point "
                 "landing on a gap slot has no published actual. For the flow target it is "
                 "true-missing and drops out on its own; for the **stock** target `clean.py` "
                 "filled it, so `y_true` was not NaN and the forecast was graded against a "
                 "straight line. `aggregate_metrics` now excludes any actual that was not "
                 "observed. Test points landing on an interpolated slot: **%d of %d** per "
                 "target (Section 8 counts each one once per window rule)."
                 % (sum(1 for f in folds for h in f.horizons if f.test_is_imputed[h]),
                    len(FORECAST_HORIZONS) * len(folds)))
    lines.append("")
    lines.append("Residual check on what remains: fold-level flagged folds **%d of %d**; the table "
                 "below compares scored points that are adjacent to a gap on either side against "
                 "those that are not, per horizon."
                 % (int(manifest["gap_contaminated"].sum()), len(manifest)))
    lines.append("")
    contam = _contamination_table(preds)
    lines.append(_md_table(contam))
    lines.append("")
    worst = contam["MAE_shift_pct_if_dropped"].abs().max()
    flagged = manifest[manifest["gap_contaminated"]]
    flagged_desc = ", ".join(
        "fold %d (%s)" % (r.fold_id, pd.Timestamp(r.origin_date).date())
        for r in flagged.itertuples()
    )
    lines.append("`MAE_shift_pct_if_dropped` is the percentage change in aggregate MAE from "
                 "excluding the gap-adjacent points entirely. Largest shift across all "
                 "target/model/horizon cells: **%.2f%%**. Fold-level flagged folds: %s. Reported "
                 "as a measured number, not waved off as negligible."
                 % (worst, flagged_desc))
    lines.append("")
    lines.append("## 8. Missing-data and abstention accounting")
    lines.append("")
    miss = (
        preds.groupby(["target", "model"])
        .apply(
            lambda g: pd.Series(
                {
                    "rows": len(g),
                    "actual_missing_NaN": int(g["y_true"].isna().sum()),
                    "actual_interpolated_excluded": int(
                        (~g["y_true_is_observed"].astype(bool) & g["y_true"].notna()).sum()
                    ),
                    "forecast_abstained_NaN": int(g["y_pred"].isna().sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    lines.append(_md_table(miss))
    lines.append("")
    lines.append("`actual_missing_NaN` -- flow-column gaps, true-missing and never zero-filled or "
                 "interpolated (invariant 2). `actual_interpolated_excluded` -- stock-column gap "
                 "slots that carry an interpolated value; excluded from scoring because an "
                 "interpolated value is not an observation. `forecast_abstained_NaN` -- the "
                 "seasonal-naive baseline returning no forecast because its m=%d lookback landed "
                 "on a true-missing flow slot. That abstention is correct behaviour under "
                 "invariant 2 (the alternative would be inventing a value), and Section 6's "
                 "common-support view is what makes the model comparison fair despite it."
                 % SEASONAL_PERIOD_M)
    lines.append("")
    lines.append("## 9. Note on MAPE for the flow target")
    lines.append("")
    dev_actuals = preds[
        (preds["target"] == TARGET_2) & preds["y_true_is_observed"].astype(bool)
    ]["y_true"].dropna()
    lines.append("MAPE for `%s` runs high across the board. The cause is **small denominators, "
                 "not zero-days**: across the scored development folds this target has **%d** "
                 "zero actuals, a minimum of **%d**, and **%.1f%%** of actuals below 10. Its one "
                 "true zero-day (2025-11-30) falls inside the held-out window and is never scored "
                 "in cross-validation. sMAPE and MASE are the metrics to read here; MAPE is "
                 "retained because the official documentation lists it, and is flagged as "
                 "unstable wherever it appears."
                 % (TARGET_2, int((dev_actuals == 0).sum()), int(dev_actuals.min()),
                    100.0 * float((dev_actuals < 10).mean())))
    lines.append("")

    BASELINE_METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _contamination_table(preds: pd.DataFrame) -> pd.DataFrame:
    from src.evaluation.metrics import mae, smape

    rows = []
    for (target, model, h), g in preds[preds["window_rule"] == "full"].groupby(
        ["target", "model", "horizon"]
    ):
        g = g[g["y_true_is_observed"].astype(bool)]
        clean = g[~g["row_gap_adjacent"]]
        dirty = g[g["row_gap_adjacent"]]
        mae_all = mae(g["y_true"], g["y_pred"])
        mae_clean = mae(clean["y_true"], clean["y_pred"])
        shift = (
            (mae_clean - mae_all) / mae_all * 100.0
            if np.isfinite(mae_all) and mae_all > 0 and np.isfinite(mae_clean)
            else float("nan")
        )
        rows.append(
            {
                "target": target,
                "model": model,
                "horizon": h,
                "MAE_all_folds": round(mae_all, 3),
                "n_clean": int(clean["y_true"].notna().sum()),
                "MAE_clean": round(mae_clean, 3),
                "n_flagged": int(dirty["y_true"].notna().sum()),
                "MAE_flagged": round(mae(dirty["y_true"], dirty["y_pred"]), 3),
                "MAE_shift_pct_if_dropped": round(shift, 3),
                "sMAPE_clean": round(smape(clean["y_true"], clean["y_pred"]), 3),
                "sMAPE_flagged": round(smape(dirty["y_true"], dirty["y_pred"]), 3),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
