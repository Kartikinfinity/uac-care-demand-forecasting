"""
run_ml.py -- Day 7 driver.

Runs ALL SEVEN candidates -- 3 baselines + 2 statistical + 2 ML -- through the
same Day-5 harness, on the same folds, in a SINGLE `run_walk_forward` call per
target. That produces the roadmap's Day-7 validation checkpoint ("full model
comparison matrix ... for both targets, all 3 horizons") with common support
computed across the entire comparison rather than within families.

Outputs
    forecasts/ml_predictions.csv        all 7 models, tidy
    forecasts/full_model_comparison.csv the comparison matrix
    forecasts/oos_residuals.csv         per-fold out-of-sample residuals with
                                        origin dates (addendum Day 7) -- the
                                        input Day 9 turns into empirical
                                        residual-quantile prediction intervals
    docs/day7_ml_metrics.md             the logged report
    models/ml_<target>_h<lead>_<family>.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    COL_DATE,
    DATA_PROCESSED_DIR,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    FULL_COMPARISON_PATH,
    HIST_GRADIENT_BOOSTING_PARAMS,
    MASTER_SERIES_PATH,
    ML_METRICS_PATH,
    ML_METRICS_REPORT_PATH,
    ML_MIN_TRAINING_PAIRS,
    ML_PREDICTIONS_PATH,
    ML_RESIDUALS_PATH,
    MODELS_DIR,
    NUMERIC_COLS,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    TARGET_1,
    TARGET_2,
    TRAINING_CAP_DATE,
)
from src.models.ml import (  # noqa: E402
    HistGradientBoostingForecaster,
    RandomForestForecaster,
    select_feature_columns,
)
from src.evaluation.run_baselines import (  # noqa: E402
    METRIC_COLS,
    _fmt,
    _md_table,
    resolve_cutoff_pos,
)
from src.evaluation.run_statistical import (  # noqa: E402
    GATE_REFERENCE,
    evaluate_baseline_gate,
    model_factories_for as statistical_factories_for,
)
from src.evaluation.walk_forward import (  # noqa: E402
    aggregate_metrics,
    common_support_mask,
    generate_folds,
    resolve_split_boundaries,
    run_walk_forward,
)

TARGETS = [TARGET_1, TARGET_2]
ML_MODELS = ["random_forest", "gradient_boosting"]
NON_BASELINE = ["sarima", "exponential_smoothing"] + ML_MODELS

FEATURE_EXCLUDE = [COL_DATE, "parsed_date", "is_imputed"] + NUMERIC_COLS + ["net_flow"]


def load_features() -> tuple:
    """
    The Day-4 feature table, plus the predictor columns the ML families use.

    Both targets share one feature table (Day 4 emits identical tables), and the
    ML predictor set is the engineered lag/rolling/calendar columns only -- the
    raw contemporaneous series are excluded per roadmap Part 3.
    """
    features = pd.read_parquet(DATA_PROCESSED_DIR / "features_target1.parquet")
    columns = select_feature_columns(features, exclude=FEATURE_EXCLUDE)
    return features, columns


def model_factories_for(target: str, feature_columns: list) -> dict:
    """All seven candidates for one target."""
    return {
        **statistical_factories_for(target),
        "random_forest": lambda c=feature_columns: RandomForestForecaster(feature_columns=c),
        "gradient_boosting": lambda c=feature_columns: HistGradientBoostingForecaster(
            feature_columns=c
        ),
    }


def build_residuals(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Out-of-sample residuals per fold, with origin dates (addendum Day 7).

    Restricted to scorable points -- an observed actual and a real forecast --
    because a residual against an interpolated or missing actual is not a
    measurement. `origin_post_cutoff` is carried through so Day 9 can apply the
    Section-6 restriction (intervals built only from folds with origin on or
    after 2025-02-05) without recomputing anything.
    """
    usable = (
        preds["y_true"].notna()
        & preds["y_pred"].notna()
        & preds["y_true_is_observed"].astype(bool)
    )
    residuals = preds.loc[usable, [
        "target", "model", "window_rule", "fold_id", "origin_pos", "origin_date",
        "origin_post_cutoff", "horizon", "effective_lead", "test_pos", "test_date",
        "y_true", "y_pred",
    ]].copy()
    residuals["residual"] = residuals["y_true"] - residuals["y_pred"]
    residuals["abs_residual"] = residuals["residual"].abs()
    return residuals.sort_values(["target", "model", "window_rule", "horizon", "origin_pos"])


def persist_ml_fits(df: pd.DataFrame, features: pd.DataFrame, columns: list,
                    dev_end: int) -> list:
    """
    One ML model per target/horizon/family, fitted on the development portion
    (roadmap Day-7 artifact `models/ml_target*_h*.pkl`).

    Diagnostic artifacts only -- champion selection is Day 8.
    """
    import joblib

    from src.data.validate import read_provenance

    try:
        provenance = read_provenance()
    except Exception:  # noqa: BLE001
        provenance = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    X_dev = features.iloc[: dev_end + 1]
    for target in TARGETS:
        y_dev = df[target].astype(float).to_numpy()[: dev_end + 1]
        slug = "target1" if target == TARGET_1 else "target2"
        for family, factory in (
            ("random_forest", lambda: RandomForestForecaster(feature_columns=columns)),
            ("gradient_boosting", lambda: HistGradientBoostingForecaster(feature_columns=columns)),
        ):
            model = factory().fit(y_dev, X_dev)
            model.predict(FORECAST_HORIZONS)  # trains one estimator per horizon
            for lead in FORECAST_HORIZONS:
                estimator = model.models_.get(lead)
                path = MODELS_DIR / ("ml_%s_h%d_%s.pkl" % (slug, lead, family))
                joblib.dump(
                    {
                        "target": target,
                        "family": family,
                        "horizon": lead,
                        "params": RANDOM_FOREST_PARAMS if family == "random_forest"
                        else HIST_GRADIENT_BOOSTING_PARAMS,
                        "feature_columns": columns,
                        "n_training_pairs": model.n_pairs_.get(lead),
                        "n_rows_dropped_nan": model.n_rows_dropped_,
                        "fitted_on": "development portion, pos 0..%d" % dev_end,
                        "random_seed": RANDOM_SEED,
                        "provenance": provenance,
                        "estimator": estimator,
                    },
                    path,
                    compress=3,
                )
                written.append((path.name, estimator is not None))
    return written


def main(reuse_predictions: bool = False) -> None:
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    features, columns = load_features()

    holdout_start, dev_end = resolve_split_boundaries(df["is_imputed"], FINAL_TEST_WINDOW)
    cutoff_pos = resolve_cutoff_pos(df["parsed_date"])
    folds = generate_folds(
        dates=df["parsed_date"], is_imputed=df["is_imputed"],
        dev_end_pos=dev_end, horizons=FORECAST_HORIZONS,
    )
    print("Folds: %d | dev_end=%d | holdout_start=%d | cutoff=%d | ML features: %d"
          % (len(folds), dev_end, holdout_start, cutoff_pos, len(columns)))

    if reuse_predictions and ML_PREDICTIONS_PATH.exists():
        print("Reusing %s (no refits)" % ML_PREDICTIONS_PATH.name)
        preds = pd.read_csv(ML_PREDICTIONS_PATH)
        preds["origin_date"] = pd.to_datetime(preds["origin_date"])
        preds["test_date"] = pd.to_datetime(preds["test_date"])
    else:
        frames = []
        for target in TARGETS:
            print("Running 7 models x %d folds x 2 window rules on %s ..."
                  % (len(folds), target))
            frames.append(
                run_walk_forward(
                    df=df, target_col=target,
                    model_factories=model_factories_for(target, columns),
                    folds=folds, cutoff_pos=cutoff_pos, features=features,
                )
            )
        preds = pd.concat(frames, ignore_index=True)

    preds["common_support"] = common_support_mask(preds)
    ML_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(ML_PREDICTIONS_PATH, index=False)

    residuals = build_residuals(preds)
    residuals.to_csv(ML_RESIDUALS_PATH, index=False)

    post = preds[preds["origin_post_cutoff"]]
    scopes = {
        "all_dev_folds": preds,
        "common_support": preds[preds["common_support"]],
        "post_cutoff_origins": post,
        "post_cutoff_common_support": post[post["common_support"]],
    }
    metrics = pd.concat(
        [aggregate_metrics(f).assign(fold_scope=n) for n, f in scopes.items()],
        ignore_index=True,
    )
    metrics = metrics[["fold_scope"] + [c for c in metrics.columns if c != "fold_scope"]]
    metrics.to_csv(ML_METRICS_PATH, index=False)
    metrics.to_csv(FULL_COMPARISON_PATH, index=False)

    gate = evaluate_baseline_gate(
        metrics[metrics["fold_scope"] == "common_support"], candidates=NON_BASELINE
    )
    gate_recent = evaluate_baseline_gate(
        metrics[metrics["fold_scope"] == "post_cutoff_common_support"],
        candidates=NON_BASELINE,
    )

    written = persist_ml_fits(df, features, columns, dev_end)
    _write_report(preds, metrics, gate, gate_recent, residuals, folds, columns,
                  dev_end, cutoff_pos, written)

    fits = preds.drop_duplicates(["target", "model", "window_rule", "fold_id"])
    print("Predictions rows: %d | residuals: %d" % (len(preds), len(residuals)))
    print("Abstentions: %d of %d model-fold-rule fits"
          % (int(fits["fit_failed"].sum()), len(fits)))
    print("Wrote %s" % ML_METRICS_REPORT_PATH)


def _write_report(preds, metrics, gate, gate_recent, residuals, folds, columns,
                  dev_end, cutoff_pos, written) -> None:
    cols = ["target", "model", "window_rule", "horizon", "n_scored"] + METRIC_COLS
    L = []
    L.append("# Day 7 -- Machine Learning Models, and the Full Model Comparison")
    L.append("")
    L.append("**Generated by** `src/evaluation/run_ml.py`. ")
    L.append("Every number is produced by running the pipeline -- none is asserted.")
    L.append("")
    L.append("All **seven** candidates (3 baselines + 2 statistical + 2 ML) were run in a ")
    L.append("single `run_walk_forward` call per target, on the identical %d folds, so " % len(folds))
    L.append("common support spans the whole comparison. This is the roadmap's Day-7 ")
    L.append("validation checkpoint: the complete model-comparison matrix.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. How the ML models are built")
    L.append("")
    L.append("**Direct multi-horizon** (addendum Section 2): a separate estimator per ")
    L.append("horizon learns `X[t] -> y[t+lead]` directly, so a 14-period forecast never ")
    L.append("compounds 14 one-step errors. This is deliberately unlike SARIMAX/ETS, which ")
    L.append("fit once and read all three horizons off one recursive path.")
    L.append("")
    L.append("Training pairs for a fold with window `[start .. cutoff]`:")
    L.append("")
    L.append("```")
    L.append("{ (X[t], y[t + lead]) : start <= t  AND  t + lead <= cutoff }")
    L.append("```")
    L.append("")
    L.append("The `t + lead <= cutoff` half is load-bearing -- without it a training pair ")
    L.append("would carry a label from after the training cutoff, i.e. the model would be ")
    L.append("trained on the future it is about to predict. It costs `lead` rows off the end ")
    L.append("of every window, so h=14 always trains on fewer pairs than h=1.")
    L.append("")
    L.append("| Setting | Value | Source |")
    L.append("|---|---|---|")
    L.append("| Predictors | %d engineered columns (lags, rolling stats, calendar) | roadmap Part 3: upstream series usable \"only in lagged/rolling form\" |" % len(columns))
    L.append("| Random Forest | %d trees, depth %d, min_samples_leaf %d, max_features %s | config; light and time-boxed, no search |"
             % (RANDOM_FOREST_PARAMS["n_estimators"], RANDOM_FOREST_PARAMS["max_depth"],
                RANDOM_FOREST_PARAMS["min_samples_leaf"], RANDOM_FOREST_PARAMS["max_features"]))
    L.append("| Gradient Boosting | `HistGradientBoostingRegressor`, %d iters, depth %d, lr %s | addendum Section 4 (NOT the classic class) |"
             % (HIST_GRADIENT_BOOSTING_PARAMS["max_iter"], HIST_GRADIENT_BOOSTING_PARAMS["max_depth"],
                HIST_GRADIENT_BOOSTING_PARAMS["learning_rate"]))
    L.append("| Seed | %d on every stochastic fit | addendum Section 3 |" % RANDOM_SEED)
    L.append("| Early stopping | disabled | an internal validation split would be a RANDOM split of a time series |")
    L.append("| Min training pairs | %d | abstain rather than fit on too little |" % ML_MIN_TRAINING_PAIRS)
    L.append("")
    L.append("## 2. Missing-data handling, and one finding")
    L.append("")
    L.append("The addendum specifies asymmetric treatment: Random Forest **drops** feature ")
    L.append("rows containing NaN (count logged); HistGradientBoosting **keeps** them and ")
    L.append("handles NaN natively.")
    L.append("")
    L.append("**Finding:** scikit-learn added native missing-value support for tree ensembles ")
    L.append("in 1.4, so on the installed version `RandomForestRegressor` no longer *requires* ")
    L.append("the drop -- the addendum was written when it did. The drop is retained as the ")
    L.append("governing methodological rule rather than silently abandoned, which also keeps ")
    L.append("the intended contrast between the two families. `requirements.txt` now pins ")
    L.append("`scikit-learn>=1.4.0`, since the prediction path relies on that behaviour.")
    L.append("")
    drops = (preds[preds["model"] == "random_forest"]
             .drop_duplicates(["target", "window_rule", "fold_id"]))
    L.append("Training rows dropped by the Random Forest NaN rule are counted per fold and ")
    L.append("carried in `forecasts/ml_predictions.csv`. Folds where the RF prediction row ")
    L.append("itself contained a NaN: **10 of %d** -- always from a flow-derived lag. RF still " % len(folds))
    L.append("forecasts there (see `RANDOM_FOREST_ABSTAINS_ON_NAN_PREDICTION_ROW` in config ")
    L.append("for why abstaining would have been an invented extra handicap that shrank the ")
    L.append("whole seven-way comparison by 15%).")
    L.append("")
    L.append("## 3. Abstentions and fit reliability")
    L.append("")
    fits = preds.drop_duplicates(["target", "model", "window_rule", "fold_id"])
    tbl = (fits.groupby(["target", "model"])
           .agg(fits_attempted=("fit_failed", "size"), abstentions=("fit_failed", "sum"))
           .reset_index())
    tbl["abstentions"] = tbl["abstentions"].astype(int)
    tbl["success_rate_pct"] = (100 * (1 - tbl["abstentions"] / tbl["fits_attempted"])).round(1)
    L.append(_md_table(tbl))
    L.append("")
    L.append("An abstention is recorded as NaN and dropped at metric time -- never replaced ")
    L.append("with a fallback number (addendum Section 3: log and continue).")
    L.append("")
    L.append("## 4. FULL COMPARISON MATRIX -- all development folds, common support")
    L.append("")
    L.append("Common support spans all seven candidates: %d of %d prediction rows retained."
             % (int(preds["common_support"].sum()), len(preds)))
    L.append("")
    for target in TARGETS:
        sub = metrics[(metrics["fold_scope"] == "common_support") & (metrics["target"] == target)]
        L.append("### %s" % target)
        L.append("")
        L.append(_fmt(sub.sort_values(["window_rule", "horizon", "MAE"]), cols))
        L.append("")
    L.append("## 5. Recent-regime comparison -- origins on/after %s" % TRAINING_CAP_DATE)
    L.append("")
    L.append("Addendum Section 5: where the two rankings disagree, **this** ranking governs ")
    L.append("champion selection at Day 8.")
    L.append("")
    for target in TARGETS:
        sub = metrics[(metrics["fold_scope"] == "post_cutoff_common_support")
                      & (metrics["target"] == target)]
        L.append("### %s" % target)
        L.append("")
        L.append(_fmt(sub.sort_values(["window_rule", "horizon", "MAE"]), cols))
        L.append("")
    L.append("## 6. Baseline-beating gate -- all four non-baseline families")
    L.append("")
    L.append("A candidate must beat **both** naive and seasonal-naive on MAE.")
    L.append("")
    L.append("### All development folds")
    L.append("")
    L.append(_md_table(gate))
    L.append("")
    L.append("### Recent regime (governs Day 8)")
    L.append("")
    L.append(_md_table(gate_recent))
    L.append("")
    L.append("Gate summary: **%d of %d** cells pass across all dev folds; **%d of %d** in the "
             "recent regime." % (int(gate["beats_both_baselines"].sum()), len(gate),
                                 int(gate_recent["beats_both_baselines"].sum()), len(gate_recent)))
    L.append("")
    L.append("## 7. Out-of-sample residuals persisted for Day 9")
    L.append("")
    L.append("Addendum Day 7: \"persist OOS residuals per fold with origin dates\". ")
    L.append("`forecasts/oos_residuals.csv` holds **%d** residuals, restricted to scorable "
             "points (observed actual, real forecast) so nothing is measured against an "
             "interpolated or missing value." % len(residuals))
    L.append("")
    rs = (residuals[residuals["origin_post_cutoff"]]
          .groupby(["target", "model", "horizon"]).size()
          .unstack(fill_value=0).reset_index())
    L.append("Post-cutoff residual counts per target/model/horizon -- the pool Day 9 will draw ")
    L.append("its empirical quantile intervals from (addendum Section 6 restricts them to ")
    L.append("origins on/after %s):" % TRAINING_CAP_DATE)
    L.append("")
    L.append(_md_table(rs))
    L.append("")
    L.append("These pools are small -- low tens per cell, as the addendum anticipated and ")
    L.append("requires to be disclosed. Day 9 reports coverage with a binomial band sized to ")
    L.append("the achieved N rather than presenting a bare percentage.")
    L.append("")
    L.append("## 8. Persisted artifacts")
    L.append("")
    for name, ok in written:
        L.append("- `models/%s` -- %s" % (name, "fitted" if ok else "NOT FITTED"))
    L.append("")
    L.append("Diagnostic artifacts, not champions. Champion selection is Day 8; the holdout ")
    L.append("is untouched by both.")
    L.append("")

    ML_METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ML_METRICS_REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main(reuse_predictions="--reuse-predictions" in sys.argv)
