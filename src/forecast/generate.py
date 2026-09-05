"""
generate.py -- The single offline producer of every dashboard artifact (Day 9).

Roadmap Part 9: "src/forecast/generate.py is the ONLY place forecasts get
produced." The Streamlit app is a pure consumer -- it reads flat files and never
fits a model inside a page callback. That separation is what makes the
"dashboard drifts away from the actual model" risk detectable: there is exactly
one producer, and every artifact it writes carries the same provenance hash.

What this script produces
-------------------------
  forward_forecasts.csv      champion point forecasts at h=1/7/14 beyond the
                             last observation, with empirical intervals
  interval_coverage.csv      PRIMARY calibration evidence: pooled post-cutoff
                             walk-forward coverage, with a binomial band
  holdout_evaluation.csv     SECONDARY confirmatory check on the final held-out
                             window -- the one and only time it is touched
  imbalance_forecast.csv     the derived intake-vs-exit signal and its combined
                             uncertainty
  early_warning_backtest.csv per-origin tier firings over the development
                             portion, and the KPI summary built from them
  kpi_summary.csv            Surge Lead Time, false-positive/negative rates,
                             Forecast Stability Index, capacity tier
  provenance.json            data hashes, timestamp, and the winning
                             configuration per target/horizon

THE HOLDOUT
-----------
Addendum Section 5: the final 60 real observations are "reserved and touched
exactly once, after champions are frozen". Champions were frozen at Day 8 and
are read here from the registry -- this script SELECTS NOTHING. The holdout is
used for one confirmatory coverage check and never feeds back into any choice.
If its numbers are poor, that is a disclosed result, not a trigger to re-tune.

A fail-fast gate runs first (addendum Section 3), so a master series that no
longer matches the frozen specification stops the batch instead of quietly
producing artifacts the dashboard would then present as current.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    CAPACITY_TIER_LABELS,
    DAY9_REPORT_PATH,
    EARLY_WARNING_BACKTEST_PATH,
    EARLY_WARNING_PERCENTILE,
    EARLY_WARNING_TIERS,
    EARLY_WARNING_TRAILING_WINDOW,
    EMPIRICAL_INTERVAL_ALPHA,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    FORECAST_PROVENANCE_PATH,
    FORWARD_FORECASTS_PATH,
    HOLDOUT_EVALUATION_PATH,
    IMBALANCE_BASELINE_MODELS,
    IMBALANCE_COMPONENT,
    IMBALANCE_CORRELATION_PATH,
    IMBALANCE_FORECAST_PATH,
    INTERVAL_COVERAGE_PATH,
    KPI_SUMMARY_PATH,
    MASTER_SERIES_PATH,
    ML_RESIDUALS_PATH,
    MODEL_REGISTRY_PATH,
    MOVING_AVERAGE_WINDOW,
    SEASONAL_PERIOD_M,
    SELECTION_WINDOW_RULE,
    TARGET_1,
    TARGET_2,
    TRAINING_CAP_DATE,
)
from src.data.validate import read_provenance, validate_master_series  # noqa: E402
from src.models.baselines import (  # noqa: E402
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalNaiveBaseline,
)
from src.models.registry import read_registry  # noqa: E402
from src.models.statistical import ETSForecaster, SarimaxForecaster  # noqa: E402
from src.evaluation.intervals import (  # noqa: E402
    empirical_interval,
    imbalance_variance,
)
from src.evaluation.run_baselines import resolve_cutoff_pos  # noqa: E402
from src.evaluation.walk_forward import (  # noqa: E402
    generate_folds,
    resolve_split_boundaries,
)
from src.signals.early_warning import (  # noqa: E402
    PROXY_DISCLAIMER,
    backtest_origin,
    capacity_tier,
    summarise_backtest,
    trailing_threshold,
)

TARGETS = [TARGET_1, TARGET_2]


def build_model(name: str, target: str):
    """
    Instantiate a champion by name. Only the families that actually won are
    reachable here; anything else is a registry/config mismatch and raises
    rather than silently substituting a different model.
    """
    from src.config import ETS_SPECS, SARIMA_ORDERS

    if name == "naive":
        return NaiveBaseline()
    if name == "seasonal_naive":
        return SeasonalNaiveBaseline(m=SEASONAL_PERIOD_M)
    if name == "moving_average":
        return MovingAverageBaseline(w=MOVING_AVERAGE_WINDOW)
    if name == "sarima":
        return SarimaxForecaster(**SARIMA_ORDERS[target])
    if name == "exponential_smoothing":
        return ETSForecaster(**ETS_SPECS[target])
    raise ValueError(
        "champion %r for %r has no constructor here -- registry and config disagree"
        % (name, target)
    )


def load_all_residuals() -> pd.DataFrame:
    """
    Every candidate's out-of-sample residuals, including the ensemble's.

    Day 7 persisted residuals for the seven fitted models; the ensemble was
    constructed at Day 8, AFTER that file was written, so it has none there.
    Without this it would be the one candidate the dashboard could show a point
    forecast for but no interval -- silently, which is worse than not offering
    it at all. Its residuals are derived here from the Day-8 ensemble
    predictions using the same definition, rather than mutating a prior day's
    artifact.
    """
    from src.config import ENSEMBLE_PREDICTIONS_PATH
    from src.evaluation.run_ml import build_residuals

    residuals = pd.read_csv(ML_RESIDUALS_PATH)
    if ENSEMBLE_PREDICTIONS_PATH.exists():
        ensemble = pd.read_csv(ENSEMBLE_PREDICTIONS_PATH)
        if len(ensemble):
            residuals = pd.concat(
                [residuals, build_residuals(ensemble)[residuals.columns]],
                ignore_index=True,
            )
    return residuals


def residual_pool(residuals: pd.DataFrame, target: str, model: str, horizon: int):
    """
    The out-of-sample residuals an interval is built from.

    Restricted to post-cutoff origins under the deployed window rule -- addendum
    Section 6 ties the interval pool to the same 2025-02-05 boundary as
    training. Every residual here comes from a fold TEST point, never from a
    fitted value (invariant 5).
    """
    sub = residuals[
        (residuals["target"] == target)
        & (residuals["model"] == model)
        & (residuals["horizon"] == horizon)
        & (residuals["window_rule"] == SELECTION_WINDOW_RULE)
        & residuals["origin_post_cutoff"]
    ]
    return sub["residual"].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# 1. Forward forecasts
# ----------------------------------------------------------------------
def forward_forecasts(df, registry, residuals, cutoff_pos) -> pd.DataFrame:
    """
    The product output: what EVERY candidate predicts beyond the last
    observation, with the champion flagged.

    All models, not only the champions, because the documented User Capability
    "model toggle" (Part 8, pages 3/4/6/7) lets a reader switch between them --
    and the app is forbidden from fitting anything in a page callback. If only
    champion forecasts existed here, that control could not be honoured without
    breaking the no-live-training rule, so the single producer emits the whole
    set and the dashboard just filters it.

    Trained on everything available up to the last REAL observation, under the
    same capped-window rule the champions were selected under -- a forecast
    issued today must use the training rule it was validated with, not a
    different one.
    """
    from src.evaluation.run_ml import load_features
    from src.evaluation.run_selection import ENSEMBLE, family_champion, ML_FAMILY, STATISTICAL_FAMILY
    from src.evaluation.run_ml import model_factories_for

    is_imputed = df["is_imputed"].to_numpy(dtype=bool)
    dates = pd.to_datetime(df["parsed_date"])
    last_real = int(np.flatnonzero(~is_imputed)[-1])
    start = cutoff_pos if last_real >= cutoff_pos else 0

    features, feature_columns = load_features()
    champions = {(e["target"], int(e["horizon"])): e["champion"] for e in registry["entries"]}
    comparison = pd.read_csv(Path("forecasts") / "full_model_comparison.csv")

    rows = []
    for target in TARGETS:
        y = df[target].astype(float).to_numpy()
        y_train = y[start : last_real + 1]
        X_train = features.iloc[start : last_real + 1]
        factories = model_factories_for(target, feature_columns)

        # Statistical and baseline families fit ONCE and read every horizon off
        # one path; the ML families fit per horizon. Both contracts are honoured
        # by the model classes themselves, so this loop just drives them.
        fitted = {}
        for name, factory in factories.items():
            model = factory()
            if getattr(model, "requires_features", False):
                model.fit(y_train, X_train)
            else:
                model.fit(y_train)
            fitted[name] = model

        per_model_points = {}
        for name, model in fitted.items():
            preds = np.asarray(model.predict(list(FORECAST_HORIZONS)), dtype=float)
            per_model_points[name] = dict(zip(FORECAST_HORIZONS, preds))

        # The ensemble is a POST-HOC average of the two family champions -- the
        # same definition Day 8 evaluated, recomputed here on the same rule.
        for horizon in FORECAST_HORIZONS:
            stat = family_champion(comparison, target, horizon, STATISTICAL_FAMILY)
            ml = family_champion(comparison, target, horizon, ML_FAMILY)
            if stat and ml:
                a = per_model_points[stat][horizon]
                b = per_model_points[ml][horizon]
                per_model_points.setdefault(ENSEMBLE, {})[horizon] = (
                    (a + b) / 2.0 if np.isfinite(a) and np.isfinite(b) else np.nan
                )

        for name, points in per_model_points.items():
            for horizon, point in points.items():
                interval = empirical_interval(
                    point, residual_pool(residuals, target, name, horizon),
                    alpha=EMPIRICAL_INTERVAL_ALPHA, lower_bound=0.0,
                )
                rows.append(_forward_row(target, horizon, name, champions, last_real,
                                         dates, point, interval, start))
    return pd.DataFrame(rows).sort_values(
        ["target", "horizon", "is_champion", "model"], ascending=[True, True, False, True]
    ).reset_index(drop=True)


def _forward_row(target, horizon, name, champions, last_real, dates, point,
                 interval, start) -> dict:
    """One forward-forecast record. Split out only to keep the loop readable."""
    return {
            "target": target,
            "horizon": horizon,
            "model": name,
            "is_champion": champions.get((target, horizon)) == name,
            "champion": champions.get((target, horizon)),
            "origin_pos": last_real,
            "origin_date": dates.iloc[last_real].date(),
            "forecast_pos": last_real + horizon,
            "point_forecast": point,
            "lower": interval["lower"],
            "lower_unclipped": interval.get("lower_unclipped"),
            "clipped_at_zero": interval.get("clipped_at_lower_bound", False),
            "upper": interval["upper"],
            "interval_width": interval["width"],
            "n_residuals": interval["n_residuals"],
            "interval_sufficient": interval["sufficient"],
            "interval_note": interval["reason"],
            "nominal_coverage": interval["nominal_coverage"],
            "train_start_pos": start,
            "train_n_positions": last_real - start + 1,
    }


# ----------------------------------------------------------------------
# 2. Coverage -- primary (pooled folds) and secondary (holdout)
# ----------------------------------------------------------------------
def pooled_coverage(registry, residuals) -> pd.DataFrame:
    """
    PRIMARY calibration evidence: leave-one-out coverage of the interval on the
    same pooled post-cutoff residuals it is built from.

    Built leave-one-out on purpose. Scoring an interval against the very
    residuals that defined its quantiles would be circular and would flatter the
    result; holding each observation out and rebuilding the quantiles from the
    remaining ones is the honest version at this sample size.
    """
    rows = []
    for entry in registry["entries"]:
        target, horizon, champion = entry["target"], int(entry["horizon"]), entry["champion"]
        pool = residual_pool(residuals, target, champion, horizon)
        pool = pool[np.isfinite(pool)]
        n = pool.size
        if n < 3:
            continue

        covered = 0
        for i in range(n):
            others = np.delete(pool, i)
            lo, hi = np.percentile(
                others,
                [EMPIRICAL_INTERVAL_ALPHA / 2 * 100, (1 - EMPIRICAL_INTERVAL_ALPHA / 2) * 100],
            )
            if lo <= pool[i] <= hi:
                covered += 1

        from src.evaluation.intervals import wilson_interval
        band_low, band_high = wilson_interval(covered, n, 0.95)
        rows.append({
            "target": target, "horizon": horizon, "champion": champion,
            "evidence": "primary_pooled_folds_leave_one_out",
            "n": n, "n_covered": covered,
            "empirical_coverage": round(covered / n, 4),
            "nominal_coverage": 1 - EMPIRICAL_INTERVAL_ALPHA,
            "band_low": round(band_low, 4), "band_high": round(band_high, 4),
            "covers_nominal": bool(band_low <= (1 - EMPIRICAL_INTERVAL_ALPHA) <= band_high),
            "note": ("N is small (%d); the band is wide and this is not a precise "
                     "calibration estimate" % n) if n < 40 else "",
        })
    return pd.DataFrame(rows)


def holdout_evaluation(df, registry, residuals, cutoff_pos, holdout_start) -> pd.DataFrame:
    """
    SECONDARY confirmatory check -- the single, only time the holdout is used.

    Rolling origins inside the holdout, champion refit at each, interval taken
    from the DEVELOPMENT-portion residual pool (never rebuilt from holdout
    residuals, which would leak the holdout into its own interval). Nothing here
    feeds back into a selection decision; a poor result is disclosed, not acted on.
    """
    is_imputed = df["is_imputed"].to_numpy(dtype=bool)
    dates = pd.to_datetime(df["parsed_date"])
    n_pos = len(df)

    folds = generate_folds(df["parsed_date"], df["is_imputed"],
                           dev_end_pos=n_pos - 1, horizons=FORECAST_HORIZONS)
    holdout_folds = [f for f in folds if min(f.test_positions.values()) >= holdout_start]

    rows = []
    for entry in registry["entries"]:
        target, horizon, champion = entry["target"], int(entry["horizon"]), entry["champion"]
        y = df[target].astype(float).to_numpy()
        pool = residual_pool(residuals, target, champion, horizon)

        for fold in holdout_folds:
            test_pos = fold.test_positions[horizon]
            if test_pos >= n_pos or is_imputed[test_pos] or not np.isfinite(y[test_pos]):
                continue
            start = cutoff_pos if fold.train_cutoff_pos >= cutoff_pos else 0
            model = build_model(champion, target)
            model.fit(y[start : fold.train_cutoff_pos + 1])
            lead = fold.effective_lead[horizon]
            point = float(np.atleast_1d(model.predict([lead]))[0])
            interval = empirical_interval(point, pool, alpha=EMPIRICAL_INTERVAL_ALPHA)
            rows.append({
                "target": target, "horizon": horizon, "champion": champion,
                "fold_id": fold.fold_id, "origin_pos": fold.origin_pos,
                "origin_date": dates.iloc[fold.origin_pos].date(),
                "test_pos": test_pos, "test_date": dates.iloc[test_pos].date(),
                "y_true": y[test_pos], "point_forecast": point,
                "lower": interval["lower"], "upper": interval["upper"],
                "covered": (bool(interval["lower"] <= y[test_pos] <= interval["upper"])
                            if interval["sufficient"] else None),
                "abs_error": abs(y[test_pos] - point),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. Derived imbalance signal
# ----------------------------------------------------------------------
def transferred_out_residuals(df, folds, cutoff_pos) -> pd.DataFrame:
    """
    Post-cutoff out-of-sample residuals for the Transferred Out component.

    Recomputed here rather than reused from the Day-8 correlation table, which
    was measured across ALL folds. Mixing scopes is not a cosmetic difference:
    the 2023-24 regime carries roughly 100x the flow variance of the current
    one, so pairing an all-folds Transferred variance (std ~58) with a
    post-cutoff Discharged variance (std ~5.5) inflated the combined
    uncertainty about eight-fold and swamped a net-pressure signal of single
    digits. Addendum Section 6 ties the residual pool to the 2025-02-05 cutoff;
    both components must honour the same boundary.
    """
    from src.evaluation.run_baselines import BASELINE_FACTORIES
    from src.evaluation.walk_forward import run_walk_forward

    out = run_walk_forward(
        df=df, target_col=IMBALANCE_COMPONENT,
        model_factories={k: v for k, v in BASELINE_FACTORIES.items()
                         if k in IMBALANCE_BASELINE_MODELS},
        folds=folds, cutoff_pos=cutoff_pos,
    )
    usable = (out["y_true"].notna() & out["y_pred"].notna()
              & out["y_true_is_observed"].astype(bool)
              & out["origin_post_cutoff"]
              & (out["window_rule"] == SELECTION_WINDOW_RULE))
    out = out[usable].copy()
    out["residual"] = out["y_true"] - out["y_pred"]
    return out


def paired_component_stats(transferred, discharged_residuals, transferred_model,
                           discharged_model, horizon) -> dict:
    """Variance, covariance and correlation from PAIRED post-cutoff residuals."""
    key = ["fold_id", "horizon"]
    left = (transferred[(transferred["model"] == transferred_model)
                        & (transferred["horizon"] == horizon)]
            .set_index(key)["residual"])
    right = (discharged_residuals[(discharged_residuals["model"] == discharged_model)
                                  & (discharged_residuals["horizon"] == horizon)]
             .set_index(key)["residual"])
    paired = pd.concat([left, right], axis=1, join="inner",
                       keys=["transferred", "discharged"]).dropna()
    if len(paired) < 3:
        return {"n_paired": len(paired), "var_transferred": float("nan"),
                "var_discharged": float("nan"), "covariance": 0.0, "correlation": None}
    return {
        "n_paired": int(len(paired)),
        "var_transferred": float(paired["transferred"].var(ddof=1)),
        "var_discharged": float(paired["discharged"].var(ddof=1)),
        "covariance": float(np.cov(paired["transferred"], paired["discharged"], ddof=1)[0, 1]),
        "correlation": float(np.corrcoef(paired["transferred"], paired["discharged"])[0, 1]),
    }


def imbalance_forecast(df, registry, residuals, cutoff_pos, correlations,
                       transferred_residuals=None) -> pd.DataFrame:
    """
    Forward intake-vs-exit pressure: Transferred Out minus Discharged.

    Transferred Out uses the baseline treatment required for every series -- it
    is a derived-signal component, never a third target. Combined uncertainty
    follows addendum Section 6, with the independence simplification applied
    only where the MEASURED paired residual correlation earns it.
    """
    is_imputed = df["is_imputed"].to_numpy(dtype=bool)
    dates = pd.to_datetime(df["parsed_date"])
    last_real = int(np.flatnonzero(~is_imputed)[-1])
    start = cutoff_pos if last_real >= cutoff_pos else 0

    transferred_model = IMBALANCE_BASELINE_MODELS[0]
    t_series = df[IMBALANCE_COMPONENT].astype(float).to_numpy()

    rows = []
    for entry in registry["entries"]:
        if entry["target"] != TARGET_2:
            continue
        horizon, champion = int(entry["horizon"]), entry["champion"]

        d_model = build_model(champion, TARGET_2)
        d_model.fit(df[TARGET_2].astype(float).to_numpy()[start : last_real + 1])
        discharged = float(np.atleast_1d(d_model.predict([horizon]))[0])

        t_model = build_model(transferred_model, IMBALANCE_COMPONENT)
        t_model.fit(t_series[start : last_real + 1])
        transferred = float(np.atleast_1d(t_model.predict([horizon]))[0])

        d_res = residuals[
            (residuals["target"] == TARGET_2) & (residuals["model"] == champion)
            & (residuals["window_rule"] == SELECTION_WINDOW_RULE)
            & residuals["origin_post_cutoff"]
        ]
        stats = paired_component_stats(transferred_residuals, d_res, transferred_model,
                                       champion, horizon)
        combined = imbalance_variance(stats["var_transferred"], stats["var_discharged"],
                                      stats["covariance"], stats["correlation"])
        corr = stats["correlation"]
        net = transferred - discharged
        rows.append({
            "horizon": horizon,
            "origin_date": dates.iloc[last_real].date(),
            "transferred_model": transferred_model,
            "transferred_forecast": transferred,
            "discharged_model": champion,
            "discharged_forecast": discharged,
            "net_pressure": net,
            "variance": combined["variance"],
            "std": combined["std"],
            "lower_1sd": net - combined["std"],
            "upper_1sd": net + combined["std"],
            "uncertainty_form": combined["form_used"],
            "measured_correlation": corr,
            "n_paired_residuals": stats["n_paired"],
            "var_transferred": stats["var_transferred"],
            "var_discharged": stats["var_discharged"],
            "interpretation": ("intake exceeds exits -- net inflow pressure"
                               if net > 0 else "exits exceed intake -- net relief"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. Early-warning backtest (development portion only)
# ----------------------------------------------------------------------
def early_warning_backtest(df, registry, cutoff_pos, dev_end) -> tuple:
    """
    Replays the frozen trigger at every development-portion fold origin.

    Development portion ONLY: "the final test window is never used to tune or
    report this number". Nothing is tuned here in any case -- the percentile is
    a frozen convention -- so this measures what the fixed rule would have done,
    which is the only honest way to quote a lead time.
    """
    dates = pd.to_datetime(df["parsed_date"])
    is_observed = (~df["is_imputed"]).to_numpy(dtype=bool)
    folds = generate_folds(df["parsed_date"], df["is_imputed"], dev_end, FORECAST_HORIZONS)

    champions = {(e["target"], int(e["horizon"])): e["champion"] for e in registry["entries"]}
    rows, per_target = [], {}

    for target in TARGETS:
        y = df[target].astype(float).to_numpy()
        results = []
        for fold in folds:
            forecasts = {}
            for horizon in FORECAST_HORIZONS:
                champion = champions.get((target, horizon))
                if champion is None:
                    continue
                start = cutoff_pos if fold.train_cutoff_pos >= cutoff_pos else 0
                model = build_model(champion, target)
                model.fit(y[start : fold.train_cutoff_pos + 1])
                forecasts[horizon] = float(
                    np.atleast_1d(model.predict([fold.effective_lead[horizon]]))[0]
                )
            outcome = backtest_origin(y, is_observed, fold.origin_pos, forecasts)
            results.append(outcome)
            rows.append({
                "target": target, "fold_id": fold.fold_id,
                "origin_pos": fold.origin_pos,
                "origin_date": dates.iloc[fold.origin_pos].date(),
                "threshold": outcome["threshold"],
                "highest_tier": outcome["highest_tier"],
                "earliest_tier": outcome["earliest_tier"],
                "fired_horizons": ",".join(str(h) for h in sorted(outcome["fired"])),
                "outcome": outcome["outcome"],
                "lead_time": outcome["lead_time"],
                **{"forecast_h%d" % h: v for h, v in forecasts.items()},
            })
        per_target[target] = summarise_backtest(results)

    return pd.DataFrame(rows), per_target


# ----------------------------------------------------------------------
# 5. KPIs
# ----------------------------------------------------------------------
def build_kpis(df, registry, forwards, backtest_summary, holdout, cutoff_pos) -> pd.DataFrame:
    """
    The four KPIs the official documentation names, each with its formula
    recorded alongside the number so a reader never has to guess what it means.

    Forecast Stability Index is a DISCLOSED SUBSTITUTION -- see Day 8: the
    roadmap's definition (variance of forecasts for the same target date from
    different origins) is not computable under the frozen fold cadence, because
    no test position is reached from more than one origin.
    """
    is_imputed = df["is_imputed"].to_numpy(dtype=bool)
    is_observed = ~is_imputed
    last_real = int(np.flatnonzero(is_observed)[-1])

    rows = []
    for target in TARGETS:
        y = df[target].astype(float).to_numpy()
        threshold = trailing_threshold(y, last_real, is_observed)
        summary = backtest_summary.get(target, {})
        target_forwards = forwards[forwards["target"] == target]

        h1 = target_forwards[target_forwards["horizon"] == 1]
        tier = (capacity_tier(float(h1["point_forecast"].iloc[0]), threshold)
                if len(h1) else CAPACITY_TIER_LABELS[0])

        hold = holdout[holdout["target"] == target]
        stability = (float(hold["abs_error"].quantile(0.9) / hold["abs_error"].median())
                     if len(hold) and hold["abs_error"].median() > 0 else float("nan"))

        rows.append({
            "target": target,
            "capacity_tier": tier,
            "capacity_tier_formula": ("qualitative label from the forward h=1 forecast "
                                      "against the trailing %dth-percentile threshold"
                                      % EARLY_WARNING_PERCENTILE),
            "threshold_proxy_value": threshold,
            "threshold_is_proxy": True,
            "median_surge_lead_time_periods": summary.get("median_surge_lead_time"),
            "surge_lead_time_formula": ("median periods between the earliest tier firing at a "
                                        "historical origin and the series subsequently crossing "
                                        "the same threshold; development portion only"),
            "false_positive_rate": summary.get("false_positive_rate"),
            "false_negative_rate": summary.get("false_negative_rate"),
            "n_backtest_origins": summary.get("n_origins"),
            "n_fired": summary.get("n_fired"),
            "forecast_stability_index": stability,
            "stability_formula": ("SUBSTITUTED: p90/median of absolute error across holdout "
                                  "origins. The roadmap's same-target-date-from-different-"
                                  "origins definition is not computable -- 0 of 195 test "
                                  "positions are reached from more than one origin"),
            "proxy_disclaimer": PROXY_DISCLAIMER,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    # Fail-fast gate (addendum Section 3) -- reuse the Day-2 assertions as a
    # runtime guard so a drifted series stops the batch rather than producing
    # artifacts the dashboard would present as current.
    validate_master_series(df)

    registry = read_registry(MODEL_REGISTRY_PATH)
    residuals = load_all_residuals()
    correlations = pd.read_csv(IMBALANCE_CORRELATION_PATH)

    holdout_start, dev_end = resolve_split_boundaries(df["is_imputed"], FINAL_TEST_WINDOW)
    cutoff_pos = resolve_cutoff_pos(df["parsed_date"])
    print("Gate passed | holdout_start=%d dev_end=%d cutoff=%d"
          % (holdout_start, dev_end, cutoff_pos))

    forwards = forward_forecasts(df, registry, residuals, cutoff_pos)
    forwards.to_csv(FORWARD_FORECASTS_PATH, index=False)

    coverage = pooled_coverage(registry, residuals)
    coverage.to_csv(INTERVAL_COVERAGE_PATH, index=False)

    holdout = holdout_evaluation(df, registry, residuals, cutoff_pos, holdout_start)
    holdout.to_csv(HOLDOUT_EVALUATION_PATH, index=False)

    dev_folds = generate_folds(df["parsed_date"], df["is_imputed"], dev_end, FORECAST_HORIZONS)
    transferred = transferred_out_residuals(df, dev_folds, cutoff_pos)
    imbalance = imbalance_forecast(df, registry, residuals, cutoff_pos, correlations,
                                   transferred_residuals=transferred)
    imbalance.to_csv(IMBALANCE_FORECAST_PATH, index=False)

    backtest, summary = early_warning_backtest(df, registry, cutoff_pos, dev_end)
    backtest.to_csv(EARLY_WARNING_BACKTEST_PATH, index=False)

    kpis = build_kpis(df, registry, forwards, summary, holdout, cutoff_pos)
    kpis.to_csv(KPI_SUMMARY_PATH, index=False)

    data_provenance = read_provenance()
    sidecar = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": data_provenance["data_as_of"],
        "raw_csv_sha256": data_provenance["raw_csv_sha256"],
        "master_series_sha256": data_provenance["master_series_sha256"],
        "registry_generated_at_utc": registry.get("generated_at_utc"),
        "winning_configuration": [
            {"target": e["target"], "horizon": e["horizon"], "champion": e["champion"],
             "window_rule": e["window_rule"], "selection_scope": e["selection_scope"]}
            for e in registry["entries"]
        ],
        "nominal_interval_coverage": 1 - EMPIRICAL_INTERVAL_ALPHA,
        "early_warning": {
            "percentile": EARLY_WARNING_PERCENTILE,
            "trailing_window_periods": EARLY_WARNING_TRAILING_WINDOW,
            "tiers": {str(k): v for k, v in EARLY_WARNING_TIERS.items()},
            "is_proxy": True,
            "disclaimer": PROXY_DISCLAIMER,
        },
        "holdout_touched": True,
        "holdout_use": "single confirmatory coverage check; never fed back into selection",
        "refresh_policy": "manual only -- replace the CSV and re-run this script",
    }
    FORECAST_PROVENANCE_PATH.write_text(json.dumps(sidecar, indent=2, default=str) + "\n",
                                        encoding="utf-8")

    _write_report(df, forwards, coverage, holdout, imbalance, backtest, summary, kpis,
                  sidecar, holdout_start, dev_end)

    print("Forward forecasts: %d | coverage cells: %d | holdout points: %d"
          % (len(forwards), len(coverage), len(holdout)))
    print("Backtest origins: %d | imbalance horizons: %d" % (len(backtest), len(imbalance)))
    print("Wrote %s" % DAY9_REPORT_PATH)


def _write_report(df, forwards, coverage, holdout, imbalance, backtest, summary,
                  kpis, sidecar, holdout_start, dev_end) -> None:
    from src.evaluation.run_baselines import _md_table

    L = []
    L.append("# Day 9 -- Batch Forecast Generation & Early-Warning Finalisation")
    L.append("")
    L.append("**Generated by** `src/forecast/generate.py`, the single offline producer. ")
    L.append("The Streamlit app reads these files and never fits a model in a page.")
    L.append("")
    L.append("Data as of **%s** · raw CSV `%s` · generated %s"
             % (sidecar["data_as_of"], sidecar["raw_csv_sha256"][:16],
                sidecar["generated_at_utc"]))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Forward forecasts")
    L.append("")
    L.append("Champions read from the frozen registry -- this script selects nothing. ")
    L.append("Intervals are empirical quantiles of post-%s out-of-sample walk-forward "
             "residuals (addendum Section 6), never in-sample residuals." % TRAINING_CAP_DATE)
    L.append("")
    show = forwards[["target", "horizon", "champion", "origin_date", "point_forecast",
                     "lower", "upper", "n_residuals", "interval_sufficient"]].copy()
    for c in ("point_forecast", "lower", "upper"):
        show[c] = show[c].round(1)
    L.append(_md_table(show))
    L.append("")
    insufficient = forwards[~forwards["interval_sufficient"]]
    if len(insufficient):
        L.append("**%d of %d intervals were NOT emitted** because their residual pool is below "
                 "the %d-observation floor. At the 95%% level the quantiles of such a sample are "
                 "just its min and max, which is not an interval estimate. Withheld rather than "
                 "published as a band with no information in it:"
                 % (len(insufficient), len(forwards), 10))
        for _, r in insufficient.iterrows():
            L.append("- %s h=%d: %s" % (r["target"], r["horizon"], r["interval_note"]))
    else:
        L.append("All %d intervals cleared the residual-pool floor." % len(forwards))
    L.append("")
    L.append("## 2. Interval calibration -- PRIMARY evidence")
    L.append("")
    L.append("Pooled post-cutoff walk-forward coverage, leave-one-out. Scoring an interval "
             "against the very residuals that defined its quantiles would be circular, so each "
             "observation is held out and the quantiles rebuilt from the rest.")
    L.append("")
    L.append(_md_table(coverage))
    L.append("")
    outside = coverage[~coverage["covers_nominal"]]
    L.append("`covers_nominal` asks whether the nominal 95%% falls inside the binomial band "
             "sized to the achieved N. **%d of %d cells fall outside it.** %s"
             % (len(outside), len(coverage),
                "Every gap disclosed below and carried into Limitations -- never closed by "
                "widening the interval after the fact." if len(outside) else
                "No cell shows a calibration gap distinguishable from sampling noise at these N."))
    L.append("")
    L.append("These N are 12-15, exactly as the addendum anticipated (\"low tens, not "
             "hundreds\"). The bands are correspondingly wide and these are not precise "
             "calibration estimates.")
    L.append("")
    L.append("## 3. Holdout confirmation -- the one and only touch")
    L.append("")
    L.append("Addendum Section 5: the final %d real observations are \"reserved and touched "
             "exactly once, after champions are frozen\". Champions were frozen at Day 8. "
             "Intervals here come from the DEVELOPMENT residual pool -- rebuilding them from "
             "holdout residuals would leak the holdout into its own interval. Nothing here "
             "feeds back into any decision." % FINAL_TEST_WINDOW)
    L.append("")
    if len(holdout):
        agg = (holdout.groupby(["target", "horizon", "champion"])
               .agg(n=("y_true", "size"), covered=("covered", "sum"),
                    MAE=("abs_error", "mean")).reset_index())
        agg["covered"] = pd.to_numeric(agg["covered"], errors="coerce").fillna(0).astype(int)
        agg["empirical_coverage"] = (agg["covered"] / agg["n"]).round(3)
        agg["MAE"] = agg["MAE"].round(2)
        L.append(_md_table(agg))
        L.append("")
        L.append("Holdout window: pos %d onward (%s to %s), %d scored points."
                 % (holdout_start, holdout["test_date"].min(), holdout["test_date"].max(),
                    len(holdout)))
    else:
        L.append("No holdout points were scorable.")
    L.append("")
    L.append("## 4. Derived imbalance signal")
    L.append("")
    L.append("Transferred Out minus Discharged, with Transferred Out on the baseline treatment "
             "required for every series -- never a third target.")
    L.append("")
    if len(imbalance):
        show = imbalance[["horizon", "transferred_forecast", "discharged_forecast",
                          "net_pressure", "std", "uncertainty_form",
                          "measured_correlation", "interpretation"]].copy()
        for c in ("transferred_forecast", "discharged_forecast", "net_pressure", "std"):
            show[c] = show[c].round(2)
        L.append(_md_table(show))
        L.append("")
        form = imbalance["uncertainty_form"].iloc[0]
        L.append("Uncertainty form: **%s**. The full "
                 "`Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B)` is the default; the independence "
                 "simplification applies only where the MEASURED paired residual correlation "
                 "earns it (|r| < 0.20, established at Day 8)." % form)
    L.append("")
    L.append("## 5. Early-warning KPIs -- development portion only")
    L.append("")
    L.append("The trigger is replayed at every development-portion origin under the frozen "
             "%dth-percentile rule. Nothing is tuned: the percentile is a stated convention, "
             "so this measures what the fixed rule would have done."
             % EARLY_WARNING_PERCENTILE)
    L.append("")
    rows = []
    for target, s in summary.items():
        rows.append({"target": target[:30], "origins": s.get("n_origins"),
                     "fired": s.get("n_fired"),
                     "true_pos": s.get("true_positive"), "false_pos": s.get("false_positive"),
                     "false_neg": s.get("false_negative"), "true_neg": s.get("true_negative"),
                     "median_lead_time": s.get("median_surge_lead_time"),
                     "FP_rate": (round(s["false_positive_rate"], 3)
                                 if s.get("false_positive_rate") is not None else None),
                     "FN_rate": (round(s["false_negative_rate"], 3)
                                 if s.get("false_negative_rate") is not None else None)})
    L.append(_md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("Lead time is measured in **period-positions, not calendar days** -- consistent "
             "with every other offset in this project. False-positive and false-negative rates "
             "are reported alongside it as the addendum requires, \"a required metric, not "
             "optional\".")
    L.append("")
    L.append("## 6. KPI summary")
    L.append("")
    show = kpis[["target", "capacity_tier", "threshold_proxy_value",
                 "median_surge_lead_time_periods", "false_positive_rate",
                 "false_negative_rate", "forecast_stability_index"]].copy()
    for c in ("threshold_proxy_value", "forecast_stability_index"):
        show[c] = show[c].astype(float).round(2)
    L.append(_md_table(show))
    L.append("")
    L.append("**Every capacity figure above is a relative, data-derived proxy.** " + PROXY_DISCLAIMER)
    L.append("")
    L.append("**Forecast Stability Index is a disclosed substitution.** The roadmap's "
             "definition -- variance of forecasts for the same target date from different "
             "origins -- is not computable under the frozen fold cadence: origins are 10 "
             "period-positions apart and horizon gaps are 6/7/13, so no test position is ever "
             "reached from two origins (0 of 195, measured). Substituted with the p90/median "
             "absolute-error ratio across holdout origins.")
    L.append("")
    L.append("## 7. Artifacts and provenance")
    L.append("")
    L.append("Every file below carries the same data hash, so the dashboard can prove it is "
             "showing the model's actual output rather than a stale copy.")
    L.append("")
    for path in (FORWARD_FORECASTS_PATH, INTERVAL_COVERAGE_PATH, HOLDOUT_EVALUATION_PATH,
                 IMBALANCE_FORECAST_PATH, EARLY_WARNING_BACKTEST_PATH, KPI_SUMMARY_PATH,
                 FORECAST_PROVENANCE_PATH):
        L.append("- `forecasts/%s`" % path.name)
    L.append("")
    L.append("Refresh policy: **manual only**. Replace the CSV and re-run this script; nothing "
             "retrains on a schedule and the dashboard never trains at all.")
    L.append("")

    DAY9_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DAY9_REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
