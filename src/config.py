"""
config.py — Central configuration for UAC Forecasting Pipeline

Every parameter fixed in the PRE_BUILD_TECHNICAL_ADDENDUM is defined here.
Nothing is hardcoded elsewhere in the codebase.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────
# PROJECT PATHS (resolved relative to project root)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw data file
RAW_CSV_FILENAME = "HHS_Unaccompanied_Alien_Children_Program (1).csv"
RAW_CSV_PATH = PROJECT_ROOT / RAW_CSV_FILENAME

# Interim artifacts
MASTER_SERIES_PATH = DATA_INTERIM_DIR / "master_series.parquet"
# Provenance sidecar (addendum Sec. 3: content hash of the raw CSV and of the
# cleaned master series, plus a generation timestamp).
PROVENANCE_PATH = DATA_INTERIM_DIR / "provenance.json"

# ──────────────────────────────────────────────────────────────────────
# COLUMN NAMES (defined once, imported everywhere)
# ──────────────────────────────────────────────────────────────────────
COL_DATE = "Date"
COL_APPREHENDED = "Children apprehended and placed in CBP custody*"
COL_CBP_CUSTODY = "Children in CBP custody"
COL_TRANSFERRED = "Children transferred out of CBP custody"
COL_HHS_CARE = "Children in HHS Care"
COL_DISCHARGED = "Children discharged from HHS Care"

# Ordered list of all numeric columns
NUMERIC_COLS = [
    COL_APPREHENDED,
    COL_CBP_CUSTODY,
    COL_TRANSFERRED,
    COL_HHS_CARE,
    COL_DISCHARGED,
]

# Stock columns (interpolated at gap slots)
STOCK_COLS = [COL_HHS_CARE, COL_CBP_CUSTODY]

# Flow columns (left as true-missing at gap slots — NEVER zero-filled or interpolated)
FLOW_COLS = [COL_APPREHENDED, COL_TRANSFERRED, COL_DISCHARGED]

# Target columns
TARGET_1 = COL_HHS_CARE       # Stock target
TARGET_2 = COL_DISCHARGED     # Flow target

# ──────────────────────────────────────────────────────────────────────
# DATA CONSTRUCTION PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# Expected row counts per the addendum Section 2
EXPECTED_REAL_ROWS = 720
EXPECTED_TEMPLATE_POSITIONS = 767  # Sun–Thu canonical template
OFF_TEMPLATE_FRIDAYS = ["2024-09-13", "2025-04-11"]  # 2 confirmed standalone Friday rows
EXPECTED_TOTAL_POSITIONS = 769  # 767 + 2 off-template Fridays
EXPECTED_GAP_COUNT = 49  # 769 - 720 = 49 gaps

# Raw file has 450 blank trailing rows
EXPECTED_RAW_ROWS = 1170
EXPECTED_BLANK_ROWS = 450

# Reporting schedule: Sunday through Thursday (the true cadence)
REPORTING_WEEKDAYS = [6, 0, 1, 2, 3]  # Sun=6, Mon=0, Tue=1, Wed=2, Thu=3

# Date format in the raw CSV
DATE_FORMAT = "%B %d, %Y"  # e.g., "December 21, 2025"

# ──────────────────────────────────────────────────────────────────────
# REGIME / TRAINING-WINDOW PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# Training-window cap date (from structural-break analysis)
TRAINING_CAP_DATE = "2025-02-05"

# Minimum usable rows before fallback to full-expanding window
MIN_USABLE_ROWS_FLOOR = 60  # lower bound of the ~60–80 range
MAX_USABLE_ROWS_FLOOR = 80  # upper bound

# ──────────────────────────────────────────────────────────────────────
# VALIDATION PROTOCOL PARAMETERS (Section 5 of addendum)
# ──────────────────────────────────────────────────────────────────────
FINAL_TEST_WINDOW = 60       # Most recent 60 real observations
WALK_FORWARD_STEP = 10       # Period step between successive fold origins
MIN_INITIAL_TRAINING = 50    # Minimum initial training size

# Forecast horizons
FORECAST_HORIZONS = [1, 7, 14]

# The single enforced value of the addendum's "~60-80 usable rows" fallback floor.
# The lower bound of the frozen range is used as the hard threshold; the harness
# also reports how many folds would flip if MAX_USABLE_ROWS_FLOOR were enforced
# instead, so the choice inside the range is auditable rather than silent.
TRAINING_FLOOR_ROWS = MIN_USABLE_ROWS_FLOOR

# The two training-window rules every fold is evaluated under (addendum Sec 5,
# "recent-regime evaluation"). Identical by construction for folds whose origin
# precedes TRAINING_CAP_DATE.
WINDOW_RULES = ["full", "capped"]

# ----------------------------------------------------------------------
# SEASONALITY / BASELINE PARAMETERS
# ----------------------------------------------------------------------
# Reporting-week seasonal period, confirmed at Day 3 against actual within-week
# seasonality in each target's differenced level (Kruskal-Wallis p < 1e-5 for
# both targets) -- not assumed from the reporting schedule.
SEASONAL_PERIOD_M = 5

# Moving-average baseline window (periods), matching the documented 7-period
# rolling window used in feature engineering.
MOVING_AVERAGE_WINDOW = 7

# ----------------------------------------------------------------------
# FEATURE-ENGINEERING PARAMETERS
# ----------------------------------------------------------------------
# Rolling-window sizes (period-positions, never calendar days).
ROLLING_WINDOWS = [7, 14]
LAG_PERIODS = [1, 7, 14]

# Minimum non-missing observations required inside a rolling window before a
# value is emitted.
#
# RESOLVED (Day-7 open-items pass) -- was previously flagged as an unresolved
# decision. The permissive setting is KEPT, on measured evidence:
#
#   1. NO SOURCE SPECIFIES A MINIMUM. Re-verified against both governing
#      documents: zero occurrences of "min_period", "minimum observation",
#      "partial/incomplete window", or "at least N". The official Unified Mentor
#      documentation asks only for "7-day and 14-day rolling mean and variance".
#
#   2. THE DECISION TOUCHES ONLY FLOW-DERIVED FEATURES. Stock columns are fully
#      interpolated and contain zero NaN, so 0.0% of their rolling cells are
#      below a full window under either setting. The choice is therefore scoped
#      to the flow series alone.
#
#   3. STRICT WINDOWS WOULD BE DESTRUCTIVE HERE. Measured on the real series:
#        rolling_7  on a flow column -- strict voids 253 values (32.9% of rows)
#        rolling_14 on a flow column -- strict voids 437 values (56.8% of rows)
#      Feature rows carrying ANY NaN -- which is exactly what Random Forest
#      drops -- would rise from 131/755 (17.4%) to 424/755 (56.2%), costing RF a
#      further 293 training rows on a dataset the roadmap already calls
#      "moderate-sized". The addendum treats RF's NaN drop as a bounded,
#      manageable handicap; strict windows would make it the dominant effect.
#
#   4. THE PERMISSIVE SETTING MATCHES THE ADDENDUM'S OWN EXPECTATION. It states
#      ML feature rows carry NaN "from flow-column gaps". Under this setting the
#      NaN arrive via the LAG columns -- verified: all 10 folds with a NaN in the
#      RF prediction row were lag_* columns, no rolling_* column. Strict windows
#      would instead make rolling columns the dominant NaN source.
#
#   5. THE EXACT VALUE IS NOT BINDING ON THIS DATA. The smallest observation
#      count actually reached is 2 for rolling_7 and 8 for rolling_14, so a
#      floor of 1 versus 2 changes nothing here. Pinned by a test.
#
# A "rolling_7" value therefore describes the window SPAN, not the sample size
# behind it -- and the companion `rolling_{w}_nobs_{col}` columns expose the true
# count per row, so nothing is hidden from a model or a reader.
ROLLING_MIN_PERIODS_MEAN = 1
ROLLING_MIN_PERIODS_VAR = 2

# ----------------------------------------------------------------------
# DAY-5 EVALUATION ARTIFACTS
# ----------------------------------------------------------------------
FOLD_MANIFEST_PATH = FORECASTS_DIR / "walk_forward_folds.csv"
BASELINE_PREDICTIONS_PATH = FORECASTS_DIR / "baseline_predictions.csv"
BASELINE_METRICS_PATH = FORECASTS_DIR / "baseline_metrics.csv"
BASELINE_METRICS_REPORT_PATH = DOCS_DIR / "day5_baseline_metrics.md"

# ──────────────────────────────────────────────────────────────────────
# EARLY-WARNING PARAMETERS (Section 8 of addendum)
# ──────────────────────────────────────────────────────────────────────
EARLY_WARNING_PERCENTILE = 90  # 90th percentile, trailing-only

# Length of the trailing window the threshold percentile is taken over.
#
# The addendum fixes the PERCENTILE ("fixed at the 90th percentile as a stated
# convention, never searched or tuned") but not the window "recent observed
# load" is measured over. No source specifies it. Deliberately set equal to
# FINAL_TEST_WINDOW rather than chosen independently, so no new free parameter
# enters the system: that length was already frozen as covering "multiple full
# weekly cycles across all three horizons".
#
# A window is essential, not cosmetic. `Children in HHS Care` falls ~5.8x across
# the series, so a full-history percentile would sit at 2023 levels and could
# never fire in 2025 -- the threshold has to track the regime it is monitoring.
# Like the percentile, this is a STATED CONVENTION and is never tuned against
# the window used to report Surge Lead Time or Capacity Breach Probability.
EARLY_WARNING_TRAILING_WINDOW = FINAL_TEST_WINDOW

# Tier mapping (addendum Section 8): the LONGEST horizon gives the earliest,
# least confident notice.
EARLY_WARNING_TIERS = {14: "Watch", 7: "Warning", 1: "Alert"}

# Qualitative labels for the dashboard's Capacity Breach card. Addendum Section
# 8 requires a tier label rather than a bare percentage, so a data-derived proxy
# can never read as an official capacity figure.
CAPACITY_TIER_LABELS = ["Normal", "Elevated", "High"]

# ──────────────────────────────────────────────────────────────────────
# STATISTICAL MODEL SPECIFICATIONS (Day 6)
# ──────────────────────────────────────────────────────────────────────
# Specifications are PRE-REGISTERED: chosen once from diagnostics computed on
# the development portion only (the 60-observation holdout is never touched),
# then held fixed while COEFFICIENTS are refit at every walk-forward origin.
# The alternative -- re-searching orders inside each fold -- is not required by
# any source document and would cost ~2,000 extra fits.
#
# Evidence chain (all recomputed on pos 0..705, never the holdout):
#   d=1        ADF/KPSS agree both targets are non-stationary in level and
#              stationary after one difference (Day 3, docs/eda_findings.md).
#   m=5        Kruskal-Wallis within-week seasonality, p<1e-5 both targets
#              (Day 3), confirmed against the level rather than assumed from
#              the reporting schedule.
#   D=1        Seasonal ACF at lag 5 decays slowly (HHS +0.628 at lag 5,
#              +0.493 at lag 10). One seasonal difference flips it to -0.321 /
#              -0.137 and lowers the sd (128.6 -> 111.0); same pattern for
#              Discharged (+0.384 -> -0.387). A negative seasonal ACF after
#              seasonal differencing is the classic seasonal-MA signature,
#              which is why Q=1 is selected for both targets below.
#
# Order search: full grid p,q in {0,1,2} x P,Q in {0,1} x D in {0,1} with
# d=1, m=5 -- 72 candidates per target, 72/72 converged for both.
SARIMA_ORDERS = {
    # AIC-best was (2,1,2)(1,1,1,5) at 8177.8; this is (2,1,2)(0,1,1,5) at
    # 8179.4 -- delta-AIC 1.66, inside the conventional 2-point
    # indistinguishability band, AND the BIC optimum (8206.7). The addendum's
    # practical-equivalence rule ("within that margin, the simpler/more stable
    # candidate wins over the numerically lowest one") selects the simpler model.
    TARGET_1: {"order": (2, 1, 2), "seasonal_order": (0, 1, 1, 5)},
    # AIC and BIC agree outright: 6351.1 / 6369.3, best on both criteria.
    TARGET_2: {"order": (0, 1, 2), "seasonal_order": (0, 1, 1, 5)},
}

# Exponential Smoothing -> ETSModel (state-space), additive throughout.
# Multiplicative error/seasonal components are excluded on evidence, not taste:
# `Children discharged from HHS Care` reaches 0 (2025-11-30) and multiplicative
# forms are undefined at zero.
ETS_SPECS = {
    # AIC and BIC both select damped additive trend + additive seasonal.
    TARGET_1: {"error": "add", "trend": "add", "damped_trend": True,
               "seasonal": "add", "seasonal_periods": 5},
    # AIC is a dead heat between additive trend (6914.74) and no trend
    # (6914.82) -- delta 0.08 -- and BIC prefers no trend (6955.3 vs 6964.2).
    # Practical equivalence again selects the simpler specification; a flow
    # series of daily discharges carrying no persistent trend is also
    # substantively sensible.
    TARGET_2: {"error": "add", "trend": None, "damped_trend": False,
               "seasonal": "add", "seasonal_periods": 5},
}

# ETSModel needs this explicitly: the classic `ExponentialSmoothing` class
# silently returns an ALL-NaN forecast on the masked (flow) series rather than
# raising. Verified live and pinned by a regression test.
ETS_MISSING_POLICY = "drop"

# Nominal level for the NATIVE SARIMAX/ETS confidence intervals. Addendum
# Section 6: these are a SECONDARY DIAGNOSTIC only -- the primary intervals are
# the empirical residual-quantile ones built at Day 9.
NATIVE_CI_ALPHA = 0.05

# Day-6 artifacts
STATISTICAL_PREDICTIONS_PATH = FORECASTS_DIR / "statistical_predictions.csv"
STATISTICAL_METRICS_PATH = FORECASTS_DIR / "statistical_metrics.csv"
STATISTICAL_METRICS_REPORT_PATH = DOCS_DIR / "day6_statistical_metrics.md"

# ──────────────────────────────────────────────────────────────────────
# MACHINE-LEARNING MODEL PARAMETERS (Day 7)
# ──────────────────────────────────────────────────────────────────────
# Deliberately modest and FIXED -- the roadmap requires tuning to be "light and
# time-boxed, no exhaustive search", and Part 5 warns these families "can overfit
# with too many correlated lag/rolling features on ~700 rows". No hyperparameter
# search is run: a search inside the walk-forward would need a nested inner loop
# to stay leakage-free, which is beyond the frozen scope.
#
# RANDOM_SEED is threaded into every stochastic fit (addendum Section 3).
RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,          # shallow: ~145-700 training rows depending on window
    "min_samples_leaf": 5,   # guards against memorising single observations
    "max_features": "sqrt",  # decorrelates highly-collinear lag/rolling features
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

HIST_GRADIENT_BOOSTING_PARAMS = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
    "l2_regularization": 1.0,
    "early_stopping": False,  # an internal validation split would be a RANDOM
                              # split of a time series -- not permitted here
    "random_state": RANDOM_SEED,
}

# Random Forest drops feature rows containing NaN, and the count is logged
# (addendum Sections 2 and 4).
#
# NOTE, recorded honestly: scikit-learn gained native missing-value support for
# tree ensembles in 1.4, and RandomForestRegressor on the installed 1.9.0 no
# longer raises on NaN input. The addendum was written when it did. The drop is
# therefore retained as the governing METHODOLOGICAL rule -- not as a technical
# necessity -- and HistGradientBoostingRegressor keeps its NaN rows, exactly as
# the addendum specifies. This preserves the deliberate contrast between the two
# families rather than silently collapsing it.
RANDOM_FOREST_DROPS_NAN_ROWS = True

# What Random Forest does when the PREDICTION row itself contains a NaN.
#
# No source document specifies this. The addendum's rule is about TRAINING rows
# ("ML feature rows with NaN inputs are dropped"); at prediction time there is
# no row to drop, only a forecast to produce or withhold. Measured: 10 of the 65
# folds have a NaN in the feature row at their training cutoff, always from a
# flow-derived lag.
#
# Set False, i.e. RF forecasts anyway, because:
#   * abstaining would be an ADDITIONAL handicap invented here rather than one
#     the sources impose;
#   * common support would then drop those 10 folds for EVERY model, shrinking
#     the whole 7-way comparison by 15% to accommodate one family;
#   * scikit-learn >= 1.4 gives trees defined, deterministic missing-value
#     handling at predict time.
# The specified handicap -- the ~20% of training pairs RF loses to the row drop
# -- is retained and logged.
RANDOM_FOREST_ABSTAINS_ON_NAN_PREDICTION_ROW = False

# Minimum usable training pairs before an ML model refuses to fit for a given
# fold/horizon. Independent of, and additional to, the ~60-80 usable-row
# training-window floor: direct multi-horizon fitting loses a further `lead`
# rows off the end of the window, because a pair needs its label to fall at or
# before the training cutoff.
ML_MIN_TRAINING_PAIRS = 30

# Day-7 artifacts
ML_PREDICTIONS_PATH = FORECASTS_DIR / "ml_predictions.csv"
ML_METRICS_PATH = FORECASTS_DIR / "ml_metrics.csv"
ML_RESIDUALS_PATH = FORECASTS_DIR / "oos_residuals.csv"
ML_METRICS_REPORT_PATH = DOCS_DIR / "day7_ml_metrics.md"
FULL_COMPARISON_PATH = FORECASTS_DIR / "full_model_comparison.csv"

# ──────────────────────────────────────────────────────────────────────
# MODEL PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# The seven evaluated families, all now implemented and evaluated (Days 4-7),
# plus the post-hoc ensemble computed at Day 8.
MODEL_FAMILIES = [
    "naive",
    "seasonal_naive",
    "moving_average",
    "exponential_smoothing",  # ETSModel with missing='drop' for Discharged
    "sarima",                 # SARIMAX
    "random_forest",          # RandomForestRegressor
    "gradient_boosting",      # HistGradientBoostingRegressor
    "ensemble",               # Post-hoc average of champion stat + champion ML
]

BASELINE_MODELS = ["naive", "seasonal_naive", "moving_average"]

# ----------------------------------------------------------------------
# CHAMPION-SELECTION RULE
# ----------------------------------------------------------------------
# Complexity ordering, used ONLY to break practical ties. Measured from the
# persisted Day-6/7 artifacts, not asserted:
#     naive / seasonal_naive / moving_average   0 estimated parameters
#     sarima                                    4-6 free parameters
#     exponential_smoothing                     8-11 free parameters
#     random_forest                             300 trees / ~29,700 nodes
#     gradient_boosting                         300 boosted trees
# Within the zero-parameter baselines the order reflects how much configured
# structure each carries: naive none, seasonal_naive a period m, moving_average
# a window w.
MODEL_COMPLEXITY_ORDER = [
    "naive",
    "seasonal_naive",
    "moving_average",
    "sarima",
    "exponential_smoothing",
    "random_forest",
    "gradient_boosting",
    "ensemble",
]

# Practical-equivalence margin (addendum Section 5: "within that margin, the
# simpler/more stable candidate wins over the numerically lowest one").
#
# The margin is NOT a fixed percentage. It is a paired bootstrap over
# per-observation absolute errors on the SAME test points, so it adapts to the
# sample size actually achieved -- which matters here because the recent-regime
# pools are 12-15 observations per cell. Measured on those pools: 5 of 6
# target/horizon cells show NO distinguishable difference between the best
# baseline and the best non-baseline candidate, and the single distinguishable
# cell favours the BASELINE. A fixed-percentage margin would have manufactured
# winners out of noise at these N.
PRACTICAL_EQUIVALENCE_RESAMPLES = 10000
PRACTICAL_EQUIVALENCE_LEVEL = 0.95
PRACTICAL_EQUIVALENCE_SEED = RANDOM_SEED

# A baseline is a first-class champion candidate. If no model clears the
# baseline-beating gate, the best BASELINE is the champion and is reported as
# such -- never overridden by a more complex model that failed the gate.
BASELINES_ARE_ELIGIBLE_CHAMPIONS = True

# The scope champion selection is decided on. Addendum Section 5: "If the two
# rankings disagree, the restricted (recent-regime) ranking governs champion
# selection." Common support is applied so every candidate is compared on
# identical test points.
SELECTION_SCOPE = "post_cutoff_common_support"

# The capped rule IS the deployed configuration: forecasts generated after
# 2025-02-05 train on post-cutoff data by the addendum's own training rule, so
# champions must be chosen under the same rule they will run under.
SELECTION_WINDOW_RULE = "capped"

# Bias tie-break. |mean signed error| / MAE is bounded in [0, 1] and equals 1
# only when every error points the same way, so it measures what fraction of a
# model's typical error is a fixed directional offset. Above this share, and
# only when a bootstrap CI on the signed error also excludes zero, a candidate
# is passed over IN FAVOUR OF a practically-equivalent alternative -- never
# eliminated outright, since the roadmap asks for "acceptable" bias, not none.
BIAS_RATIO_TIE_BREAK_THRESHOLD = 0.5

# ----------------------------------------------------------------------
# DAY-8 ARTIFACTS
# ----------------------------------------------------------------------
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# ----------------------------------------------------------------------
# UNCERTAINTY (Day 9) -- addendum Section 6
# ----------------------------------------------------------------------
# Nominal level for the PRIMARY intervals: empirical quantiles of out-of-sample
# walk-forward residuals, restricted to folds with origin on or after the
# training cutoff. Matched to NATIVE_CI_ALPHA so the primary and the secondary
# diagnostic are quoted at the same level and are directly comparable.
EMPIRICAL_INTERVAL_ALPHA = NATIVE_CI_ALPHA

# Below this many residuals an interval is NOT emitted. At the 95% level the
# 2.5th and 97.5th percentiles of n<10 are just the sample min and max, which
# is not an interval estimate in any meaningful sense -- better to return
# nothing and say so than to publish a band with no information in it.
MIN_RESIDUALS_FOR_INTERVAL = 10

# ----------------------------------------------------------------------
# DAY-9 ARTIFACTS
# ----------------------------------------------------------------------
FORWARD_FORECASTS_PATH = FORECASTS_DIR / "forward_forecasts.csv"
INTERVAL_COVERAGE_PATH = FORECASTS_DIR / "interval_coverage.csv"
HOLDOUT_EVALUATION_PATH = FORECASTS_DIR / "holdout_evaluation.csv"
IMBALANCE_FORECAST_PATH = FORECASTS_DIR / "imbalance_forecast.csv"
EARLY_WARNING_BACKTEST_PATH = FORECASTS_DIR / "early_warning_backtest.csv"
KPI_SUMMARY_PATH = FORECASTS_DIR / "kpi_summary.csv"
FORECAST_PROVENANCE_PATH = FORECASTS_DIR / "provenance.json"
DAY9_REPORT_PATH = DOCS_DIR / "day9_forecast_generation.md"
CHAMPION_METRICS_PATH = FORECASTS_DIR / "champion_selection.csv"
ENSEMBLE_PREDICTIONS_PATH = FORECASTS_DIR / "ensemble_predictions.csv"
IMBALANCE_CORRELATION_PATH = FORECASTS_DIR / "imbalance_residual_correlation.csv"
SELECTION_RATIONALE_PATH = DOCS_DIR / "model_selection_rationale.md"

# ----------------------------------------------------------------------
# DERIVED IMBALANCE SIGNAL (addendum Section 6)
# ----------------------------------------------------------------------
# Transferred Out is a DERIVED-signal component, not a third target: its forward
# value uses the same baseline treatment required for every series, never a full
# champion-selection track.
IMBALANCE_COMPONENT = COL_TRANSFERRED
IMBALANCE_BASELINE_MODELS = ["seasonal_naive", "moving_average"]

# Below this |correlation| the simplified independence form of
# Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B) may be used. The addendum sets the
# pre-registered EXPECTATION of near-independence from the raw-series proxy
# (0.657 raw / 0.074 first-difference / 0.112 detrended) but is explicit that
# this "is a prior, not the final answer" -- the decision is made from the
# measured paired out-of-sample residual correlation at Day 8.
IMBALANCE_INDEPENDENCE_THRESHOLD = 0.2
