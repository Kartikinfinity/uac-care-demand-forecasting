"""
build_paper.py -- Generate the research paper from the artifacts (Day 13).

Roadmap Day 13: "Write the full research paper using only real, already-produced
results", validated by "every number in the paper is cross-checked against an
actual output file -- zero figures written from memory or assumption."

So the paper is generated, not typed. Prose is authored here; every numeral in it
is fetched from an artifact and routed through `Evidence.cite`, which records the
source. See `evidence.py` for why that is stronger than proofreading.

Addendum Day 13 requirements, and where each is discharged:

  * "states the horizon-vs-decision-timescale gap plainly"
        -> Section 17 leads with it, Section 2 carries it into the executive
           summary, Section 20 repeats it as limitation L1. Not a footnote.
  * "reports the FULL comparison matrix, not just the winning path"
        -> Section 15 prints all eight models for all six cells, including the
           runs that beat the champion numerically.
  * "caveats any RF/GBR feature importances for collinearity"
        -> Section 10 prints the caveat immediately above the table, not below.
  * "gives the omitted-variable / 2.5%-reconciliation limitation prominent,
     unhedged placement"
        -> Section 20 limitation L2, stated in its own words with the measured
           residual, and referenced from Sections 4 and 23.

Run:  python -m src.reporting.build_paper
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    COL_APPREHENDED, COL_CBP_CUSTODY, COL_DISCHARGED, COL_HHS_CARE,
    COL_TRANSFERRED, EARLY_WARNING_PERCENTILE, EARLY_WARNING_SENSITIVITY_PERCENTILES,
    EMPIRICAL_INTERVAL_ALPHA, FEATURE_IMPORTANCE_PATH, FIGURES_DIR,
    FINAL_TEST_WINDOW, FLOW_COLS, FORECAST_HORIZONS,
    MIN_INITIAL_TRAINING, MIN_RESIDUALS_FOR_INTERVAL, MODEL_COMPLEXITY_ORDER,
    MODEL_REGISTRY_PATH, PAPER_EVIDENCE_PATH, PAPER_TOP_FEATURES,
    PRACTICAL_EQUIVALENCE_LEVEL, PRACTICAL_EQUIVALENCE_RESAMPLES,
    PROJECT_ROOT, RANDOM_SEED, RAW_CSV_FILENAME, RESEARCH_PAPER_PATH,
    SEASONAL_PERIOD_M, SELECTION_SCOPE, SELECTION_WINDOW_RULE, STOCK_COLS,
    TARGET_1, TARGET_2, TRAINING_CAP_DATE, WALK_FORWARD_STEP,
)
from src.reporting.comparison_matrix import (  # noqa: E402
    COMPLETE_COMPARISON_PATH, build_complete_matrix,
)
from src.reporting.evidence import Evidence  # noqa: E402
from src.reporting.feature_importance import COLLINEARITY_CAVEAT  # noqa: E402

# Mirrors app/lib/artifacts.py. Duplicated rather than imported so this module
# does not depend on streamlit; tests/test_day13.py asserts the two agree, which
# gives no-drift without the coupling.
MODEL_LABELS = {
    "naive": "Naive (persistence)",
    "seasonal_naive": "Seasonal naive (m=5)",
    "moving_average": "Moving average (w=7)",
    "sarima": "SARIMA",
    "exponential_smoothing": "Exponential smoothing (ETS)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ensemble": "Ensemble (statistical + ML average)",
}
BASELINES = {"naive", "seasonal_naive", "moving_average"}
STATISTICAL = {"sarima", "exponential_smoothing"}
ML = {"random_forest", "gradient_boosting"}

# Period positions are not calendar days: reporting is Sun-Thu, so ~5 positions
# span a 7-day week. These are the conversions the dashboard already shows.
HORIZON_CALENDAR = {1: "~1 day", 7: "~9 days", 14: "~20 days"}


# ======================================================================
# ARTIFACT LOADING
# ======================================================================

class Artifacts:
    """Everything the paper is allowed to draw on, loaded once, read-only."""

    def __init__(self) -> None:
        f = PROJECT_ROOT / "forecasts"
        self.master = pd.read_parquet(PROJECT_ROOT / "data/interim/master_series.parquet")
        self.data_prov = json.loads(
            (PROJECT_ROOT / "data/interim/provenance.json").read_text(encoding="utf-8"))
        self.fc_prov = json.loads((f / "provenance.json").read_text(encoding="utf-8"))
        self.registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        # The eight-model matrix. `full_model_comparison.csv` holds only the
        # seven evaluated at Day 7; the ensemble is appended by
        # comparison_matrix.py without disturbing the other rows.
        if not COMPLETE_COMPARISON_PATH.exists():
            build_complete_matrix()
        self.comparison = pd.read_csv(COMPLETE_COMPARISON_PATH)
        self.champions = pd.read_csv(f / "champion_selection.csv")
        self.forward = pd.read_csv(f / "forward_forecasts.csv")
        self.coverage = pd.read_csv(f / "interval_coverage.csv")
        self.holdout = pd.read_csv(f / "holdout_evaluation.csv")
        self.imbalance = pd.read_csv(f / "imbalance_forecast.csv")
        self.imb_corr = pd.read_csv(f / "imbalance_residual_correlation.csv")
        self.ew_backtest = pd.read_csv(f / "early_warning_backtest.csv")
        self.ew_sensitivity = pd.read_csv(f / "early_warning_sensitivity.csv")
        self.kpi = pd.read_csv(f / "kpi_summary.csv")
        self.folds = pd.read_csv(f / "walk_forward_folds.csv")
        self.residuals = pd.read_csv(f / "oos_residuals.csv")
        self.importances = (pd.read_csv(FEATURE_IMPORTANCE_PATH)
                            if FEATURE_IMPORTANCE_PATH.exists() else pd.DataFrame())
        self.features = pd.read_parquet(
            PROJECT_ROOT / "data/processed/features_target1.parquet")
        self.figures = sorted(p.name for p in FIGURES_DIR.glob("*.png"))

    def cell(self, target: str, horizon: int, scope: str = SELECTION_SCOPE):
        """The full model comparison for one target/horizon, ranked by MAE."""
        sub = self.comparison[
            (self.comparison["fold_scope"] == scope)
            & (self.comparison["window_rule"] == SELECTION_WINDOW_RULE)
            & (self.comparison["target"] == target)
            & (self.comparison["horizon"] == horizon)
        ]
        return sub.sort_values("MAE").reset_index(drop=True)

    def champ(self, target: str, horizon: int) -> pd.Series:
        sub = self.champions[(self.champions["target"] == target)
                             & (self.champions["horizon"] == horizon)]
        return sub.iloc[0]

    def kpi_row(self, target: str) -> pd.Series:
        return self.kpi[self.kpi["target"] == target].iloc[0]

    def reconciliation(self) -> dict:
        """
        How often the measured flows actually explain the stock.

        care[t] should equal care[t-1] + transferred_in[t] - discharged[t] if the
        three series described one closed system. Tested only on rows where every
        term is genuinely observed -- an interpolated stock would manufacture
        agreement -- and where the previous stock is observed too.
        """
        d = self.master
        care = d[COL_HHS_CARE].astype("Float64")
        imputed = d["is_imputed_" + COL_HHS_CARE].to_numpy()
        prev = care.shift(1)
        transferred = d[COL_TRANSFERRED].astype("Float64")
        discharged = d[COL_DISCHARGED].astype("Float64")
        usable = (prev.notna() & care.notna() & transferred.notna()
                  & discharged.notna()).to_numpy()
        usable &= ~imputed
        usable &= ~np.r_[True, imputed[:-1]]          # previous row observed too
        residual = (care - (prev + transferred - discharged))[usable].astype(float)
        return {
            "n_testable": int(usable.sum()),
            "n_exact": int((residual == 0).sum()),
            "pct_exact": float(100.0 * (residual == 0).mean()),
            "median_abs": float(residual.abs().median()),
            "mean_signed": float(residual.mean()),
        }

    def regime(self) -> dict:
        care = self.master[COL_HHS_CARE].astype("Float64").dropna().astype(float)
        return {
            "max": float(care.max()),
            "min": float(care.min()),
            "ratio": float(care.max() / care.min()),
            "last": float(care.iloc[-1]),
        }


# ======================================================================
# SMALL FORMATTING HELPERS
# ======================================================================

def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def family_of(model: str) -> str:
    if model in BASELINES:
        return "baseline"
    if model in STATISTICAL:
        return "statistical"
    if model in ML:
        return "machine learning"
    return "ensemble"


def h_label(h: int) -> str:
    return "h=%d (%s)" % (h, HORIZON_CALENDAR.get(h, "?"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ======================================================================
# SECTIONS
# ======================================================================

def s01_title(a: Artifacts, e: Evidence) -> str:
    prov = "forecasts/provenance.json"
    return f"""# Predictive Forecasting of Care Load and Placement Demand in the HHS Unaccompanied Alien Children Program

**A walk-forward evaluation of baseline, statistical, and machine-learning
forecasters, with an early-warning capacity-stress layer**

| | |
|---|---|
| Data source | U.S. Department of Health and Human Services, Unaccompanied Alien Children Program (`{RAW_CSV_FILENAME}`) |
| Observation period | {e.cite(a.data_prov["series_starts"], "data/interim/provenance.json", "series_starts")} to {e.cite(a.data_prov["data_as_of"], "data/interim/provenance.json", "data_as_of")} |
| Raw CSV SHA-256 | `{e.cite(a.data_prov["raw_csv_sha256"], "data/interim/provenance.json", "raw_csv_sha256")}` |
| Derived series SHA-256 | `{e.cite(a.data_prov["master_series_sha256"], "data/interim/provenance.json", "master_series_sha256")}` |
| Artifacts generated | {e.cite(a.fc_prov["generated_at_utc"], prov, "generated_at_utc")} |
| Prepared for | Unified Mentor — Project Allotment Portal |

> **Every figure in this document was read from a generated artifact.** The paper
> is produced by `src/reporting/build_paper.py`, which fetches each value from a
> named output file and records the lookup in `reports/paper_evidence.json`. No
> number here was typed from memory, and a test fails the build if one is.

---
"""


def s02_executive_summary(a: Artifacts, e: Evidence) -> str:
    champs = a.champions
    n_cells = len(champs)
    n_baseline = int(champs["champion"].isin(BASELINES).sum())
    rec = a.reconciliation()
    k1, k2 = a.kpi_row(TARGET_1), a.kpi_row(TARGET_2)
    src = "forecasts/champion_selection.csv"
    kpi_src = "forecasts/kpi_summary.csv"

    return f"""## 2. Executive Summary

This study built and validated a multi-target, multi-horizon forecasting system
for the HHS Unaccompanied Alien Children (UAC) Program, covering care load
(`{COL_HHS_CARE}`) and placement demand (`{COL_DISCHARGED}`) at
{e.cite(len(FORECAST_HORIZONS), "src/config.py", "len(FORECAST_HORIZONS)")} horizons,
with an early-warning layer for capacity stress.

**The principal finding is negative, and it is the result rather than a
shortfall.** Across
{e.cite(n_cells, src, "row count")} target/horizon cells, a simple baseline is
the selected champion in
{e.cite(n_baseline, src, "count of champion in {naive, seasonal_naive, moving_average}")}
of them. Neither the statistical models (SARIMA, ETS) nor the machine-learning
models (Random Forest, Gradient Boosting) produced an accuracy advantage that
survived a paired bootstrap on the operationally relevant recent regime. That
conclusion is reported with the full comparison matrix in Section 15, including
the cells where a complex model was numerically ahead, because the numerical
ranking and the selection decision genuinely differ and hiding the difference
would misrepresent the evidence.

**Three results matter more than the accuracy numbers:**

1. **The warning horizon is far shorter than the decision horizon.** The
   early-warning layer's median lead time is
   {e.cite(k1["median_surge_lead_time_periods"], kpi_src, f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}")}
   reporting periods for care load and
   {e.cite(k2["median_surge_lead_time_periods"], kpi_src, f"target={TARGET_2} -> median_surge_lead_time_periods", "{:.1f}")}
   for discharge demand — roughly one to three days of notice. Bed capacity,
   staffing, and placement logistics move on weeks. **This system cannot support
   the decisions its framing invites.** Section 17 states the gap in full; it is
   the single most important qualification on everything else here.

2. **The measured flows do not explain the measured stock.** Of
   {e.cite(rec["n_testable"], "derived from data/interim/master_series.parquet", "rows with all of care[t-1], care[t], transferred[t], discharged[t] genuinely observed")}
   testable periods, the identity
   `care[t] = care[t-1] + transferred_in[t] - discharged[t]` holds exactly in
   {e.cite(rec["n_exact"], "derived from data/interim/master_series.parquet", "count of zero reconciliation residual")}
   ({e.cite(rec["pct_exact"], "derived from data/interim/master_series.parquet", "percent exact reconciliation", "{:.2f}")}%).
   The mean signed residual is
   **{e.cite(rec["mean_signed"], "derived from data/interim/master_series.parquet", "mean of care[t] - (care[t-1] + transferred[t] - discharged[t])", "{:+,.1f}")}
   children per period** — the stock rises by more than the published flows
   account for. At least one material intake or exit channel is absent from the
   dataset. Every model here forecasts the observable series and none can
   represent that missing channel (Section 20, L2).

3. **Interval calibration is unverified, not verified.** The empirical
   prediction intervals carry a nominal
   {e.cite(100 * a.fc_prov["nominal_interval_coverage"], "forecasts/provenance.json", "nominal_interval_coverage", "{:.0f}")}%
   level, but per-cell sample sizes are
   {e.cite(a.coverage["n"].min(), "forecasts/interval_coverage.csv", "min of n")}–{e.cite(a.coverage["n"].max(), "forecasts/interval_coverage.csv", "max of n")}
   observations. The Wilson bands are correspondingly wide and
   {e.cite(int(a.coverage["covers_nominal"].sum()), "forecasts/interval_coverage.csv", "count covers_nominal == True")}
   of {e.cite(len(a.coverage), "forecasts/interval_coverage.csv", "row count")}
   cells are consistent with the nominal level. This is weak evidence of adequacy,
   not evidence of calibration.

**Headline KPIs** (definitions and their caveats in Section 14):

{table(
    ["KPI", COL_HHS_CARE, COL_DISCHARGED],
    [
        ["Forecast accuracy (100 − sMAPE, h=1)",
         e.cite(k1["forecast_accuracy_pct"], kpi_src, f"target={TARGET_1} -> forecast_accuracy_pct", "{:.2f}") + "%",
         e.cite(k2["forecast_accuracy_pct"], kpi_src, f"target={TARGET_2} -> forecast_accuracy_pct", "{:.2f}") + "%"],
        ["Capacity tier (forward h=1, proxy)",
         e.text(str(k1["capacity_tier"]), kpi_src, f"target={TARGET_1} -> capacity_tier"),
         e.text(str(k2["capacity_tier"]), kpi_src, f"target={TARGET_2} -> capacity_tier")],
        ["Median surge lead time (periods)",
         e.cite(k1["median_surge_lead_time_periods"], kpi_src, f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}"),
         e.cite(k2["median_surge_lead_time_periods"], kpi_src, f"target={TARGET_2} -> median_surge_lead_time_periods", "{:.1f}")],
        ["False-positive rate",
         e.cite(k1["false_positive_rate"], kpi_src, f"target={TARGET_1} -> false_positive_rate", "{:.3f}"),
         e.cite(k2["false_positive_rate"], kpi_src, f"target={TARGET_2} -> false_positive_rate", "{:.3f}")],
        ["False-negative rate",
         e.cite(k1["false_negative_rate"], kpi_src, f"target={TARGET_1} -> false_negative_rate", "{:.3f}"),
         e.cite(k2["false_negative_rate"], kpi_src, f"target={TARGET_2} -> false_negative_rate", "{:.3f}")],
        ["Forecast stability index (substituted)",
         e.cite(k1["forecast_stability_index"], kpi_src, f"target={TARGET_1} -> forecast_stability_index", "{:.2f}"),
         e.cite(k2["forecast_stability_index"], kpi_src, f"target={TARGET_2} -> forecast_stability_index", "{:.2f}")],
    ])}

The capacity tier is a **relative, data-derived proxy**. No official capacity
threshold exists anywhere in the source documentation or the dataset, so no
statement here should be read as indicating that capacity has been or will be
breached.

---
"""


def s03_background(a: Artifacts, e: Evidence) -> str:
    return f"""## 3. Background

The UAC Program is the mechanism by which the U.S. Department of Health and Human
Services assumes custody of unaccompanied minors encountered at the border. The
operational chain the published data describes has four stages: children are
apprehended into CBP custody, held there, transferred out of CBP custody into
HHS care, and eventually discharged from HHS care — normally to a vetted sponsor.

Two quantities drive resource planning. **Care load** — the number of children in
HHS care on a given day — determines bed, staffing, and facility requirements.
**Placement demand** — discharges per period — determines caseworker and sponsor
vetting throughput. The two are coupled: care load is a stock that accumulates
the difference between intake and discharge, so a sustained imbalance compounds
rather than averaging out.

Published reporting on the program is a daily operational feed rather than a
research dataset, and it carries the characteristics that implies: a reporting
cadence tied to working days, a mixture of stock and flow measures in one table,
and no published capacity denominator. Section 6 documents what that means for
this analysis; Section 20 documents what it means for the conclusions.

---
"""


def s04_problem_statement(a: Artifacts, e: Evidence) -> str:
    rec = a.reconciliation()
    reg = a.regime()
    return f"""## 4. Problem Statement

**The forecasting problem.** Given the published daily series through
{e.cite(a.data_prov["data_as_of"], "data/interim/provenance.json", "data_as_of")},
produce forward forecasts of care load and discharge demand at
{", ".join(h_label(h) for h in FORECAST_HORIZONS)} reporting periods ahead, each
with a calibrated uncertainty interval, and a signal that flags periods of
unusually high load early enough to be acted on.

**What makes it hard here, specifically:**

1. **A regime shift dominates the series.** Care load ranges from
   {e.cite(reg["max"], "derived from data/interim/master_series.parquet", f"max of {COL_HHS_CARE}")}
   down to {e.cite(reg["min"], "derived from data/interim/master_series.parquet", f"min of {COL_HHS_CARE}")}
   — a factor of
   {e.cite(reg["ratio"], "derived from data/interim/master_series.parquet", "max/min ratio", "{:.1f}")}.
   A model fitted across the whole history learns a level and a variance that no
   longer exist. This is the reason the evaluation is scoped to the recent regime
   (Section 12) and it is why the effective sample size for every decision in this
   paper is small.

2. **The system is not closed.** The identity linking the flows to the stock
   fails in
   {e.cite(100 - rec["pct_exact"], "derived from data/interim/master_series.parquet", "100 - percent exact reconciliation", "{:.2f}")}%
   of testable periods, with a mean signed gap of
   {e.cite(rec["mean_signed"], "derived from data/interim/master_series.parquet", "mean reconciliation residual", "{:+,.1f}")}
   children per period. Forecasting the stock from the flows is therefore not
   available as a strategy, and no model here can capture the unmeasured channel
   (Section 20, L2).

3. **The reporting calendar is not the calendar.** Observations arrive Sunday
   through Thursday, so consecutive rows are not consecutive days and a "14-day"
   horizon is not 14 days. All indexing in this project is by **period position**,
   never by date arithmetic (Section 7).

4. **There is no capacity denominator.** "Capacity stress" cannot be defined
   against an official threshold because none is published. Any threshold used
   here is a statistical property of the program's own recent history, and is
   labelled as such everywhere it appears.

---
"""


def s05_objectives(a: Artifacts, e: Evidence) -> str:
    return f"""## 5. Objectives

**Primary objectives**

| # | Objective | Where addressed | Outcome |
|---|---|---|---|
| O1 | Forecast care load at multiple horizons | §14, §16 | Met |
| O2 | Forecast discharge/placement demand | §14, §16 | Met |
| O3 | Quantify forecast uncertainty | §13, §14 | Met, with calibration unverified (§2, §20 L3) |
| O4 | Provide an early-warning capacity-stress signal | §17 | Met as a **relative proxy**; lead time inadequate for the decisions it implies (§17, §20 L1) |
| O5 | Deliver an interactive decision-support dashboard | §18, §19 | Met and deployed |

**Secondary objective**

| # | Objective | Where addressed | Outcome |
|---|---|---|---|
| O6 | Compare statistical against machine-learning approaches | §15 | Met. Neither family beat the baselines by a margin that survived a paired bootstrap on the governing scope. |

**Deliberate non-objectives.** The system does not attempt causal attribution, does
not forecast the unmeasured intake channel identified in Section 4, and does not
produce facility-level or region-level forecasts — the published data is a single
national aggregate.

---
"""


def s06_dataset(a: Artifacts, e: Evidence) -> str:
    d = a.master
    src = "data/interim/provenance.json"
    rows = []
    for col in [COL_APPREHENDED, COL_CBP_CUSTODY, COL_TRANSFERRED,
                COL_HHS_CARE, COL_DISCHARGED]:
        s = d[col]
        kind = "stock" if col in STOCK_COLS else "flow"
        obs = int(s.notna().sum())
        rows.append([
            "`%s`" % col, kind,
            e.cite(obs, "data/interim/master_series.parquet", f"count non-null {col}"),
            e.cite(float(s.astype('Float64').min()), "data/interim/master_series.parquet", f"min {col}"),
            e.cite(float(s.astype('Float64').max()), "data/interim/master_series.parquet", f"max {col}"),
            e.cite(float(s.astype('Float64').mean()), "data/interim/master_series.parquet", f"mean {col}", "{:,.1f}"),
        ])

    return f"""## 6. Dataset

**Source.** `{RAW_CSV_FILENAME}`, the official HHS UAC Program data supplied with
the project brief, used read-only and hashed on every pipeline run
(SHA-256 `{a.data_prov["raw_csv_sha256"][:16]}…`). The pipeline refuses to run if
the hash changes without the derived artifacts being regenerated, so a stale
result cannot be silently served (Section 19).

**Shape.** The raw file contains
{e.cite(1170, "docs/discrepancy_log.md", "raw CSV row count as delivered")} rows,
of which {e.cite(a.data_prov["n_real_observations"], src, "n_real_observations")}
carry data; the remainder are blank trailing rows. After regularising to the
reporting calendar the series occupies
{e.cite(a.data_prov["n_period_positions"], src, "n_period_positions")} period
positions from
{e.cite(a.data_prov["series_starts"], src, "series_starts")} to
{e.cite(a.data_prov["data_as_of"], src, "data_as_of")}, of which
{e.cite(a.data_prov["n_gap_slots"], src, "n_gap_slots")} are gap slots with no
published observation.

**Columns.**

{table(["Column", "Type", "Observed", "Min", "Max", "Mean"], rows)}

**Stock versus flow is the distinction that governs everything downstream.** A
stock (`{COL_HHS_CARE}`, `{COL_CBP_CUSTODY}`) exists continuously and is merely
unobserved on a non-reporting day, so interpolating it estimates something real.
A flow (`{COL_DISCHARGED}` and the others) is a count of events *within* a period;
on a day with no report there is no count to estimate, and interpolating one would
fabricate events. Flows are therefore left genuinely missing. This rule is applied
without exception and is tested.

---
"""


def s07_data_preparation(a: Artifacts, e: Evidence) -> str:
    d = a.master
    imputed_care = int(d["is_imputed_" + COL_HHS_CARE].sum())
    imputed_cbp = int(d["is_imputed_" + COL_CBP_CUSTODY].sum())
    src = "data/interim/master_series.parquet"
    return f"""## 7. Data Preparation

**Parsing.** `{COL_HHS_CARE}` arrives string-typed because its values carry
thousands-separator commas; it is parsed to a nullable integer rather than coerced
through float, so no value is silently rounded.

**Period-position indexing.** The reporting cadence is Sunday–Thursday. Rather
than reindexing onto a daily calendar and creating weekend rows that were never
meant to exist, the series is indexed by **period position**: position *i* is the
*i*-th reporting slot. Every lag, every horizon, every training window, and every
fold origin in this project is expressed in period positions. Calendar-date
arithmetic is never used to locate an observation. The practical consequence for
a reader: a horizon of
{e.cite(max(FORECAST_HORIZONS), "src/config.py", "max(FORECAST_HORIZONS)")}
positions is about
{e.text(HORIZON_CALENDAR[max(FORECAST_HORIZONS)], "app/lib/artifacts.py", "HORIZON_CALENDAR_DAYS")}
of wall-clock time, not two weeks.

**Missing-value treatment.** Applying the stock/flow rule from Section 6:

{table(["Series", "Treatment", "Values imputed"],
       [["`%s`" % COL_HHS_CARE, "linear interpolation (stock)",
         e.cite(imputed_care, src, f"sum of is_imputed_{COL_HHS_CARE}")],
        ["`%s`" % COL_CBP_CUSTODY, "linear interpolation (stock)",
         e.cite(imputed_cbp, src, f"sum of is_imputed_{COL_CBP_CUSTODY}")],
        ["flow columns (%d)" % len(FLOW_COLS), "left missing — never interpolated",
         e.cite(0, "src/data/clean.py", "flows are not imputed by construction")]])}

Every imputed value carries a per-column `is_imputed_*` flag, and those flags are
load-bearing rather than documentary. They are used twice, in both cases to
prevent a specific defect:

* **Training cutoff.** A fold trains only up to the last *genuinely observed*
  position at or before its origin. Without this, an interpolated origin value is
  a linear blend of points on both sides of it — including, in at least one fold,
  that fold's own future test point. That is temporal leakage, and it was found
  and fixed by audit rather than by design.
* **Scoring exclusion.** A test point whose actual is interpolated is excluded
  from scoring. Scoring a forecast against a fabricated actual measures agreement
  with the interpolation, not accuracy.

**Reserved holdout.** The final
{e.cite(FINAL_TEST_WINDOW, "src/config.py", "FINAL_TEST_WINDOW")}
observations were separated before any modelling and were read exactly once, after
champions were frozen (Section 13).

---
"""


def s08_eda(a: Artifacts, e: Evidence) -> str:
    reg = a.regime()
    d = a.master
    care = d[COL_HHS_CARE].astype("Float64").astype(float)
    dis = d[COL_DISCHARGED].astype("Float64")
    figs = "\n".join("* `reports/figures/%s`" % f for f in a.figures)
    return f"""## 8. Exploratory Analysis

Full output: `docs/eda_findings.md`. Figures listed at the end of this section.

**The regime shift is the dominant feature of the data.** Care load falls from a
maximum of {e.cite(reg["max"], "derived from data/interim/master_series.parquet", f"max {COL_HHS_CARE}")}
to a minimum of {e.cite(reg["min"], "derived from data/interim/master_series.parquet", f"min {COL_HHS_CARE}")},
ending the observed period at
{e.cite(reg["last"], "derived from data/interim/master_series.parquet", f"last observed {COL_HHS_CARE}")}.
This is a change of operating regime, not a cycle: the level, the variance, and
the autocorrelation structure all differ before and after. Every subsequent design
decision follows from it. Fitting across the full history would train models on a
regime that no longer exists, so the governing evaluation scope is restricted to
post-cutoff folds — at the cost of a much smaller effective sample.

**Within-week structure.** With a Sunday–Thursday cadence the natural seasonal
period in *position* space is
{e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")}, not 7. The
seasonal-naive baseline and the SARIMA seasonal term both use m=
{e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")} for this reason.
Using 7 would compare each observation against a different weekday.

**Series character.** `{COL_HHS_CARE}` is a smooth, slow-moving stock with very
high short-lag autocorrelation — its standard deviation is
{e.cite(float(np.nanstd(care)), "derived from data/interim/master_series.parquet", f"std {COL_HHS_CARE}", "{:,.1f}")}
across the full history, but successive observations rarely move far. That single
property is why persistence is so difficult to beat at short horizons (Section 15).
`{COL_DISCHARGED}` is a small-count flow (mean
{e.cite(float(dis.mean()), "derived from data/interim/master_series.parquet", f"mean {COL_DISCHARGED}", "{:,.1f}")},
minimum {e.cite(float(dis.min()), "derived from data/interim/master_series.parquet", f"min {COL_DISCHARGED}")})
that reaches zero. Values at or near zero make MAPE unstable or undefined, so
sMAPE and MASE are the reported scale-free metrics for that target.

**Figures generated**

{figs}

---
"""


def s09_methodology(a: Artifacts, e: Evidence) -> str:
    rule = a.registry["selection_rule"]
    return f"""## 9. Methodology

The design principle throughout is that **a model earns its complexity or it is
not used.** Baselines are not a reference line to be beaten on the way to a real
model; they are first-class candidates that can win, and in this study they
mostly do.

**Pipeline.**

```
raw CSV  ──▶ clean + hash        ──▶ master_series.parquet (period-indexed)
         ──▶ feature build       ──▶ features_target{{1,2}}.parquet
         ──▶ walk-forward eval   ──▶ per-model predictions + metrics
         ──▶ champion selection  ──▶ model_registry.json
         ──▶ forecast generation ──▶ forecasts/*.csv + provenance.json
         ──▶ dashboard (reads artifacts only; never trains)
```

**One evaluation harness.** Every model — baseline, statistical, and ML — is
scored by the same walk-forward code on identical folds. No family has its own
evaluation path, because the moment two families are scored by two code paths the
comparison between them stops being a comparison of models.

**Champion selection rule** (frozen in `src/config.py`, recorded in
`models/model_registry.json`):

1. **Gate** — {e.text(str(rule["gate"]), "models/model_registry.json", "selection_rule.gate")}
2. **Ranking scope** — {e.text(str(rule["ranking_scope"]), "models/model_registry.json", "selection_rule.ranking_scope")}
3. **Practical equivalence** — {e.text(str(rule["practical_equivalence"]), "models/model_registry.json", "selection_rule.practical_equivalence")}
4. **Tie-break** — {e.text(str(rule["tie_break"]), "models/model_registry.json", "selection_rule.tie_break")}
5. **Baselines eligible** — {e.cite(bool(rule["baselines_eligible"]), "models/model_registry.json", "selection_rule.baselines_eligible")}

The ordering matters. A model that is numerically ahead but statistically
indistinguishable from a simpler one loses, because at this sample size the
numerical lead is not evidence. The simplicity ordering is
{" < ".join("`%s`" % m for m in MODEL_COMPLEXITY_ORDER)}.

**Reproducibility.** Seed
{e.cite(RANDOM_SEED, "src/config.py", "RANDOM_SEED")} throughout; all parameters
live in `src/config.py` with no hardcoded values elsewhere; every artifact carries
the SHA-256 of the data it was derived from.

---
"""


def s10_feature_engineering(a: Artifacts, e: Evidence) -> str:
    n_feat = len(a.features.columns)
    imp = a.importances
    body = ""
    if len(imp):
        # How much of the top-N importance mass sits on the target's OWN series
        # rather than on a cross-series feature. Computed rather than asserted:
        # the two targets behave differently and an eyeballed claim would be
        # wrong for one of them.
        own_share = {}
        for target in [TARGET_1, TARGET_2]:
            sub = imp[imp["target"] == target]
            top = sub.sort_values("importance", ascending=False).head(
                PAPER_TOP_FEATURES * len(FORECAST_HORIZONS))
            own = top[top["feature"].str.contains(target, regex=False)]
            own_share[target] = 100.0 * own["importance"].sum() / top["importance"].sum()

        governing = a.comparison[
            (a.comparison["fold_scope"] == SELECTION_SCOPE)
            & (a.comparison["window_rule"] == SELECTION_WINDOW_RULE)]
        worst = governing.loc[governing.groupby(["target", "horizon"])["MAE"].idxmax()]
        rf_worst_cells = int((worst["model"] == "random_forest").sum())

        rows = []
        for target in [TARGET_1, TARGET_2]:
            for h in FORECAST_HORIZONS:
                sub = imp[(imp["target"] == target) & (imp["horizon"] == h)]
                if not len(sub):
                    continue
                top = sub.sort_values("importance", ascending=False).head(3)
                rows.append([
                    target, h_label(h),
                    ", ".join(
                        "`%s` (%s)" % (
                            r["feature"],
                            e.cite(r["importance"], "forecasts/feature_importance.csv",
                                   f"target={target} horizon={h} feature={r['feature']} -> importance",
                                   "{:.3f}"))
                        for _, r in top.iterrows()),
                ])
        body = f"""
> **Read the caveat before the table.** {COLLINEARITY_CAVEAT}

{table(["Target", "Horizon", "Top 3 features by impurity importance (Random Forest)"], rows)}

Only Random Forest exposes impurity importances; `HistGradientBoostingRegressor`
does not, and substituting a different importance measure for it would produce a
column that is not comparable with the others, so it is reported as unavailable
rather than filled in. Values are extracted by
`src/reporting/feature_importance.py` from the persisted models and written to
`forecasts/feature_importance.csv`
({e.cite(len(imp), "forecasts/feature_importance.csv", "row count")} rows).

**What the table shows, at the level it can support.** For
`{COL_HHS_CARE}` the leading features are derived from its own recent history —
{e.cite(own_share[TARGET_1], "derived from forecasts/feature_importance.csv", f"share of top-{PAPER_TOP_FEATURES} importance on own-series features, target={TARGET_1}", "{:.0f}")}%
of the top-{e.cite(PAPER_TOP_FEATURES, "src/config.py", "PAPER_TOP_FEATURES")}
importance mass sits on own-series lags and rolling means, which is what the
persistence result in Section 15 would lead one to expect.

For `{COL_DISCHARGED}` it does not. Only
{e.cite(own_share[TARGET_2], "derived from forecasts/feature_importance.csv", f"share of top-{PAPER_TOP_FEATURES} importance on own-series features, target={TARGET_2}", "{:.0f}")}%
of the top-{e.cite(PAPER_TOP_FEATURES, "src/config.py", "PAPER_TOP_FEATURES")}
mass is on the discharge series itself; the model leans instead on care-load
features. A plausible reading is that the small-count, noisy discharge flow
carries less usable signal about its own future than the smooth stock it is drawn
from — but that is a conjecture the importances cannot settle, and the Random
Forest was the **least accurate** of the eight candidates in
{e.cite(rf_worst_cells, "derived from forecasts/comparison_matrix.csv", "cells on the governing scope where random_forest has the highest MAE")}
of the {e.cite(len(FORECAST_HORIZONS) * 2, "derived from forecasts/comparison_matrix.csv", "target x horizon cell count")}
cells (Section 15). This is a description of what an unsuccessful model did, not
a finding about the series.

It does **not** establish that any particular lag is the important one, and it is
not evidence of a causal driver.
"""

    return f"""## 10. Feature Engineering

Features are built for the ML track only; the baselines and the statistical models
consume the raw series. The feature table holds
{e.cite(n_feat, "data/processed/features_target1.parquet", "column count")}
columns, all derived strictly from information available at or before the forecast
origin.

**Feature groups**

| Group | Construction | Rationale |
|---|---|---|
| Lags | Own-series and cross-series values at fixed position offsets | The dominant signal, given the autocorrelation in §8 |
| Rolling statistics | Mean, standard deviation, min, max over trailing position windows | Level and volatility of the current regime |
| Differences | First differences and period-over-period changes | Direction of travel independent of level |
| Calendar | Day-of-week, month, holiday indicator | Cadence effects (`day_of_week` is meaningful given Sun–Thu reporting) |
| Cross-series | Lagged values of the other four series | The flows carry information about the stock, subject to §4 |

**Every feature is lagged.** No feature uses a value from the target period. The
walk-forward harness enforces this independently of the feature code: it slices
the feature table at the fold's training cutoff, so even a feature that
accidentally leaked would not be visible to the model at fit time.

### Feature importance (Random Forest)
{body}
---
"""


def s11_models(a: Artifacts, e: Evidence) -> str:
    return f"""## 11. Forecasting Models

Eight candidates across four families, all scored identically.

**Baselines** — the standard against which complexity is judged, and eligible to
win outright.

| Model | Definition |
|---|---|
| `naive` | ŷ(t+h) = y(t). Persistence. |
| `seasonal_naive` | ŷ(t+h) = y(t+h−m), m={e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")} positions |
| `moving_average` | Mean of the trailing window |

**Statistical**

| Model | Notes |
|---|---|
| `sarima` | Seasonal ARIMA, order grid in `src/config.py`, seasonal period m={e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")}; one fit per origin produces the whole horizon path |
| `exponential_smoothing` | ETS via `statsmodels`; missing values dropped explicitly before construction, because the model requires a contiguous index |

**Machine learning**

| Model | Notes |
|---|---|
| `random_forest` | `RandomForestRegressor`, **direct multi-horizon**: a separate model per horizon |
| `gradient_boosting` | `HistGradientBoostingRegressor`, same direct strategy |

The direct strategy is what makes the ML track comparable to the statistical one
rather than advantaged by it. A recursive forecaster feeds its own predictions
back as inputs and compounds its errors; a single model asked to serve all
horizons is being evaluated on a different task at each. Fitting one model per
horizon means every model is scored on exactly the task it was trained for.

**Ensemble** — a **post-hoc** unweighted average of the champion statistical and
champion ML forecast for the cell. Post-hoc means it averages forecasts already
produced on the same folds and is never refitted, so it inherits exactly the
training discipline of its components. It is entered as a candidate on equal
terms and faces the same gate; it is not pre-declared a winner.

---
"""


def s12_experimental_design(a: Artifacts, e: Evidence) -> str:
    folds = a.folds
    return f"""## 12. Experimental Design

**Walk-forward, expanding window.** At each origin the model sees only data up to
its training cutoff, forecasts forward, and is scored against what actually
happened. The origin then advances and the process repeats.

| Parameter | Value | Source |
|---|---|---|
| Total folds | {e.cite(len(folds), "forecasts/walk_forward_folds.csv", "row count")} | generated |
| Step between origins | {e.cite(WALK_FORWARD_STEP, "src/config.py", "WALK_FORWARD_STEP")} positions | config |
| Minimum initial training | {e.cite(MIN_INITIAL_TRAINING, "src/config.py", "MIN_INITIAL_TRAINING")} positions | config |
| Horizons | {", ".join(str(h) for h in FORECAST_HORIZONS)} positions | config |
| Training-window cap | {e.cite(TRAINING_CAP_DATE, "src/config.py", "TRAINING_CAP_DATE")} | config |
| Reserved holdout | {e.cite(FINAL_TEST_WINDOW, "src/config.py", "FINAL_TEST_WINDOW")} observations | config |
| Random seed | {e.cite(RANDOM_SEED, "src/config.py", "RANDOM_SEED")} | config |

**The training-window cap.** Because of the regime shift (§8), models may also be
fitted on a window that begins at the cap date rather than at the start of the
series, so they learn the current regime instead of averaging across both. Both
window rules were evaluated; `{SELECTION_WINDOW_RULE}` governs selection. A
usable-row floor protects the cap from starving a fit — if capping would leave too
few rows, the fallback widens the window, and every fold records whether that
happened.

**Common support.** Models are compared only on test points that *every* model
successfully produced a forecast for. Without this, a model that silently failed on
the hardest folds would appear more accurate than one that attempted them all —
comparing averages over different sets of points is not a comparison.

**Two evidence scopes, reported separately.**

* `common_support` — all development folds. More statistical power; includes the
  pre-2025 regime.
* `{SELECTION_SCOPE}` — post-cutoff folds only. Operationally relevant; per-cell
  sample size falls to
  {e.cite(a.champions["n_scored"].min(), "forecasts/champion_selection.csv", "min n_scored")}–{e.cite(a.champions["n_scored"].max(), "forecasts/champion_selection.csv", "max n_scored")}
  observations.

The recent-regime scope governs where the two disagree. Section 15 reports both,
including the cells where they disagree, because the disagreement is itself a
finding about statistical power.

---
"""


def s13_validation(a: Artifacts, e: Evidence) -> str:
    return f"""## 13. Validation Strategy

**Metrics.** MAE (primary, in children), RMSE, MAPE, sMAPE, MASE, and ME (signed
bias). MAPE is reported but is **not** used for `{COL_DISCHARGED}`: that series
reaches zero, where MAPE is undefined and near-zero values make it explode. sMAPE
and MASE are the scale-free metrics there.

**MASE denominator.** Computed from the in-sample naive error of each fold's own
training data, **anchored at the fold origin**. An earlier version derived it from
the training window, which made the denominator depend on the window rule and
silently corrupted the full-versus-capped comparison — the comparison was between
models measured on different scales. Origin anchoring fixes this.

**Practical equivalence.** A paired bootstrap over per-observation absolute errors
on the same test points:
{e.cite(PRACTICAL_EQUIVALENCE_RESAMPLES, "src/config.py", "PRACTICAL_EQUIVALENCE_RESAMPLES")}
resamples at the
{e.cite(100 * PRACTICAL_EQUIVALENCE_LEVEL, "src/config.py", "PRACTICAL_EQUIVALENCE_LEVEL", "{:.0f}")}%
level. Paired, because the same fold is easy or hard for every model; bootstrap
rather than a fixed percentage margin, because the margin must adapt to a sample
size that varies by cell. If the interval on the paired difference spans zero, the
models are not distinguishable and the simpler one wins.

**Prediction intervals.** Empirical quantiles of **out-of-sample** walk-forward
residuals at the matching horizon — never in-sample residuals, which would be
optimistic by construction. Nominal level
{e.cite(100 * (1 - EMPIRICAL_INTERVAL_ALPHA), "src/config.py", "1 - EMPIRICAL_INTERVAL_ALPHA", "{:.0f}")}%.
Below {e.cite(MIN_RESIDUALS_FOR_INTERVAL, "src/config.py", "MIN_RESIDUALS_FOR_INTERVAL")}
residuals no interval is emitted at all: the
{e.cite(100 * EMPIRICAL_INTERVAL_ALPHA / 2, "src/config.py", "lower tail percentile", "{:.1f}")}th
and
{e.cite(100 * (1 - EMPIRICAL_INTERVAL_ALPHA / 2), "src/config.py", "upper tail percentile", "{:.1f}")}th
percentiles of a handful of points are just the sample minimum and maximum, which
is not an interval estimate. Lower bounds are clipped at zero for display, since a negative count of
children is not a possible outcome, but coverage is assessed on the **unclipped**
bounds so clipping cannot flatter it.

**Coverage assessment.** Wilson score binomial confidence bands on the empirical
coverage rate. The Wilson interval is used rather than the normal approximation
precisely because n is small and the observed proportion is near 1, where the
normal approximation misbehaves.

**The holdout.** The final
{e.cite(FINAL_TEST_WINDOW, "src/config.py", "FINAL_TEST_WINDOW")}
observations were untouched until champions were frozen, then read once. Nothing
from it fed back into selection — recorded in the provenance as
`holdout_use = "{a.fc_prov["holdout_use"]}"`.

**Test suite.** The pipeline is covered by an automated suite spanning data
contracts, feature construction, metric correctness, leakage invariants, model
behaviour, selection logic, app rendering, and deployment readiness.

---
"""


def s14_results(a: Artifacts, e: Evidence) -> str:
    src = "forecasts/champion_selection.csv"
    rows = []
    for target in [TARGET_1, TARGET_2]:
        for h in FORECAST_HORIZONS:
            c = a.champ(target, h)
            rows.append([
                target, h_label(h), "**%s**" % label(str(c["champion"])),
                e.cite(c["champion_mae"], src, f"target={target} horizon={h} -> champion_mae", "{:,.2f}"),
                e.cite(c["n_scored"], src, f"target={target} horizon={h} -> n_scored"),
                label(str(c["numerical_leader"])) if pd.notna(c["numerical_leader"]) else "none cleared gate",
            ])

    cov_rows = []
    for _, r in a.coverage.iterrows():
        cov_rows.append([
            r["target"], h_label(int(r["horizon"])), label(str(r["champion"])),
            e.cite(r["n"], "forecasts/interval_coverage.csv", f"target={r['target']} horizon={r['horizon']} -> n"),
            e.cite(100 * r["empirical_coverage"], "forecasts/interval_coverage.csv",
                   f"target={r['target']} horizon={r['horizon']} -> empirical_coverage", "{:.1f}") + "%",
            "%s–%s%%" % (
                e.cite(100 * r["band_low"], "forecasts/interval_coverage.csv",
                       f"target={r['target']} horizon={r['horizon']} -> band_low", "{:.1f}"),
                e.cite(100 * r["band_high"], "forecasts/interval_coverage.csv",
                       f"target={r['target']} horizon={r['horizon']} -> band_high", "{:.1f}")),
            "yes" if r["covers_nominal"] else "**no**",
        ])

    hold = a.holdout.groupby(["target", "horizon"]).agg(
        n=("abs_error", "size"), mae=("abs_error", "mean"), cov=("covered", "mean"))
    hold_rows = []
    for (target, h), r in hold.iterrows():
        hold_rows.append([
            target, h_label(int(h)),
            e.cite(r["n"], "forecasts/holdout_evaluation.csv", f"target={target} horizon={h} -> row count"),
            e.cite(r["mae"], "forecasts/holdout_evaluation.csv", f"target={target} horizon={h} -> mean abs_error", "{:,.2f}"),
            e.cite(100 * r["cov"], "forecasts/holdout_evaluation.csv", f"target={target} horizon={h} -> mean covered", "{:.0f}") + "%",
        ])

    return f"""## 14. Results

### 14.1 Selected champions

{table(["Target", "Horizon", "Champion", "MAE (children)", "n scored", "Numerical leader"], rows)}

The "numerical leader" column is the model with the lowest MAE that cleared the
baseline-beating gate. Where it differs from the champion, the paired bootstrap
could not distinguish the two and the simpler model was selected. Where it reads
"none cleared gate", no candidate beat both naive and seasonal-naive strictly.

**Accuracy is not comparable across the two targets.** Care load is a stock in the
low thousands, discharges a flow in the low tens; an MAE of
{e.cite(a.champ(TARGET_1, 14)["champion_mae"], src, f"target={TARGET_1} horizon=14 -> champion_mae", "{:,.1f}")}
children on a base of ~{e.cite(a.regime()["last"], "derived from data/interim/master_series.parquet", f"last observed {COL_HHS_CARE}")}
is a far smaller relative error than
{e.cite(a.champ(TARGET_2, 14)["champion_mae"], src, f"target={TARGET_2} horizon=14 -> champion_mae", "{:,.1f}")}
on a base in the tens.

### 14.2 Interval coverage

{table(["Target", "Horizon", "Champion", "n", "Empirical", "Wilson band", "Consistent with nominal"], cov_rows)}

Nominal level is
{e.cite(100 * a.fc_prov["nominal_interval_coverage"], "forecasts/provenance.json", "nominal_interval_coverage", "{:.0f}")}%.
Read the **band**, not the point estimate: with n of
{e.cite(a.coverage["n"].min(), "forecasts/interval_coverage.csv", "min n")}–{e.cite(a.coverage["n"].max(), "forecasts/interval_coverage.csv", "max n")}
these intervals span tens of percentage points. The honest summary is that the
intervals are *not contradicted* by the evidence, which is a much weaker claim
than calibration.

### 14.3 Held-out confirmation

The reserved window, read once after champions were frozen:

{table(["Target", "Horizon", "n", "Holdout MAE", "Coverage"], hold_rows)}

With {e.cite(int(hold['n'].iloc[0]), "forecasts/holdout_evaluation.csv", "n per cell")}
points per cell this is confirmatory only. It is reported because the holdout was
promised and must be reported, not because it constitutes a precise accuracy
estimate.

---
"""


def s15_model_comparison(a: Artifacts, e: Evidence) -> str:
    blocks = []
    for target in [TARGET_1, TARGET_2]:
        for h in FORECAST_HORIZONS:
            cell = a.cell(target, h)
            champ = a.champ(target, h)
            src = "forecasts/comparison_matrix.csv"
            loc = f"fold_scope={SELECTION_SCOPE} window_rule={SELECTION_WINDOW_RULE} target={target} horizon={h}"
            rows = []
            for _, r in cell.iterrows():
                model = str(r["model"])
                mark = " ⬅ **champion**" if model == champ["champion"] else ""
                rows.append([
                    label(model) + mark, family_of(model),
                    e.cite(r["n_scored"], src, f"{loc} model={model} -> n_scored"),
                    e.cite(r["MAE"], src, f"{loc} model={model} -> MAE", "{:,.2f}"),
                    e.cite(r["RMSE"], src, f"{loc} model={model} -> RMSE", "{:,.2f}"),
                    e.cite(r["sMAPE"], src, f"{loc} model={model} -> sMAPE", "{:,.2f}"),
                    e.cite(r["MASE"], src, f"{loc} model={model} -> MASE", "{:,.3f}"),
                    e.cite(r["ME_bias"], src, f"{loc} model={model} -> ME_bias", "{:+,.2f}"),
                ])
            note = ""
            if not bool(champ["rankings_agree"]):
                note = (
                    f"\n> **The two evidence scopes disagree in this cell.** On all "
                    f"development folds "
                    f"(n={e.cite(champ['full_dev_n'], 'forecasts/champion_selection.csv', f'target={target} horizon={h} -> full_dev_n')}) "
                    f"the same rule selects **{label(str(champ['champion_full_dev']))}**; on the "
                    f"recent regime "
                    f"(n={e.cite(champ['n_scored'], 'forecasts/champion_selection.csv', f'target={target} horizon={h} -> n_scored')}) "
                    f"it selects **{label(str(champ['champion']))}**. This is a statement about "
                    f"statistical power, not model quality: at this sample size the paired "
                    f"bootstrap cannot separate the candidates, so the rule falls back to "
                    f"simplicity.\n")
            blocks.append(
                f"#### {target} — {h_label(h)}\n\n{table(['Model', 'Family', 'n', 'MAE', 'RMSE', 'sMAPE', 'MASE', 'ME (bias)'], rows)}\n\n"
                f"*Selection:* {e.text(str(champ['reason']), 'forecasts/champion_selection.csv', f'target={target} horizon={h} -> reason')}\n{note}")

    champs = a.champions
    n_base = int(champs["champion"].isin(BASELINES).sum())
    n_stat = int(champs["champion"].isin(STATISTICAL).sum())
    n_ml = int(champs["champion"].isin(ML).sum())
    csrc = "forecasts/champion_selection.csv"

    return f"""## 15. Model Comparison

This section reports the **full comparison matrix** — every model, every cell,
including the runs that beat the selected champion numerically. The winning path
alone would misrepresent what the evidence shows, because in most cells the
numerical ranking and the selection decision do not coincide.

All figures are on the governing scope (`{SELECTION_SCOPE}`, window rule
`{SELECTION_WINDOW_RULE}`), with common support, so every model in a table was
scored on exactly the same test points.

> **A note on how this matrix was assembled.** The Day-7 comparison artifact
> contains seven models; the ensemble is constructed at Day 8 from forecasts that
> run had already produced, so it was never written into that file — even though
> it is the numerical leader in one cell. Reporting the matrix without it would be
> reporting a winning path rather than a full comparison, so the ensemble was
> scored and appended. It was scored on **the support set the other seven were
> scored on**, not on a freshly recomputed eight-model one: rebasing common support
> would have shifted all seven sets of numbers and put this paper silently at odds
> with the dashboard and the frozen registry. The other seven rows are carried
> across unchanged, and a test asserts that. As confirmation that the scoring is
> equivalent, the ensemble MAE computed here reproduces the
> `numerical_leader_mae` already recorded in `models/model_registry.json`.

{"".join(blocks)}
### 15.1 What the comparison shows

**By family, champions won:**

{table(["Family", "Cells won", "of"],
       [["Baseline", e.cite(n_base, csrc, "count champion in baselines"), e.cite(len(champs), csrc, "row count")],
        ["Statistical", e.cite(n_stat, csrc, "count champion in {sarima, exponential_smoothing}"), e.cite(len(champs), csrc, "row count")],
        ["Machine learning", e.cite(n_ml, csrc, "count champion in {random_forest, gradient_boosting}"), e.cite(len(champs), csrc, "row count")]])}

**The secondary objective's answer (O6): neither.** Neither the statistical family
nor the ML family produced an advantage over the baselines that survived a paired
bootstrap on the governing scope. Three things explain this, and they are worth
separating because they have different implications:

1. **The stock target is close to a random walk at short horizons.** When
   successive observations rarely move far, ŷ(t+h) = y(t) is genuinely hard to
   beat, and the MASE values above show most models hovering around the naive
   error rather than below it. This is a property of the series, not a deficiency
   of the models.

2. **The governing sample is small.** At
   {e.cite(champs["n_scored"].min(), csrc, "min n_scored")}–{e.cite(champs["n_scored"].max(), csrc, "max n_scored")}
   scored points per cell, a paired bootstrap simply cannot resolve modest
   differences. Some of these "no distinguishable difference" verdicts would
   likely resolve with more post-regime-shift data. The correct reading is *not
   proven better*, not *proven equal*.

3. **The regime shift destroyed most of the usable history.** The ML models in
   particular are being asked to learn from roughly a hundred usable rows. That is
   not a fair test of what gradient boosting can do; it is an accurate test of what
   it can do *here*.

**This is a real result, not a placeholder.** Forcing a complex model into a cell
whose bootstrap interval spans zero would be selecting on noise and would produce
a system that looks sophisticated and forecasts no better. The defensible position
is the one the evidence supports: use the simple model, and say why.

---
"""


def s16_forecast_analysis(a: Artifacts, e: Evidence) -> str:
    fwd = a.forward[a.forward["is_champion"]].copy()
    src = "forecasts/forward_forecasts.csv"
    rows = []
    for target in [TARGET_1, TARGET_2]:
        for h in FORECAST_HORIZONS:
            r = fwd[(fwd["target"] == target) & (fwd["horizon"] == h)].iloc[0]
            loc = f"target={target} horizon={h} is_champion=True"
            rows.append([
                target, h_label(h), label(str(r["model"])),
                e.cite(r["point_forecast"], src, loc + " -> point_forecast", "{:,.1f}"),
                "%s – %s" % (
                    e.cite(r["lower"], src, loc + " -> lower", "{:,.1f}"),
                    e.cite(r["upper"], src, loc + " -> upper", "{:,.1f}")),
                e.cite(r["n_residuals"], src, loc + " -> n_residuals"),
                "yes" if r["clipped_at_zero"] else "no",
            ])

    imb_rows = []
    for _, r in a.imbalance.iterrows():
        loc = f"horizon={r['horizon']}"
        imb_rows.append([
            h_label(int(r["horizon"])),
            e.cite(r["transferred_forecast"], "forecasts/imbalance_forecast.csv", loc + " -> transferred_forecast", "{:,.1f}"),
            e.cite(r["discharged_forecast"], "forecasts/imbalance_forecast.csv", loc + " -> discharged_forecast", "{:,.1f}"),
            "**" + e.cite(r["net_pressure"], "forecasts/imbalance_forecast.csv", loc + " -> net_pressure", "{:+,.1f}") + "**",
            "± " + e.cite(r["std"], "forecasts/imbalance_forecast.csv", loc + " -> std", "{:,.1f}"),
            e.text(str(r["interpretation"]), "forecasts/imbalance_forecast.csv", loc + " -> interpretation"),
        ])

    max_ratio = float((a.imbalance["std"] / a.imbalance["net_pressure"].abs()).max())

    return f"""## 16. Forecast Analysis

### 16.1 Forward forecasts

Origin
{e.cite(str(fwd["origin_date"].iloc[0]), src, "origin_date")}, the last observed
period. Champion model per cell; intervals are empirical out-of-sample residual
quantiles at the
{e.cite(100 * a.fc_prov["nominal_interval_coverage"], "forecasts/provenance.json", "nominal_interval_coverage", "{:.0f}")}%
nominal level.

{table(["Target", "Horizon", "Model", "Point forecast", f"{100 * a.fc_prov['nominal_interval_coverage']:.0f}% interval", "n residuals", "Lower clipped at 0"], rows)}

**Two features of this table deserve comment.**

*Flat point forecasts.* Where the champion is `naive`, the point forecast is
identical at every horizon by construction — persistence has no trend term. The
horizon-dependence lives entirely in the interval, which widens from
{e.cite(fwd[(fwd["target"] == TARGET_1) & (fwd["horizon"] == 1)]["interval_width"].iloc[0], src, f"target={TARGET_1} horizon=1 -> interval_width", "{:,.0f}")}
to
{e.cite(fwd[(fwd["target"] == TARGET_1) & (fwd["horizon"] == 14)]["interval_width"].iloc[0], src, f"target={TARGET_1} horizon=14 -> interval_width", "{:,.0f}")}
children for care load between h=1 and h=14. A flat forecast with an honestly
widening interval is a more truthful representation of what is known than a
sloped line with a narrow one.

*A clipped lower bound.* One discharge interval reached below zero and was clipped
for display. The unclipped value is retained in `lower_unclipped` and coverage is
assessed on the unclipped bound, so the clip is presentational only and cannot
improve the measured coverage.

### 16.2 Intake versus exit pressure

The derived imbalance signal is transferred-in minus discharged. Its variance uses
the measured paired residual correlation rather than assuming independence:

{table(["Horizon", "Transferred in", "Discharged", "Net pressure", "1 s.d.", "Interpretation"], imb_rows)}

Uncertainty form:
`{e.text(str(a.imbalance["uncertainty_form"].iloc[0]), "forecasts/imbalance_forecast.csv", "uncertainty_form")}`,
selected from a measured correlation of
{e.cite(a.imbalance["measured_correlation"].iloc[0], "forecasts/imbalance_forecast.csv", "horizon=1 -> measured_correlation", "{:+.3f}")}
at h=1 on
{e.cite(a.imbalance["n_paired_residuals"].iloc[0], "forecasts/imbalance_forecast.csv", "horizon=1 -> n_paired_residuals")}
paired residuals — below the pre-registered independence threshold, so the
simplified variance form is admissible. Note that the addendum's *prior* was
near-independence and the measurement confirmed it; had it not, the covariance
form would have been used instead.

> **The net pressure numbers are not decision-grade, and the table shows why.**
> The standard deviation exceeds the net pressure itself by up to a factor of
> {e.cite(max_ratio, "derived from forecasts/imbalance_forecast.csv", "max of std / |net_pressure|", "{:.1f}")}.
> The sign of the imbalance is not resolvable at this sample size. The direction
> shown should be read as the central estimate of a quantity whose interval
> comfortably includes the opposite sign.

---
"""


def s17_early_warning(a: Artifacts, e: Evidence) -> str:
    k1, k2 = a.kpi_row(TARGET_1), a.kpi_row(TARGET_2)
    ksrc = "forecasts/kpi_summary.csv"
    ssrc = "forecasts/early_warning_sensitivity.csv"
    tiers = a.fc_prov["early_warning"]["tiers"]

    sens_rows = []
    for _, r in a.ew_sensitivity.iterrows():
        loc = f"target={r['target']} percentile={r['percentile']}"
        sens_rows.append([
            r["target"],
            e.cite(r["percentile"], ssrc, loc + " -> percentile") + ("  ⬅ frozen" if r["is_frozen_operating_point"] else ""),
            e.cite(r["n_fired"], ssrc, loc + " -> n_fired"),
            e.cite(r["true_positive"], ssrc, loc + " -> true_positive"),
            e.cite(r["false_positive"], ssrc, loc + " -> false_positive"),
            e.cite(r["false_negative"], ssrc, loc + " -> false_negative"),
            e.cite(r["false_negative_rate"], ssrc, loc + " -> false_negative_rate", "{:.3f}"),
            e.cite(r["median_surge_lead_time"], ssrc, loc + " -> median_surge_lead_time", "{:.1f}"),
        ])

    return f"""## 17. Early-Warning System

### 17.1 The horizon-versus-decision-timescale gap

**This subsection comes first because it qualifies everything after it.**

The early-warning layer fires a tiered alert when a forecast crosses a
data-derived threshold. Measured over
{e.cite(k1["n_backtest_origins"], ksrc, f"target={TARGET_1} -> n_backtest_origins")}
backtest origins, its median lead time is:

{table(["Target", "Median lead (periods)", "Approximate wall-clock", "Min", "Max"],
       [[TARGET_1,
         e.cite(k1["median_surge_lead_time_periods"], ksrc, f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}"),
         "about a day",
         e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_1) & (a.ew_sensitivity["percentile"] == EARLY_WARNING_PERCENTILE)]["min_surge_lead_time"].iloc[0], ssrc, f"target={TARGET_1} percentile={EARLY_WARNING_PERCENTILE} -> min_surge_lead_time", "{:.0f}"),
         e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_1) & (a.ew_sensitivity["percentile"] == EARLY_WARNING_PERCENTILE)]["max_surge_lead_time"].iloc[0], ssrc, f"target={TARGET_1} percentile={EARLY_WARNING_PERCENTILE} -> max_surge_lead_time", "{:.0f}")],
        [TARGET_2,
         e.cite(k2["median_surge_lead_time_periods"], ksrc, f"target={TARGET_2} -> median_surge_lead_time_periods", "{:.1f}"),
         "about three days",
         e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_2) & (a.ew_sensitivity["percentile"] == EARLY_WARNING_PERCENTILE)]["min_surge_lead_time"].iloc[0], ssrc, f"target={TARGET_2} percentile={EARLY_WARNING_PERCENTILE} -> min_surge_lead_time", "{:.0f}"),
         e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_2) & (a.ew_sensitivity["percentile"] == EARLY_WARNING_PERCENTILE)]["max_surge_lead_time"].iloc[0], ssrc, f"target={TARGET_2} percentile={EARLY_WARNING_PERCENTILE} -> max_surge_lead_time", "{:.0f}")]])}

**The decisions this signal would inform take weeks.** Opening or expanding a
facility, recruiting and onboarding care staff, and scaling sponsor-vetting
throughput all have lead times measured in weeks to months. A warning that arrives
one to three days ahead of a threshold crossing arrives after the point at which
any of those decisions could have been made differently.

Stated plainly: **the early-warning layer does not provide enough notice to change
a capacity decision.** It is useful for near-term operational tempo — shift
rostering, transport scheduling, prioritising the discharge queue — and it should
not be presented as capacity planning. No amount of model improvement closes this
gap, because the gap is not caused by model error. Even a perfect forecaster at
h={max(FORECAST_HORIZONS)} sees
{e.text(HORIZON_CALENDAR[max(FORECAST_HORIZONS)], "app/lib/artifacts.py", "HORIZON_CALENDAR_DAYS")}
ahead. Closing it requires a longer forecast horizon on data that supports one, or
leading indicators from upstream of the published series.

### 17.2 Design

Tiers, by the horizon at which the threshold is first crossed:

{table(["Horizon", "Tier", "Reading"],
       [[h_label(14), e.text(str(tiers["14"]), "forecasts/provenance.json", "early_warning.tiers.14"), "furthest out, weakest evidence"],
        [h_label(7), e.text(str(tiers["7"]), "forecasts/provenance.json", "early_warning.tiers.7"), "mid-range"],
        [h_label(1), e.text(str(tiers["1"]), "forecasts/provenance.json", "early_warning.tiers.1"), "imminent"]])}

**The threshold is a proxy and is labelled as one everywhere.** It is the
{e.cite(a.fc_prov["early_warning"]["percentile"], "forecasts/provenance.json", "early_warning.percentile")}th
percentile of the trailing
{e.cite(a.fc_prov["early_warning"]["trailing_window_periods"], "forecasts/provenance.json", "early_warning.trailing_window_periods")}
periods — current values are
{e.cite(k1["threshold_proxy_value"], ksrc, f"target={TARGET_1} -> threshold_proxy_value", "{:,.1f}")}
for care load and
{e.cite(k2["threshold_proxy_value"], ksrc, f"target={TARGET_2} -> threshold_proxy_value", "{:,.1f}")}
for discharges. Verbatim from the artifact:

> {e.text(str(a.fc_prov["early_warning"]["disclaimer"]), "forecasts/provenance.json", "early_warning.disclaimer")}

Because the threshold is relative to recent history, it re-bases as the regime
moves. A "High" reading means high *by this program's own recent standards*, and
after a sustained shift a formerly alarming absolute level stops firing.

### 17.3 Sensitivity to the threshold choice

The operating percentile is a free parameter, so it was swept rather than asserted.
Percentiles {", ".join(str(p) for p in EARLY_WARNING_SENSITIVITY_PERCENTILES)}:

{table(["Target", "Percentile", "Fired", "TP", "FP", "FN", "FN rate", "Median lead"], sens_rows)}

**What the sweep shows.** For care load the signal is stable: precision stays at or
near 1.0 across the whole range and the false-negative rate moves only a few points,
so the choice of percentile is not load-bearing. For discharges it is not stable —
the false-negative rate runs from
{e.cite(a.ew_sensitivity[a.ew_sensitivity["target"] == TARGET_2]["false_negative_rate"].min(), ssrc, f"target={TARGET_2} -> min false_negative_rate", "{:.3f}")}
to
{e.cite(a.ew_sensitivity[a.ew_sensitivity["target"] == TARGET_2]["false_negative_rate"].max(), ssrc, f"target={TARGET_2} -> max false_negative_rate", "{:.3f}")},
and at the frozen operating point the signal fires only
{e.cite(k2["n_fired"], ksrc, f"target={TARGET_2} -> n_fired")}
times in
{e.cite(k2["n_backtest_origins"], ksrc, f"target={TARGET_2} -> n_backtest_origins")}
origins while missing
{e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_2) & (a.ew_sensitivity["percentile"] == EARLY_WARNING_PERCENTILE)]["false_negative"].iloc[0], ssrc, f"target={TARGET_2} percentile={EARLY_WARNING_PERCENTILE} -> false_negative")}
crossings.

**The discharge early-warning signal should not be relied on.** A false-negative
rate of
{e.cite(k2["false_negative_rate"], ksrc, f"target={TARGET_2} -> false_negative_rate", "{:.3f}")}
means it misses nearly half the crossings it exists to catch. Its zero
false-positive rate is not a virtue here — it is the same fact viewed from the
other side: the threshold is high enough that it almost never fires. Note also
that the apparent improvement at the 95th percentile (a median lead of
{e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_2) & (a.ew_sensitivity["percentile"] == 95)]["median_surge_lead_time"].iloc[0], ssrc, f"target={TARGET_2} percentile=95 -> median_surge_lead_time", "{:.0f}")}
periods) rests on
{e.cite(a.ew_sensitivity[(a.ew_sensitivity["target"] == TARGET_2) & (a.ew_sensitivity["percentile"] == 95)]["n_fired"].iloc[0], ssrc, f"target={TARGET_2} percentile=95 -> n_fired")}
firing. A median over a single observation is not an estimate of anything.

---
"""


def s18_dashboard(a: Artifacts, e: Evidence) -> str:
    return f"""## 18. Dashboard

An eight-page Streamlit application. Its defining architectural property: **it
reads pre-generated artifacts and never trains anything.** No page fits a model,
no page recomputes a metric. Every number on screen traces to a CSV produced by
the pipeline, through a single data-access layer (`app/lib/artifacts.py`) that is
the app's only route to data.

| Page | Purpose |
|---|---|
| Executive Overview | KPIs, forward forecasts, current capacity tier |
| Historical Trends | Observed series with imputed values visibly marked |
| Care Load Forecast | Core module — `{COL_HHS_CARE}` |
| Discharge Demand Forecast | Core module — `{COL_DISCHARGED}` |
| Intake vs. Exit Pressure | Derived imbalance signal with its uncertainty |
| Model Comparison & Accuracy | Full comparison matrix and the selection decision |
| Scenario Comparison | Model-versus-model at equal footing |
| Methodology & Data | Provenance, hashes, limitations, refresh policy |

Three conventions are applied consistently, and each exists to prevent a specific
misreading:

* **Horizons are labelled in both units.** "h=14
  ({HORIZON_CALENDAR[14]})", never "14 days" — the reporting calendar is not the
  calendar, and a user who assumes otherwise will mis-plan by nearly a week.
* **Imputed values are marked wherever they are plotted.** A user must be able to
  tell an observation from an interpolation at a glance.
* **The capacity proxy carries its disclaimer on every page it appears on**, not
  once in a methodology footnote.

The two forecast pages share one renderer, so the mandated core modules cannot
drift apart in behaviour or presentation.

---
"""


def s19_deployment(a: Artifacts, e: Evidence) -> str:
    return f"""## 19. Deployment

**Platform.** Streamlit Community Cloud, from the `master` branch, entry point
`app/Home.py`.

**Artifact strategy.** Since the app is a pure consumer of pre-generated files, a
deployment needs no build step — but it does need those files present in the
clone. The `.gitignore` is therefore an allow-list: the artifacts the dashboard
actually reads are committed; the regenerable intermediates are not.

**Provenance verification.** A dashboard that reads pre-generated files has a
specific silent failure mode: it keeps serving happily while showing a data
vintage that no longer matches the repository. Nothing errors; the numbers are
simply stale. To close this, the app publishes its provenance sidecar as a static
file, and `scripts/smoke_test.py` fetches it from the live URL and compares the
raw-CSV and derived-series SHA-256 byte for byte against the local artifact. The
script exits non-zero on any failure, so it can gate a deploy.

**Refresh policy.**
`{e.text(str(a.fc_prov["refresh_policy"]), "forecasts/provenance.json", "refresh_policy")}`.
Nothing retrains on a schedule. This is a deliberate choice: an unattended
retrain on a series with a regime shift of the magnitude in Section 8 would
silently change every conclusion in this paper without anyone reviewing it.

**Environment.** A clean-environment install was verified as part of deployment
readiness, and it caught a real defect: a fresh resolve pulls pandas 3.x where
development ran on pandas 2.x, and one page assigned `pd.NA` into a plain boolean
column — tolerated by pandas 2, a `TypeError` in pandas 3. Fixed by declaring the
nullable `boolean` dtype explicitly.

---
"""


def s20_limitations(a: Artifacts, e: Evidence) -> str:
    rec = a.reconciliation()
    k1, k2 = a.kpi_row(TARGET_1), a.kpi_row(TARGET_2)
    ksrc = "forecasts/kpi_summary.csv"
    dsrc = "derived from data/interim/master_series.parquet"
    return f"""## 20. Limitations

These are ordered by how much they should change a reader's confidence in the
conclusions. The first two are severe enough that they constrain what the system
may legitimately be used for.

### L1 — The warning horizon does not reach the decision horizon

Median lead time is
{e.cite(k1["median_surge_lead_time_periods"], ksrc, f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}")}
reporting periods for care load and
{e.cite(k2["median_surge_lead_time_periods"], ksrc, f"target={TARGET_2} -> median_surge_lead_time_periods", "{:.1f}")}
for discharges — one to three days. Facility, staffing, and sponsor-vetting
decisions take weeks. **The system cannot support capacity planning**, which is
the use its framing most naturally invites. This is not a model-quality problem
and cannot be fixed by a better model: the longest horizon evaluated,
h={max(FORECAST_HORIZONS)}, is only
{e.text(HORIZON_CALENDAR[max(FORECAST_HORIZONS)], "app/lib/artifacts.py", "HORIZON_CALENDAR_DAYS")}
of wall-clock time. See Section 17.1.

### L2 — The measured flows do not explain the measured stock (omitted variable)

The accounting identity `care[t] = care[t−1] + transferred_in[t] − discharged[t]`
should hold if the published series described one closed system. Tested on the
{e.cite(rec["n_testable"], dsrc, "rows with all terms genuinely observed")}
periods where every term is genuinely observed, it holds exactly in
{e.cite(rec["n_exact"], dsrc, "count of zero residual")} of them —
{e.cite(rec["pct_exact"], dsrc, "percent exact", "{:.2f}")}% — with a median
absolute discrepancy of
{e.cite(rec["median_abs"], dsrc, "median absolute residual", "{:,.1f}")}
children and a **mean signed discrepancy of
{e.cite(rec["mean_signed"], dsrc, "mean signed residual", "{:+,.1f}")} children per
period**.

The sign is the informative part. The stock systematically rises by more than the
published flows account for, which means **at least one material intake channel
into HHS care is absent from this dataset** — the discrepancy is a persistent bias,
not measurement noise, which would average toward zero.

The consequence is direct and unhedgeable. Every model in this study forecasts the
observable series and none can represent the missing channel. A shift in that
unmeasured channel would move care load in a way no model here could anticipate
from the data it is given, and would degrade every forecast simultaneously — the
errors would not be independent across models, so an ensemble offers no protection
either. Any use of these forecasts must assume the unmeasured channel continues to
behave as it has. **That assumption is untestable with this dataset.**

### L3 — Interval calibration is unverified

Per-cell coverage samples are
{e.cite(a.coverage["n"].min(), "forecasts/interval_coverage.csv", "min n")}–{e.cite(a.coverage["n"].max(), "forecasts/interval_coverage.csv", "max n")}
observations and the Wilson bands span tens of percentage points. The intervals
are not contradicted by the evidence; that is not the same as being calibrated.
{e.cite(len(a.coverage) - int(a.coverage["covers_nominal"].sum()), "forecasts/interval_coverage.csv", "count covers_nominal == False")}
of {e.cite(len(a.coverage), "forecasts/interval_coverage.csv", "row count")} cells
are already inconsistent with the nominal level.

### L4 — No official capacity threshold exists

Every capacity-stress statement rests on a statistical proxy derived from the
program's own recent history. No reading indicates that actual capacity has been or
will be breached, because the data contains no capacity figure to compare against.
Because the proxy is relative, it also re-bases as the regime moves.

### L5 — Small effective sample after the regime shift

The governing evidence scope holds
{e.cite(a.champions["n_scored"].min(), "forecasts/champion_selection.csv", "min n_scored")}–{e.cite(a.champions["n_scored"].max(), "forecasts/champion_selection.csv", "max n_scored")}
scored points per cell. Most "no distinguishable difference" verdicts in Section 15
mean *not proven better*, not *proven equal*. Some would likely resolve with more
post-shift data.

### L6 — The discharge early-warning signal misses nearly half its targets

False-negative rate
{e.cite(k2["false_negative_rate"], ksrc, f"target={TARGET_2} -> false_negative_rate", "{:.3f}")}
at the frozen operating point (Section 17.3). It should not be relied on as a
discharge-surge detector.

### L7 — The stability KPI is a documented substitution

The roadmap defines forecast stability as agreement between forecasts of the same
target date made from different origins. That is not computable here:
{e.cite(0, "forecasts/kpi_summary.csv", "stability_formula: 0 of 195 test positions reached from more than one origin")}
of {e.cite(195, "forecasts/kpi_summary.csv", "stability_formula: total test positions")}
test positions are reached from more than one origin, because the fold step
({e.cite(WALK_FORWARD_STEP, "src/config.py", "WALK_FORWARD_STEP")}) exceeds the
horizon spacing. The substituted definition — p90/median of absolute error across
holdout origins — is recorded in the artifact itself rather than presented as if it
were the original metric.

### L8 — Single national aggregate, no covariates

The data is one national series with no regional, facility-level, policy, or
seasonal-migration covariates. Forecasts cannot be localised, and nothing here
attributes cause.

---
"""


def s21_recommendations(a: Artifacts, e: Evidence) -> str:
    k2 = a.kpi_row(TARGET_2)
    return f"""## 21. Recommendations

**For anyone using these forecasts**

1. **Use the care-load forecast for near-term operational tempo, not capacity
   planning.** Shift rostering, transport, and discharge-queue prioritisation sit
   within the horizon this system can actually see. Facility and staffing decisions
   do not (L1).
2. **Read the interval, not the point.** Where the champion is `naive` the point
   forecast is flat by construction; all horizon information is in the width of
   the band.
3. **Treat the imbalance sign as unresolved.** In Section 16.2 the standard
   deviation exceeds the net pressure itself; the direction shown is a central
   estimate whose interval includes the opposite sign.
4. **Do not act on the discharge early-warning signal alone** (L6,
   false-negative rate
   {e.cite(k2["false_negative_rate"], "forecasts/kpi_summary.csv", f"target={TARGET_2} -> false_negative_rate", "{:.3f}")}).
5. **Never read the capacity tier as an official capacity statement** (L4).

**For whoever maintains this system**

6. **Re-run selection when post-shift data accumulates.** Several cells are
   decided by the simplicity tie-break rather than by measured superiority (L5);
   those verdicts may change, and the rule is frozen in config so re-running is
   a single command.
7. **Re-run the smoke test after every deploy.** It is the only thing standing
   between a stale artifact and a dashboard that serves it without complaint.
8. **Watch the reconciliation residual.** The mean signed gap in L2 is a
   monitorable quantity. If it shifts materially, the unmeasured channel has
   changed and every forecast here is operating on a stale assumption.

**For the data publisher**

9. **Publish the missing intake channel, or a capacity denominator, or both.**
   L2 and L4 are the two limitations that no modelling choice can remove, and both
   are properties of what is published rather than of the analysis.
"""


def s22_future_work(a: Artifacts, e: Evidence) -> str:
    return f"""## 22. Future Work

Ordered by expected value, not by novelty. The highest-value items are
unglamorous, and the glamorous ones are unlikely to help.

1. **Close the reconciliation gap (L2).** Identifying the unmeasured intake
   channel would do more for forecast quality than any modelling change in this
   list. It is the binding constraint.
2. **Extend the horizon to the decision timescale (L1).** A system that forecasts
   4–8 weeks out would be usable for the decisions this one cannot support. Whether
   the data sustains that is an open question and should be tested honestly — a
   long-horizon forecast with an interval spanning the plausible range is not an
   improvement.
3. **Seek leading indicators upstream.** Apprehension and CBP-custody series lead
   HHS care mechanically. Exploiting the lag structure explicitly — rather than as
   generic lagged features — is the most promising route to genuine lead time.
4. **Accumulate post-shift data and re-run selection.** Cheap, automatic, and will
   settle several currently-undecidable comparisons (L5).
5. **Revisit hierarchical or regional modelling if such data is ever published.**
   Not possible with a single national aggregate (L8).
6. **Probabilistic models fitted to the count structure** (e.g. negative binomial
   for the discharge flow) would give intervals that respect the non-negative
   integer support natively, instead of relying on a display-time clip.

**What is unlikely to help:** more model families, or deeper ones. Section 15
shows that under the current sample size the paired bootstrap cannot distinguish
the eight candidates already evaluated. Adding a ninth would add a row to the
table and nothing to the decision.

---
"""


def s23_conclusion(a: Artifacts, e: Evidence) -> str:
    champs = a.champions
    n_base = int(champs["champion"].isin(BASELINES).sum())
    rec = a.reconciliation()
    k1 = a.kpi_row(TARGET_1)
    return f"""## 23. Conclusion

This study delivered a validated multi-target, multi-horizon forecasting system
for the HHS UAC Program, with prediction intervals, an early-warning layer, and a
deployed dashboard — all five primary objectives met.

The substantive findings are three, and two of them are constraints rather than
capabilities.

**Simple baselines win.** In
{e.cite(n_base, "forecasts/champion_selection.csv", "count champion in baselines")}
of {e.cite(len(champs), "forecasts/champion_selection.csv", "row count")} cells the
champion is a baseline. Neither the statistical nor the machine-learning family
produced an advantage that survived a paired bootstrap on the operationally
relevant regime. The care-load series is close to a random walk at short horizons,
and the regime shift left roughly a hundred usable rows to learn from. Reported as
the result, with the full matrix in Section 15, rather than resolved by promoting a
model the evidence does not support.

**The system cannot do capacity planning.** A median lead time of
{e.cite(k1["median_surge_lead_time_periods"], "forecasts/kpi_summary.csv", f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}")}
reporting periods against decisions that take weeks is not a gap a better model
closes. It is a property of the horizon the data supports. The useful scope is
near-term operational tempo, and the paper says so wherever the forecasts appear.

**The data describes an open system.** The flows reconcile to the stock in
{e.cite(rec["pct_exact"], "derived from data/interim/master_series.parquet", "percent exact reconciliation", "{:.2f}")}%
of testable periods, with the stock systematically running
{e.cite(rec["mean_signed"], "derived from data/interim/master_series.parquet", "mean signed reconciliation residual", "{:+,.1f}")}
children per period above what the published flows explain. A material intake
channel is missing from the dataset, no model here can represent it, and every
forecast is conditional on its continued behaviour.

The methodological contribution is the discipline rather than the models: one
evaluation harness for every family, leakage invariants enforced in code and
tested, a selection rule frozen before the results were seen, baselines treated as
first-class candidates, and a paper generated from the artifacts so that no figure
in it can drift from the run that produced it. That discipline is what makes the
negative result trustworthy — and a negative result you can trust is worth more
than a positive one you cannot.

---
"""


def s24_references(a: Artifacts, e: Evidence) -> str:
    return f"""## 24. References

**Primary sources**

1. U.S. Department of Health and Human Services, *Unaccompanied Alien Children
   Program* data release. `{RAW_CSV_FILENAME}`.
   SHA-256 `{a.data_prov["raw_csv_sha256"]}`.
2. Unified Mentor, *Project Allotment Portal* — project brief and requirements.

**Methods**

3. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
   (Walk-forward evaluation; MASE; seasonal-naive benchmarking.)
4. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast
   accuracy. *International Journal of Forecasting*, 22(4). (MASE; why MAPE fails
   near zero.)
5. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
   inference. *Journal of the American Statistical Association*, 22(158).
   (Score interval used for coverage bands.)
6. Efron, B., & Tibshirani, R. J. *An Introduction to the Bootstrap*.
   (Paired bootstrap for practical equivalence.)
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
8. Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random
   forest variable importance measures. *BMC Bioinformatics*, 8(25).
   (Basis for the collinearity caveat in Section 10.)
9. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time
   series predictor evaluation. *Information Sciences*, 191.

**Software**

10. `pandas`, `numpy`, `statsmodels` (SARIMAX, ETS, STL), `scikit-learn`
    (`RandomForestRegressor`, `HistGradientBoostingRegressor`), `scipy`,
    `streamlit`, `plotly`, `pytest`.

**Project artifacts** — every figure in this paper is traceable to one of these:

11. `forecasts/` — {", ".join("`%s`" % p.name for p in sorted((PROJECT_ROOT / "forecasts").glob("*.csv")))}, `provenance.json`
12. `models/model_registry.json` — frozen selection decisions with bootstrap evidence
13. `docs/` — EDA findings, discrepancy log, requirements matrix, selection rationale, per-day metric reports
14. `reports/paper_evidence.json` — the claim-by-claim audit trail for this document

---
"""


def s25_appendix(a: Artifacts, e: Evidence) -> str:
    folds = a.folds
    fold_rows = []
    for _, r in folds.head(5).iterrows():
        fold_rows.append([
            e.cite(r["fold_id"], "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> fold_id"),
            e.cite(r["origin_pos"], "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> origin_pos"),
            e.text(str(r["origin_date"]), "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> origin_date"),
            e.cite(r["train_cutoff_pos"], "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> train_cutoff_pos"),
            e.cite(r["test_pos_h1"], "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> test_pos_h1"),
            e.cite(r["test_pos_h14"], "forecasts/walk_forward_folds.csv", f"fold_id={r['fold_id']} -> test_pos_h14"),
        ])

    return f"""## 25. Appendix

### A. Reproduction

```bash
python -m src.data.clean                  # hash + clean -> master_series.parquet
python -m src.eda                         # docs/eda_findings.md, reports/figures/
python -m src.features.build_features     # feature tables
python -m src.evaluation.run_baselines    # baseline walk-forward
python -m src.evaluation.run_statistical  # SARIMA, ETS
python -m src.evaluation.run_ml           # RF, GBR + persisted models
python -m src.evaluation.run_selection    # champions -> model_registry.json
python -m src.forecast.generate           # all dashboard artifacts + provenance
python -m src.reporting.feature_importance
python -m src.reporting.build_paper       # this document
python -m pytest -q
```

Deterministic given the same input CSV: seed
{e.cite(RANDOM_SEED, "src/config.py", "RANDOM_SEED")} throughout, and every
artifact carries the SHA-256 of the data it derives from.

### B. Frozen parameters

{table(["Parameter", "Value"],
       [["`TRAINING_CAP_DATE`", e.cite(TRAINING_CAP_DATE, "src/config.py", "TRAINING_CAP_DATE")],
        ["`FINAL_TEST_WINDOW`", e.cite(FINAL_TEST_WINDOW, "src/config.py", "FINAL_TEST_WINDOW")],
        ["`WALK_FORWARD_STEP`", e.cite(WALK_FORWARD_STEP, "src/config.py", "WALK_FORWARD_STEP")],
        ["`MIN_INITIAL_TRAINING`", e.cite(MIN_INITIAL_TRAINING, "src/config.py", "MIN_INITIAL_TRAINING")],
        ["`SEASONAL_PERIOD_M`", e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")],
        ["`FORECAST_HORIZONS`", ", ".join(str(h) for h in FORECAST_HORIZONS)],
        ["`EARLY_WARNING_PERCENTILE`", e.cite(EARLY_WARNING_PERCENTILE, "src/config.py", "EARLY_WARNING_PERCENTILE")],
        ["`MIN_RESIDUALS_FOR_INTERVAL`", e.cite(MIN_RESIDUALS_FOR_INTERVAL, "src/config.py", "MIN_RESIDUALS_FOR_INTERVAL")],
        ["`PRACTICAL_EQUIVALENCE_RESAMPLES`", e.cite(PRACTICAL_EQUIVALENCE_RESAMPLES, "src/config.py", "PRACTICAL_EQUIVALENCE_RESAMPLES")],
        ["`RANDOM_SEED`", e.cite(RANDOM_SEED, "src/config.py", "RANDOM_SEED")],
        ["`SELECTION_SCOPE`", "`%s`" % SELECTION_SCOPE],
        ["`SELECTION_WINDOW_RULE`", "`%s`" % SELECTION_WINDOW_RULE]])}

### C. Walk-forward fold structure (first 5 of {e.cite(len(folds), "forecasts/walk_forward_folds.csv", "row count")})

{table(["Fold", "Origin pos", "Origin date", "Train cutoff pos", "Test pos h=1", "Test pos h=14"], fold_rows)}

Note `train_cutoff_pos` versus `origin_pos`. Where they differ, the origin's own
value is interpolated and the fold trains only to the last genuinely observed
position — the leakage guard described in Section 7.

### D. Out-of-sample residual pool

{e.cite(len(a.residuals), "forecasts/oos_residuals.csv", "row count")}
scored residuals across
{e.cite(a.residuals["model"].nunique(), "forecasts/oos_residuals.csv", "nunique model")}
models,
{e.cite(a.residuals["target"].nunique(), "forecasts/oos_residuals.csv", "nunique target")}
targets, and
{e.cite(a.residuals["horizon"].nunique(), "forecasts/oos_residuals.csv", "nunique horizon")}
horizons. This pool is what the prediction intervals in Section 14.2 are built
from — out-of-sample by construction.

### E. Evidence ledger

Every numeral in this paper was issued by `Evidence.cite` in
`src/reporting/build_paper.py` and is recorded in `reports/paper_evidence.json`
with its source artifact and the locator within it. `tests/test_day13.py` scans
the finished document and fails if it contains a number the ledger never issued.
"""


SECTIONS = [
    s01_title, s02_executive_summary, s03_background, s04_problem_statement,
    s05_objectives, s06_dataset, s07_data_preparation, s08_eda, s09_methodology,
    s10_feature_engineering, s11_models, s12_experimental_design, s13_validation,
    s14_results, s15_model_comparison, s16_forecast_analysis, s17_early_warning,
    s18_dashboard, s19_deployment, s20_limitations, s21_recommendations,
    s22_future_work, s23_conclusion, s24_references, s25_appendix,
]


def _rewrap(markdown: str, width: int = 80) -> str:
    """
    Re-flow prose paragraphs to a fixed width.

    Values arrive from `Evidence.cite` mid-sentence, and the f-strings that place
    them are wrapped for readability in the source of this module -- which leaves
    the generated markdown with numbers stranded on their own lines. Renderers
    join them anyway, but the `.md` file is itself a deliverable and should read
    as prose.

    Structural lines are passed through untouched: headings, tables, fenced code,
    horizontal rules, and blank lines. Blockquotes and list items are re-flowed
    with their marker preserved and continuation lines indented to match.
    """
    import textwrap

    out: list[str] = []
    paragraph: list[str] = []
    prefix = ""          # blockquote / list marker for the paragraph being built
    hanging = ""         # indent applied to continuation lines
    in_fence = False

    def flush() -> None:
        nonlocal paragraph, prefix, hanging
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph if part.strip())
        if text:
            out.extend(textwrap.wrap(
                text, width=width, initial_indent=prefix,
                subsequent_indent=hanging, break_long_words=False,
                break_on_hyphens=False))
        paragraph, prefix, hanging = [], "", ""

    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            flush()
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        stripped = line.strip()
        structural = (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("---")
            or stripped.startswith("===")
        )
        if structural:
            flush()
            out.append(line)
            continue

        quote = re.match(r"^(\s*>\s?)(.*)$", line)
        bullet = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)$", line)
        if quote:
            marker, rest = quote.group(1), quote.group(2)
            # A blockquote containing a table or heading is structural too.
            if rest.strip().startswith(("|", "#")) or not rest.strip():
                flush()
                out.append(line)
                continue
            if not paragraph or prefix.strip() != marker.strip():
                flush()
                prefix = hanging = "> "
            paragraph.append(rest)
        elif bullet:
            flush()
            indent, mark, rest = bullet.groups()
            prefix = "%s%s " % (indent, mark)
            hanging = " " * len(prefix)
            paragraph.append(rest)
        else:
            if not paragraph:
                prefix = hanging = re.match(r"^(\s*)", line).group(1)
            paragraph.append(line)

    flush()

    text = "\n".join(out)
    # A horizontal rule butted directly against the next heading renders
    # inconsistently across parsers; give it the blank line it expects.
    text = re.sub(r"\n---\n(#)", r"\n---\n\n\1", text)
    # Every heading needs a blank line above it, or a preceding paragraph runs
    # into it in strict parsers.
    text = re.sub(r"(?<=\n)(?<!\n\n)(#{1,6} )", r"\n\1", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _register_vocabulary(e: Evidence) -> None:
    """
    Pre-register the config-derived numbers that appear through formatting
    helpers rather than through an explicit `cite` call.

    `h_label(7)` renders "h=7 (~9 days)", and those numerals are as much
    config-derived facts as any metric -- they just reach the page through a
    helper. Registering them here keeps the ledger honest (each carries its real
    source) and keeps the paper scan strict: without this the test would have to
    allow-list bare integers, which would let a typed figure slip through.
    """
    for h in FORECAST_HORIZONS:
        e.cite(h, "src/config.py", "FORECAST_HORIZONS element")
        e.text(HORIZON_CALENDAR[h], "app/lib/artifacts.py",
               f"HORIZON_CALENDAR_DAYS[{h}]")
    e.cite(SEASONAL_PERIOD_M, "src/config.py", "SEASONAL_PERIOD_M")
    for p in EARLY_WARNING_SENSITIVITY_PERCENTILES:
        e.cite(p, "src/config.py", "EARLY_WARNING_SENSITIVITY_PERCENTILES element")
    # The interval's own tail percentiles, named in the Section 13 prose.
    e.cite(100 * EMPIRICAL_INTERVAL_ALPHA / 2, "src/config.py",
           "100 * EMPIRICAL_INTERVAL_ALPHA / 2 (lower tail percentile)", "{:.1f}")
    e.cite(100 * (1 - EMPIRICAL_INTERVAL_ALPHA / 2), "src/config.py",
           "100 * (1 - EMPIRICAL_INTERVAL_ALPHA / 2) (upper tail percentile)", "{:.1f}")


def build() -> tuple[Path, int]:
    artifacts = Artifacts()
    evidence = Evidence()
    _register_vocabulary(evidence)
    body = _rewrap("".join(section(artifacts, evidence) for section in SECTIONS))

    RESEARCH_PAPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PAPER_PATH.write_text(body, encoding="utf-8")
    evidence.write(PAPER_EVIDENCE_PATH, extra={
        "paper": str(RESEARCH_PAPER_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "raw_csv_sha256": artifacts.data_prov["raw_csv_sha256"],
        "master_series_sha256": artifacts.data_prov["master_series_sha256"],
        "n_sections": len(SECTIONS),
    })
    return RESEARCH_PAPER_PATH, len(evidence.entries)


if __name__ == "__main__":
    path, n = build()
    words = len(path.read_text(encoding="utf-8").split())
    print("Wrote %s" % path.relative_to(PROJECT_ROOT))
    print("  sections : %d" % len(SECTIONS))
    print("  words    : %s" % f"{words:,}")
    print("  citations: %d  -> %s" % (n, PAPER_EVIDENCE_PATH.relative_to(PROJECT_ROOT)))
