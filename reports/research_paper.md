# Predictive Forecasting of Care Load and Placement Demand in the HHS Unaccompanied Alien Children Program

**A walk-forward evaluation of baseline, statistical, and machine-learning
forecasters, with an early-warning capacity-stress layer**

| | |
|---|---|
| Data source | U.S. Department of Health and Human Services, Unaccompanied Alien Children Program (`HHS_Unaccompanied_Alien_Children_Program (1).csv`) |
| Observation period | 2023-01-12 to 2025-12-21 |
| Raw CSV SHA-256 | `061af0a97a1b3bda7a36f0ce8df08b6994847b0a4c402828cf2ef835d4947198` |
| Derived series SHA-256 | `3c6b093d73744140ac7d1d68d308e9f3f0f070fdc3d26e8c6e3d03bf24243c6d` |
| Artifacts generated | 2026-09-10T15:17:09+00:00 |
| Prepared for | Unified Mentor — Project Allotment Portal |

> **Every figure in this document was read from a generated artifact.** The
> paper is produced by `src/reporting/build_paper.py`, which fetches each value
> from a named output file and records the lookup in
> `reports/paper_evidence.json`. No number here was typed from memory, and a
> test fails the build if one is.

---

## 2. Executive Summary

This study built and validated a multi-target, multi-horizon forecasting system
for the HHS Unaccompanied Alien Children (UAC) Program, covering care load
(`Children in HHS Care`) and placement demand (`Children discharged from HHS
Care`) at 3 horizons, with an early-warning layer for capacity stress.

**The principal finding is negative, and it is the result rather than a
shortfall.** Across 6 target/horizon cells, a simple baseline is the selected
champion in 6 of them. Neither the statistical models (SARIMA, ETS) nor the
machine-learning models (Random Forest, Gradient Boosting) produced an accuracy
advantage that survived a paired bootstrap on the operationally relevant recent
regime. That conclusion is reported with the full comparison matrix in Section
15, including the cells where a complex model was numerically ahead, because the
numerical ranking and the selection decision genuinely differ and hiding the
difference would misrepresent the evidence.

**Three results matter more than the accuracy numbers:**

1. **The warning horizon is far shorter than the decision horizon.** The
   early-warning layer's median lead time is 1.0 reporting periods for care load
   and 2.5 for discharge demand — roughly one to three days of notice. Bed
   capacity, staffing, and placement logistics move on weeks. **This system
   cannot support the decisions its framing invites.** Section 17 states the gap
   in full; it is the single most important qualification on everything else
   here.

2. **The measured flows do not explain the measured stock.** Of 682 testable
   periods, the identity `care[t] = care[t-1] + transferred_in[t] -
   discharged[t]` holds exactly in 18 (2.64%). The mean signed residual is
   **+42.0 children per period** — the stock rises by more than the published
   flows account for. At least one material intake or exit channel is absent
   from the dataset. Every model here forecasts the observable series and none
   can represent that missing channel (Section 20, L2).

3. **Interval calibration is unverified, not verified.** The empirical
   prediction intervals carry a nominal 95% level, but per-cell sample sizes are
   12–15 observations. The Wilson bands are correspondingly wide and 4 of 6
   cells are consistent with the nominal level. This is weak evidence of
   adequacy, not evidence of calibration.

**Headline KPIs** (definitions and their caveats in Section 14):

| KPI | Children in HHS Care | Children discharged from HHS Care |
|---|---|---|
| Forecast accuracy (100 − sMAPE, h=1) | 99.56% | 63.37% |
| Capacity tier (forward h=1, proxy) | High | Normal |
| Median surge lead time (periods) | 1.0 | 2.5 |
| False-positive rate | 0.000 | 0.000 |
| False-negative rate | 0.115 | 0.476 |
| Forecast stability index (substituted) | 2.05 | 1.34 |

The capacity tier is a **relative, data-derived proxy**. No official capacity
threshold exists anywhere in the source documentation or the dataset, so no
statement here should be read as indicating that capacity has been or will be
breached.

---

## 3. Background

The UAC Program is the mechanism by which the U.S. Department of Health and
Human Services assumes custody of unaccompanied minors encountered at the
border. The operational chain the published data describes has four stages:
children are apprehended into CBP custody, held there, transferred out of CBP
custody into HHS care, and eventually discharged from HHS care — normally to a
vetted sponsor.

Two quantities drive resource planning. **Care load** — the number of children
in HHS care on a given day — determines bed, staffing, and facility
requirements. **Placement demand** — discharges per period — determines
caseworker and sponsor vetting throughput. The two are coupled: care load is a
stock that accumulates the difference between intake and discharge, so a
sustained imbalance compounds rather than averaging out.

Published reporting on the program is a daily operational feed rather than a
research dataset, and it carries the characteristics that implies: a reporting
cadence tied to working days, a mixture of stock and flow measures in one table,
and no published capacity denominator. Section 6 documents what that means for
this analysis; Section 20 documents what it means for the conclusions.

---

## 4. Problem Statement

**The forecasting problem.** Given the published daily series through
2025-12-21, produce forward forecasts of care load and discharge demand at h=1
(~1 day), h=7 (~9 days), h=14 (~20 days) reporting periods ahead, each with a
calibrated uncertainty interval, and a signal that flags periods of unusually
high load early enough to be acted on.

**What makes it hard here, specifically:**

1. **A regime shift dominates the series.** Care load ranges from 11,516 down to
   1,972 — a factor of 5.8. A model fitted across the whole history learns a
   level and a variance that no longer exist. This is the reason the evaluation
   is scoped to the recent regime (Section 12) and it is why the effective
   sample size for every decision in this paper is small.

2. **The system is not closed.** The identity linking the flows to the stock
   fails in 97.36% of testable periods, with a mean signed gap of +42.0 children
   per period. Forecasting the stock from the flows is therefore not available
   as a strategy, and no model here can capture the unmeasured channel (Section
   20, L2).

3. **The reporting calendar is not the calendar.** Observations arrive Sunday
   through Thursday, so consecutive rows are not consecutive days and a "14-day"
   horizon is not 14 days. All indexing in this project is by **period
   position**, never by date arithmetic (Section 7).

4. **There is no capacity denominator.** "Capacity stress" cannot be defined
   against an official threshold because none is published. Any threshold used
   here is a statistical property of the program's own recent history, and is
   labelled as such everywhere it appears.

---

## 5. Objectives

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

**Deliberate non-objectives.** The system does not attempt causal attribution,
does not forecast the unmeasured intake channel identified in Section 4, and
does not produce facility-level or region-level forecasts — the published data
is a single national aggregate.

---

## 6. Dataset

**Source.** `HHS_Unaccompanied_Alien_Children_Program (1).csv`, the official HHS
UAC Program data supplied with the project brief, used read-only and hashed on
every pipeline run (SHA-256 `061af0a97a1b3bda…`). The pipeline refuses to run if
the hash changes without the derived artifacts being regenerated, so a stale
result cannot be silently served (Section 19).

**Shape.** The raw file contains 1,170 rows, of which 720 carry data; the
remainder are blank trailing rows. After regularising to the reporting calendar
the series occupies 769 period positions from 2023-01-12 to 2025-12-21, of which
49 are gap slots with no published observation.

**Columns.**

| Column | Type | Observed | Min | Max | Mean |
|---|---|---|---|---|---|
| `Children apprehended and placed in CBP custody*` | flow | 720 | 0 | 333 | 93.5 |
| `Children in CBP custody` | stock | 769 | 7 | 531 | 169.5 |
| `Children transferred out of CBP custody` | flow | 720 | 0 | 440 | 128.7 |
| `Children in HHS Care` | stock | 769 | 1,972 | 11,516 | 6,063.6 |
| `Children discharged from HHS Care` | flow | 720 | 0 | 505 | 173.4 |

**Stock versus flow is the distinction that governs everything downstream.** A
stock (`Children in HHS Care`, `Children in CBP custody`) exists continuously
and is merely unobserved on a non-reporting day, so interpolating it estimates
something real. A flow (`Children discharged from HHS Care` and the others) is a
count of events *within* a period; on a day with no report there is no count to
estimate, and interpolating one would fabricate events. Flows are therefore left
genuinely missing. This rule is applied without exception and is tested.

---

## 7. Data Preparation

**Parsing.** `Children in HHS Care` arrives string-typed because its values
carry thousands-separator commas; it is parsed to a nullable integer rather than
coerced through float, so no value is silently rounded.

**Period-position indexing.** The reporting cadence is Sunday–Thursday. Rather
than reindexing onto a daily calendar and creating weekend rows that were never
meant to exist, the series is indexed by **period position**: position *i* is
the *i*-th reporting slot. Every lag, every horizon, every training window, and
every fold origin in this project is expressed in period positions.
Calendar-date arithmetic is never used to locate an observation. The practical
consequence for a reader: a horizon of 14 positions is about ~20 days of
wall-clock time, not two weeks.

**Missing-value treatment.** Applying the stock/flow rule from Section 6:

| Series | Treatment | Values imputed |
|---|---|---|
| `Children in HHS Care` | linear interpolation (stock) | 49 |
| `Children in CBP custody` | linear interpolation (stock) | 49 |
| flow columns (3) | left missing — never interpolated | 0 |

Every imputed value carries a per-column `is_imputed_*` flag, and those flags
are load-bearing rather than documentary. They are used twice, in both cases to
prevent a specific defect:

* **Training cutoff.** A fold trains only up to the last *genuinely observed*
  position at or before its origin. Without this, an interpolated origin value
  is a linear blend of points on both sides of it — including, in at least one
  fold, that fold's own future test point. That is temporal leakage, and it was
  found and fixed by audit rather than by design.
* **Scoring exclusion.** A test point whose actual is interpolated is excluded
  from scoring. Scoring a forecast against a fabricated actual measures
  agreement with the interpolation, not accuracy.

**Reserved holdout.** The final 60 observations were separated before any
modelling and were read exactly once, after champions were frozen (Section 13).

---

## 8. Exploratory Analysis

Full output: `docs/eda_findings.md`. Figures listed at the end of this section.

**The regime shift is the dominant feature of the data.** Care load falls from a
maximum of 11,516 to a minimum of 1,972, ending the observed period at 2,484.
This is a change of operating regime, not a cycle: the level, the variance, and
the autocorrelation structure all differ before and after. Every subsequent
design decision follows from it. Fitting across the full history would train
models on a regime that no longer exists, so the governing evaluation scope is
restricted to post-cutoff folds — at the cost of a much smaller effective
sample.

**Within-week structure.** With a Sunday–Thursday cadence the natural seasonal
period in *position* space is 5, not 7. The seasonal-naive baseline and the
SARIMA seasonal term both use m= 5 for this reason. Using 7 would compare each
observation against a different weekday.

**Series character.** `Children in HHS Care` is a smooth, slow-moving stock with
very high short-lag autocorrelation — its standard deviation is 2,829.0 across
the full history, but successive observations rarely move far. That single
property is why persistence is so difficult to beat at short horizons (Section
15). `Children discharged from HHS Care` is a small-count flow (mean 173.4,
minimum 0) that reaches zero. Values at or near zero make MAPE unstable or
undefined, so sMAPE and MASE are the reported scale-free metrics for that
target.

**Figures generated**

* `reports/figures/acf_pacf_discharged.png`
* `reports/figures/acf_pacf_hhs_care.png`
* `reports/figures/correlation_heatmaps.png`
* `reports/figures/regime_shift_overview.png`
* `reports/figures/stl_decomp_discharged.png`
* `reports/figures/stl_decomp_hhs_care.png`
* `reports/figures/structural_break_discharged.png`
* `reports/figures/structural_break_hhs_care.png`
* `reports/figures/weekly_seasonality_discharged.png`
* `reports/figures/weekly_seasonality_hhs_care.png`

---

## 9. Methodology

The design principle throughout is that **a model earns its complexity or it is
not used.** Baselines are not a reference line to be beaten on the way to a real
model; they are first-class candidates that can win, and in this study they
mostly do.

**Pipeline.**

```
raw CSV  ──▶ clean + hash        ──▶ master_series.parquet (period-indexed)
         ──▶ feature build       ──▶ features_target{1,2}.parquet
         ──▶ walk-forward eval   ──▶ per-model predictions + metrics
         ──▶ champion selection  ──▶ model_registry.json
         ──▶ forecast generation ──▶ forecasts/*.csv + provenance.json
         ──▶ dashboard (reads artifacts only; never trains)
```

**One evaluation harness.** Every model — baseline, statistical, and ML — is
scored by the same walk-forward code on identical folds. No family has its own
evaluation path, because the moment two families are scored by two code paths
the comparison between them stops being a comparison of models.

**Champion selection rule** (frozen in `src/config.py`, recorded in
`models/model_registry.json`):

1. **Gate** — must beat BOTH naive and seasonal-naive on MAE, strictly
2. **Ranking scope** — post_cutoff_common_support
3. **Practical equivalence** — paired bootstrap over per-observation absolute
   errors, 10000 resamples at the 95% level; a candidate whose interval spans
   zero is tied with the numerical leader
4. **Tie-break** — prefer un-biased, then prefer lower complexity rank
5. **Baselines eligible** — yes

The ordering matters. A model that is numerically ahead but statistically
indistinguishable from a simpler one loses, because at this sample size the
numerical lead is not evidence. The simplicity ordering is `naive` <
`seasonal_naive` < `moving_average` < `sarima` < `exponential_smoothing` <
`random_forest` < `gradient_boosting` < `ensemble`.

**Reproducibility.** Seed 42 throughout; all parameters live in `src/config.py`
with no hardcoded values elsewhere; every artifact carries the SHA-256 of the
data it was derived from.

---

## 10. Feature Engineering

Features are built for the ML track only; the baselines and the statistical
models consume the raw series. The feature table holds 68 columns, all derived
strictly from information available at or before the forecast origin.

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

> **Read the caveat before the table.** Impurity-based importance distributes
> credit arbitrarily among correlated predictors. The feature set is lags and
> rolling windows of the same five series, so these values indicate which SERIES
> the model used, not which specific lag matters most, and they are not evidence
> of a causal driver.

| Target | Horizon | Top 3 features by impurity importance (Random Forest) |
|---|---|---|
| Children in HHS Care | h=1 (~1 day) | `lag_1_Children in HHS Care` (0.136), `rolling_7_mean_Children in HHS Care` (0.135), `rolling_14_mean_Children in HHS Care` (0.104) |
| Children in HHS Care | h=7 (~9 days) | `rolling_7_mean_Children in HHS Care` (0.132), `lag_1_Children in HHS Care` (0.132), `rolling_14_mean_Children in HHS Care` (0.099) |
| Children in HHS Care | h=14 (~20 days) | `rolling_7_mean_Children in HHS Care` (0.135), `lag_1_Children in HHS Care` (0.130), `rolling_14_mean_Children in HHS Care` (0.093) |
| Children discharged from HHS Care | h=1 (~1 day) | `rolling_7_mean_Children in HHS Care` (0.109), `lag_1_Children in HHS Care` (0.103), `lag_7_Children in HHS Care` (0.096) |
| Children discharged from HHS Care | h=7 (~9 days) | `rolling_7_mean_Children in HHS Care` (0.121), `lag_1_Children in HHS Care` (0.118), `rolling_14_mean_Children in HHS Care` (0.085) |
| Children discharged from HHS Care | h=14 (~20 days) | `rolling_7_mean_Children in HHS Care` (0.113), `lag_1_Children in HHS Care` (0.105), `lag_1_Children discharged from HHS Care` (0.095) |

Only Random Forest exposes impurity importances; `HistGradientBoostingRegressor`
does not, and substituting a different importance measure for it would produce a
column that is not comparable with the others, so it is reported as unavailable
rather than filled in. Values are extracted by
`src/reporting/feature_importance.py` from the persisted models and written to
`forecasts/feature_importance.csv` (342 rows).

**What the table shows, at the level it can support.** For `Children in HHS
Care` the leading features are derived from its own recent history — 77% of the
top-8 importance mass sits on own-series lags and rolling means, which is what
the persistence result in Section 15 would lead one to expect.

For `Children discharged from HHS Care` it does not. Only 24% of the top-8 mass
is on the discharge series itself; the model leans instead on care-load
features. A plausible reading is that the small-count, noisy discharge flow
carries less usable signal about its own future than the smooth stock it is
drawn from — but that is a conjecture the importances cannot settle, and the
Random Forest was the **least accurate** of the eight candidates in 2 of the 6
cells (Section 15). This is a description of what an unsuccessful model did, not
a finding about the series.

It does **not** establish that any particular lag is the important one, and it
is not evidence of a causal driver.

---

## 11. Forecasting Models

Eight candidates across four families, all scored identically.

**Baselines** — the standard against which complexity is judged, and eligible to
win outright.

| Model | Definition |
|---|---|
| `naive` | ŷ(t+h) = y(t). Persistence. |
| `seasonal_naive` | ŷ(t+h) = y(t+h−m), m=5 positions |
| `moving_average` | Mean of the trailing window |

**Statistical**

| Model | Notes |
|---|---|
| `sarima` | Seasonal ARIMA, order grid in `src/config.py`, seasonal period m=5; one fit per origin produces the whole horizon path |
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

## 12. Experimental Design

**Walk-forward, expanding window.** At each origin the model sees only data up
to its training cutoff, forecasts forward, and is scored against what actually
happened. The origin then advances and the process repeats.

| Parameter | Value | Source |
|---|---|---|
| Total folds | 65 | generated |
| Step between origins | 10 positions | config |
| Minimum initial training | 50 positions | config |
| Horizons | 1, 7, 14 positions | config |
| Training-window cap | 2025-02-05 | config |
| Reserved holdout | 60 observations | config |
| Random seed | 42 | config |

**The training-window cap.** Because of the regime shift (§8), models may also
be fitted on a window that begins at the cap date rather than at the start of
the series, so they learn the current regime instead of averaging across both.
Both window rules were evaluated; `capped` governs selection. A usable-row floor
protects the cap from starving a fit — if capping would leave too few rows, the
fallback widens the window, and every fold records whether that happened.

**Common support.** Models are compared only on test points that *every* model
successfully produced a forecast for. Without this, a model that silently failed
on the hardest folds would appear more accurate than one that attempted them all
— comparing averages over different sets of points is not a comparison.

**Two evidence scopes, reported separately.**

* `common_support` — all development folds. More statistical power; includes the
  pre-2025 regime.
* `post_cutoff_common_support` — post-cutoff folds only. Operationally relevant;
  per-cell sample size falls to 12–15 observations.

The recent-regime scope governs where the two disagree. Section 15 reports both,
including the cells where they disagree, because the disagreement is itself a
finding about statistical power.

---

## 13. Validation Strategy

**Metrics.** MAE (primary, in children), RMSE, MAPE, sMAPE, MASE, and ME (signed
bias). MAPE is reported but is **not** used for `Children discharged from HHS
Care`: that series reaches zero, where MAPE is undefined and near-zero values
make it explode. sMAPE and MASE are the scale-free metrics there.

**MASE denominator.** Computed from the in-sample naive error of each fold's own
training data, **anchored at the fold origin**. An earlier version derived it
from the training window, which made the denominator depend on the window rule
and silently corrupted the full-versus-capped comparison — the comparison was
between models measured on different scales. Origin anchoring fixes this.

**Practical equivalence.** A paired bootstrap over per-observation absolute
errors on the same test points: 10,000 resamples at the 95% level. Paired,
because the same fold is easy or hard for every model; bootstrap rather than a
fixed percentage margin, because the margin must adapt to a sample size that
varies by cell. If the interval on the paired difference spans zero, the models
are not distinguishable and the simpler one wins.

**Prediction intervals.** Empirical quantiles of **out-of-sample** walk-forward
residuals at the matching horizon — never in-sample residuals, which would be
optimistic by construction. Nominal level 95%. Below 10 residuals no interval is
emitted at all: the 2.5th and 97.5th percentiles of a handful of points are just
the sample minimum and maximum, which is not an interval estimate. Lower bounds
are clipped at zero for display, since a negative count of children is not a
possible outcome, but coverage is assessed on the **unclipped** bounds so
clipping cannot flatter it.

**Coverage assessment.** Wilson score binomial confidence bands on the empirical
coverage rate. The Wilson interval is used rather than the normal approximation
precisely because n is small and the observed proportion is near 1, where the
normal approximation misbehaves.

**The holdout.** The final 60 observations were untouched until champions were
frozen, then read once. Nothing from it fed back into selection — recorded in
the provenance as `holdout_use = "single confirmatory coverage check; never fed
back into selection"`.

**Test suite.** The pipeline is covered by an automated suite spanning data
contracts, feature construction, metric correctness, leakage invariants, model
behaviour, selection logic, app rendering, and deployment readiness.

---

## 14. Results

### 14.1 Selected champions

| Target | Horizon | Champion | MAE (children) | n scored | Numerical leader |
|---|---|---|---|---|---|
| Children in HHS Care | h=1 (~1 day) | **Naive (persistence)** | 10.13 | 15 | none cleared gate |
| Children in HHS Care | h=7 (~9 days) | **Naive (persistence)** | 70.21 | 14 | Exponential smoothing (ETS) |
| Children in HHS Care | h=14 (~20 days) | **Naive (persistence)** | 147.64 | 14 | Exponential smoothing (ETS) |
| Children discharged from HHS Care | h=1 (~1 day) | **Naive (persistence)** | 4.14 | 14 | Ensemble (statistical + ML average) |
| Children discharged from HHS Care | h=7 (~9 days) | **Naive (persistence)** | 8.46 | 13 | Moving average (w=7) |
| Children discharged from HHS Care | h=14 (~20 days) | **Seasonal naive (m=5)** | 5.33 | 12 | none cleared gate |

The "numerical leader" column is the model with the lowest MAE that cleared the
baseline-beating gate. Where it differs from the champion, the paired bootstrap
could not distinguish the two and the simpler model was selected. Where it reads
"none cleared gate", no candidate beat both naive and seasonal-naive strictly.

**Accuracy is not comparable across the two targets.** Care load is a stock in
the low thousands, discharges a flow in the low tens; an MAE of 147.6 children
on a base of ~2,484 is a far smaller relative error than 5.3 on a base in the
tens.

### 14.2 Interval coverage

| Target | Horizon | Champion | n | Empirical | Wilson band | Consistent with nominal |
|---|---|---|---|---|---|---|
| Children in HHS Care | h=1 (~1 day) | Naive (persistence) | 15 | 86.7% | 62.1–96.3% | yes |
| Children in HHS Care | h=7 (~9 days) | Naive (persistence) | 14 | 71.4% | 45.4–88.3% | **no** |
| Children in HHS Care | h=14 (~20 days) | Naive (persistence) | 14 | 85.7% | 60.1–96.0% | yes |
| Children discharged from HHS Care | h=1 (~1 day) | Naive (persistence) | 15 | 80.0% | 54.8–93.0% | **no** |
| Children discharged from HHS Care | h=7 (~9 days) | Naive (persistence) | 14 | 85.7% | 60.1–96.0% | yes |
| Children discharged from HHS Care | h=14 (~20 days) | Seasonal naive (m=5) | 12 | 83.3% | 55.2–95.3% | yes |

Nominal level is 95%. Read the **band**, not the point estimate: with n of 12–15
these intervals span tens of percentage points. The honest summary is that the
intervals are *not contradicted* by the evidence, which is a much weaker claim
than calibration.

### 14.3 Held-out confirmation

The reserved window, read once after champions were frozen:

| Target | Horizon | n | Holdout MAE | Coverage |
|---|---|---|---|---|
| Children discharged from HHS Care | h=1 (~1 day) | 5 | 5.00 | 100% |
| Children discharged from HHS Care | h=7 (~9 days) | 5 | 5.20 | 100% |
| Children discharged from HHS Care | h=14 (~20 days) | 5 | 4.50 | 100% |
| Children in HHS Care | h=1 (~1 day) | 5 | 6.20 | 100% |
| Children in HHS Care | h=7 (~9 days) | 5 | 37.20 | 100% |
| Children in HHS Care | h=14 (~20 days) | 5 | 82.60 | 100% |

With 5 points per cell this is confirmatory only. It is reported because the
holdout was promised and must be reported, not because it constitutes a precise
accuracy estimate.

---

## 15. Model Comparison

This section reports the **full comparison matrix** — every model, every cell,
including the runs that beat the selected champion numerically. The winning path
alone would misrepresent what the evidence shows, because in most cells the
numerical ranking and the selection decision do not coincide.

All figures are on the governing scope (`post_cutoff_common_support`, window
rule `capped`), with common support, so every model in a table was scored on
exactly the same test points.

> **A note on how this matrix was assembled.** The Day-7 comparison artifact
> contains seven models; the ensemble is constructed at Day 8 from forecasts
> that run had already produced, so it was never written into that file — even
> though it is the numerical leader in one cell. Reporting the matrix without it
> would be reporting a winning path rather than a full comparison, so the
> ensemble was scored and appended. It was scored on **the support set the other
> seven were scored on**, not on a freshly recomputed eight-model one: rebasing
> common support would have shifted all seven sets of numbers and put this paper
> silently at odds with the dashboard and the frozen registry. The other seven
> rows are carried across unchanged, and a test asserts that. As confirmation
> that the scoring is equivalent, the ensemble MAE computed here reproduces the
> `numerical_leader_mae` already recorded in `models/model_registry.json`.

#### Children in HHS Care — h=1 (~1 day)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Naive (persistence) ⬅ **champion** | baseline | 15 | 10.13 | 13.97 | 0.44 | 0.037 | +0.00 |
| SARIMA | statistical | 15 | 11.44 | 16.36 | 0.50 | 0.041 | +10.18 |
| Ensemble (statistical + ML average) | ensemble | 15 | 18.70 | 30.45 | 0.79 | 0.067 | +16.06 |
| Exponential smoothing (ETS) | statistical | 15 | 28.57 | 41.63 | 1.27 | 0.102 | +24.70 |
| Gradient Boosting | machine learning | 15 | 34.16 | 51.94 | 1.44 | 0.124 | +21.95 |
| Moving average (w=7) | baseline | 15 | 42.97 | 59.51 | 1.84 | 0.156 | +16.13 |
| Seasonal naive (m=5) | baseline | 15 | 56.40 | 82.52 | 2.39 | 0.203 | +27.33 |
| Random Forest | machine learning | 15 | 103.19 | 171.10 | 4.22 | 0.370 | +50.42 |

*Selection:* no candidate beat both naive and seasonal-naive; the best baseline
is the champion and is preserved as the result

#### Children in HHS Care — h=7 (~9 days)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Exponential smoothing (ETS) | statistical | 14 | 61.52 | 77.77 | 2.70 | 0.225 | +42.88 |
| SARIMA | statistical | 14 | 62.39 | 79.88 | 2.82 | 0.223 | -29.93 |
| Naive (persistence) ⬅ **champion** | baseline | 14 | 70.21 | 86.87 | 3.08 | 0.256 | +26.50 |
| Moving average (w=7) | baseline | 14 | 105.73 | 133.77 | 4.58 | 0.385 | +43.84 |
| Seasonal naive (m=5) | baseline | 14 | 110.57 | 143.26 | 4.79 | 0.401 | +48.14 |
| Ensemble (statistical + ML average) | ensemble | 14 | 111.11 | 149.34 | 4.83 | 0.404 | +77.24 |
| Random Forest | machine learning | 14 | 184.79 | 277.91 | 7.72 | 0.668 | +111.60 |
| Gradient Boosting | machine learning | 14 | 198.30 | 312.97 | 8.21 | 0.714 | +135.38 |

*Selection:* champion is the simplest candidate not distinguishable from the
numerical leader

> **The two evidence scopes disagree in this cell.** On all development folds
> (n=55) the same rule selects **SARIMA**; on the recent regime (n=14) it
> selects **Naive (persistence)**. This is a statement about statistical power,
> not model quality: at this sample size the paired bootstrap cannot separate
> the candidates, so the rule falls back to simplicity.

#### Children in HHS Care — h=14 (~20 days)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Exponential smoothing (ETS) | statistical | 14 | 108.74 | 127.25 | 4.91 | 0.403 | -6.40 |
| Naive (persistence) ⬅ **champion** | baseline | 14 | 147.64 | 176.41 | 6.53 | 0.542 | +43.93 |
| Seasonal naive (m=5) | baseline | 14 | 152.64 | 181.30 | 6.76 | 0.561 | +46.79 |
| Ensemble (statistical + ML average) | ensemble | 14 | 156.60 | 233.55 | 6.79 | 0.567 | +101.57 |
| Moving average (w=7) | baseline | 14 | 179.84 | 220.55 | 7.88 | 0.658 | +63.37 |
| SARIMA | statistical | 14 | 220.44 | 307.00 | 10.98 | 0.779 | -168.23 |
| Random Forest | machine learning | 14 | 287.01 | 461.72 | 11.61 | 1.032 | +209.54 |
| Gradient Boosting | machine learning | 14 | 411.42 | 605.42 | 15.98 | 1.460 | +321.60 |

*Selection:* champion is the simplest candidate not distinguishable from the
numerical leader

> **The two evidence scopes disagree in this cell.** On all development folds
> (n=60) the same rule selects **SARIMA**; on the recent regime (n=14) it
> selects **Naive (persistence)**. This is a statement about statistical power,
> not model quality: at this sample size the paired bootstrap cannot separate
> the candidates, so the rule falls back to simplicity.

#### Children discharged from HHS Care — h=1 (~1 day)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Ensemble (statistical + ML average) | ensemble | 14 | 3.17 | 5.85 | 19.26 | 0.091 | +0.34 |
| Exponential smoothing (ETS) | statistical | 14 | 3.82 | 6.27 | 24.23 | 0.113 | -1.42 |
| Naive (persistence) ⬅ **champion** | baseline | 14 | 4.14 | 5.46 | 36.63 | 0.121 | -1.43 |
| Gradient Boosting | machine learning | 14 | 4.44 | 7.00 | 28.17 | 0.129 | +2.10 |
| Moving average (w=7) | baseline | 14 | 4.47 | 8.23 | 24.90 | 0.128 | +1.02 |
| SARIMA | statistical | 14 | 5.99 | 8.01 | 58.81 | 0.175 | -2.95 |
| Seasonal naive (m=5) | baseline | 14 | 6.64 | 13.61 | 31.12 | 0.191 | +2.79 |
| Random Forest | machine learning | 14 | 7.35 | 18.33 | 30.66 | 0.208 | +5.23 |

*Selection:* champion is the simplest candidate not distinguishable from the
numerical leader

> **The two evidence scopes disagree in this cell.** On all development folds
> (n=60) the same rule selects **SARIMA**; on the recent regime (n=14) it
> selects **Naive (persistence)**. This is a statement about statistical power,
> not model quality: at this sample size the paired bootstrap cannot separate
> the candidates, so the rule falls back to simplicity.

#### Children discharged from HHS Care — h=7 (~9 days)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Moving average (w=7) | baseline | 13 | 6.61 | 8.50 | 48.60 | 0.196 | +0.99 |
| Exponential smoothing (ETS) | statistical | 13 | 6.78 | 9.66 | 50.01 | 0.200 | +1.68 |
| Ensemble (statistical + ML average) | ensemble | 13 | 8.36 | 12.24 | 50.84 | 0.243 | +2.73 |
| Naive (persistence) ⬅ **champion** | baseline | 13 | 8.46 | 11.92 | 61.22 | 0.248 | -1.23 |
| Seasonal naive (m=5) | baseline | 13 | 9.69 | 15.37 | 55.04 | 0.282 | +5.85 |
| Random Forest | machine learning | 13 | 10.65 | 17.02 | 55.77 | 0.309 | +3.79 |
| SARIMA | statistical | 13 | 11.10 | 13.34 | 95.87 | 0.334 | -0.21 |
| Gradient Boosting | machine learning | 13 | 12.23 | 19.02 | 63.97 | 0.354 | +5.61 |

*Selection:* champion is the simplest candidate not distinguishable from the
numerical leader

> **The two evidence scopes disagree in this cell.** On all development folds
> (n=46) the same rule selects **Seasonal naive (m=5)**; on the recent regime
> (n=13) it selects **Naive (persistence)**. This is a statement about
> statistical power, not model quality: at this sample size the paired bootstrap
> cannot separate the candidates, so the rule falls back to simplicity.

#### Children discharged from HHS Care — h=14 (~20 days)

| Model | Family | n | MAE | RMSE | sMAPE | MASE | ME (bias) |
|---|---|---|---|---|---|---|---|
| Seasonal naive (m=5) ⬅ **champion** | baseline | 12 | 5.33 | 7.06 | 57.86 | 0.157 | -1.00 |
| Exponential smoothing (ETS) | statistical | 12 | 7.33 | 9.90 | 67.64 | 0.215 | +2.10 |
| Ensemble (statistical + ML average) | ensemble | 12 | 7.52 | 11.65 | 54.76 | 0.217 | +4.28 |
| Moving average (w=7) | baseline | 12 | 7.68 | 10.72 | 55.82 | 0.226 | +6.42 |
| Gradient Boosting | machine learning | 12 | 8.26 | 14.56 | 49.69 | 0.236 | +6.46 |
| Naive (persistence) | baseline | 12 | 9.33 | 10.97 | 77.81 | 0.277 | +4.17 |
| Random Forest | machine learning | 12 | 12.01 | 21.47 | 62.81 | 0.341 | +10.21 |
| SARIMA | statistical | 12 | 18.07 | 19.85 | 171.31 | 0.531 | -14.27 |

*Selection:* no candidate beat both naive and seasonal-naive; the best baseline
is the champion and is preserved as the result

### 15.1 What the comparison shows

**By family, champions won:**

| Family | Cells won | of |
|---|---|---|
| Baseline | 6 | 6 |
| Statistical | 0 | 6 |
| Machine learning | 0 | 6 |

**The secondary objective's answer (O6): neither.** Neither the statistical
family nor the ML family produced an advantage over the baselines that survived
a paired bootstrap on the governing scope. Three things explain this, and they
are worth separating because they have different implications:

1. **The stock target is close to a random walk at short horizons.** When
   successive observations rarely move far, ŷ(t+h) = y(t) is genuinely hard to
   beat, and the MASE values above show most models hovering around the naive
   error rather than below it. This is a property of the series, not a
   deficiency of the models.

2. **The governing sample is small.** At 12–15 scored points per cell, a paired
   bootstrap simply cannot resolve modest differences. Some of these "no
   distinguishable difference" verdicts would likely resolve with more
   post-regime-shift data. The correct reading is *not proven better*, not
   *proven equal*.

3. **The regime shift destroyed most of the usable history.** The ML models in
   particular are being asked to learn from roughly a hundred usable rows. That
   is not a fair test of what gradient boosting can do; it is an accurate test
   of what it can do *here*.

**This is a real result, not a placeholder.** Forcing a complex model into a
cell whose bootstrap interval spans zero would be selecting on noise and would
produce a system that looks sophisticated and forecasts no better. The
defensible position is the one the evidence supports: use the simple model, and
say why.

---

## 16. Forecast Analysis

### 16.1 Forward forecasts

Origin 2025-12-21, the last observed period. Champion model per cell; intervals
are empirical out-of-sample residual quantiles at the 95% nominal level.

| Target | Horizon | Model | Point forecast | 95% interval | n residuals | Lower clipped at 0 |
|---|---|---|---|---|---|---|
| Children in HHS Care | h=1 (~1 day) | Naive (persistence) | 2,484.0 | 2,458.1 – 2,510.1 | 15 | no |
| Children in HHS Care | h=7 (~9 days) | Naive (persistence) | 2,484.0 | 2,329.0 – 2,558.7 | 14 | no |
| Children in HHS Care | h=14 (~20 days) | Naive (persistence) | 2,484.0 | 2,112.0 – 2,618.1 | 14 | no |
| Children discharged from HHS Care | h=1 (~1 day) | Naive (persistence) | 14.0 | 6.0 – 24.0 | 15 | no |
| Children discharged from HHS Care | h=7 (~9 days) | Naive (persistence) | 14.0 | 0.0 – 39.7 | 14 | yes |
| Children discharged from HHS Care | h=14 (~20 days) | Seasonal naive (m=5) | 16.0 | 1.9 – 24.7 | 12 | no |

**Two features of this table deserve comment.**

*Flat point forecasts.* Where the champion is `naive`, the point forecast is
identical at every horizon by construction — persistence has no trend term. The
horizon-dependence lives entirely in the interval, which widens from 52 to 506
children for care load between h=1 and h=14. A flat forecast with an honestly
widening interval is a more truthful representation of what is known than a
sloped line with a narrow one.

*A clipped lower bound.* One discharge interval reached below zero and was
clipped for display. The unclipped value is retained in `lower_unclipped` and
coverage is assessed on the unclipped bound, so the clip is presentational only
and cannot improve the measured coverage.

### 16.2 Intake versus exit pressure

The derived imbalance signal is transferred-in minus discharged. Its variance
uses the measured paired residual correlation rather than assuming independence:

| Horizon | Transferred in | Discharged | Net pressure | 1 s.d. | Interpretation |
|---|---|---|---|---|---|
| h=1 (~1 day) | 9.0 | 14.0 | **-5.0** | ± 8.0 | exits exceed intake -- net relief |
| h=7 (~9 days) | 15.0 | 14.0 | **+1.0** | ± 14.3 | intake exceeds exits -- net inflow pressure |
| h=14 (~20 days) | 6.0 | 16.0 | **-10.0** | ± 9.0 | exits exceed intake -- net relief |

Uncertainty form: `independence`, selected from a measured correlation of -0.153
at h=1 on 14 paired residuals — below the pre-registered independence threshold,
so the simplified variance form is admissible. Note that the addendum's *prior*
was near-independence and the measurement confirmed it; had it not, the
covariance form would have been used instead.

> **The net pressure numbers are not decision-grade, and the table shows why.**
> The standard deviation exceeds the net pressure itself by up to a factor of
> 14.3. The sign of the imbalance is not resolvable at this sample size. The
> direction shown should be read as the central estimate of a quantity whose
> interval comfortably includes the opposite sign.

---

## 17. Early-Warning System

### 17.1 The horizon-versus-decision-timescale gap

**This subsection comes first because it qualifies everything after it.**

The early-warning layer fires a tiered alert when a forecast crosses a
data-derived threshold. Measured over 65 backtest origins, its median lead time
is:

| Target | Median lead (periods) | Approximate wall-clock | Min | Max |
|---|---|---|---|---|
| Children in HHS Care | 1.0 | about a day | 1 | 1 |
| Children discharged from HHS Care | 2.5 | about three days | 2 | 3 |

**The decisions this signal would inform take weeks.** Opening or expanding a
facility, recruiting and onboarding care staff, and scaling sponsor-vetting
throughput all have lead times measured in weeks to months. A warning that
arrives one to three days ahead of a threshold crossing arrives after the point
at which any of those decisions could have been made differently.

Stated plainly: **the early-warning layer does not provide enough notice to
change a capacity decision.** It is useful for near-term operational tempo —
shift rostering, transport scheduling, prioritising the discharge queue — and it
should not be presented as capacity planning. No amount of model improvement
closes this gap, because the gap is not caused by model error. Even a perfect
forecaster at h=14 sees ~20 days ahead. Closing it requires a longer forecast
horizon on data that supports one, or leading indicators from upstream of the
published series.

### 17.2 Design

Tiers, by the horizon at which the threshold is first crossed:

| Horizon | Tier | Reading |
|---|---|---|
| h=14 (~20 days) | Watch | furthest out, weakest evidence |
| h=7 (~9 days) | Warning | mid-range |
| h=1 (~1 day) | Alert | imminent |

**The threshold is a proxy and is labelled as one everywhere.** It is the 90th
percentile of the trailing 60 periods — current values are 2,450.6 for care load
and 16.0 for discharges. Verbatim from the artifact:

> Relative, data-derived proxy for unusually high load by this programme's own
> recent standards. No official capacity threshold exists in the source
> documentation or the dataset; this is not an official capacity figure and does
> not indicate that capacity has been or will be breached.

Because the threshold is relative to recent history, it re-bases as the regime
moves. A "High" reading means high *by this program's own recent standards*, and
after a sustained shift a formerly alarming absolute level stops firing.

### 17.3 Sensitivity to the threshold choice

The operating percentile is a free parameter, so it was swept rather than
asserted. Percentiles 75, 80, 85, 90, 95:

| Target | Percentile | Fired | TP | FP | FN | FN rate | Median lead |
|---|---|---|---|---|---|---|---|
| Children in HHS Care | 75 | 16 | 16 | 0 | 5 | 0.102 | 1.0 |
| Children in HHS Care | 80 | 16 | 15 | 1 | 5 | 0.102 | 1.0 |
| Children in HHS Care | 85 | 15 | 15 | 0 | 4 | 0.080 | 1.0 |
| Children in HHS Care | 90  ⬅ frozen | 13 | 13 | 0 | 6 | 0.115 | 1.0 |
| Children in HHS Care | 95 | 11 | 11 | 0 | 7 | 0.130 | 1.0 |
| Children discharged from HHS Care | 75 | 8 | 8 | 0 | 42 | 0.737 | 1.0 |
| Children discharged from HHS Care | 80 | 5 | 5 | 0 | 43 | 0.717 | 1.0 |
| Children discharged from HHS Care | 85 | 3 | 3 | 0 | 36 | 0.581 | 2.0 |
| Children discharged from HHS Care | 90  ⬅ frozen | 2 | 2 | 0 | 30 | 0.476 | 2.5 |
| Children discharged from HHS Care | 95 | 1 | 1 | 0 | 20 | 0.312 | 13.0 |

**What the sweep shows.** For care load the signal is stable: precision stays at
or near 1.0 across the whole range and the false-negative rate moves only a few
points, so the choice of percentile is not load-bearing. For discharges it is
not stable — the false-negative rate runs from 0.312 to 0.737, and at the frozen
operating point the signal fires only 2 times in 65 origins while missing 30
crossings.

**The discharge early-warning signal should not be relied on.** A false-negative
rate of 0.476 means it misses nearly half the crossings it exists to catch. Its
zero false-positive rate is not a virtue here — it is the same fact viewed from
the other side: the threshold is high enough that it almost never fires. Note
also that the apparent improvement at the 95th percentile (a median lead of 13
periods) rests on 1 firing. A median over a single observation is not an
estimate of anything.

---

## 18. Dashboard

An eight-page Streamlit application. Its defining architectural property: **it
reads pre-generated artifacts and never trains anything.** No page fits a model,
no page recomputes a metric. Every number on screen traces to a CSV produced by
the pipeline, through a single data-access layer (`app/lib/artifacts.py`) that
is the app's only route to data.

| Page | Purpose |
|---|---|
| Executive Overview | KPIs, forward forecasts, current capacity tier |
| Historical Trends | Observed series with imputed values visibly marked |
| Care Load Forecast | Core module — `Children in HHS Care` |
| Discharge Demand Forecast | Core module — `Children discharged from HHS Care` |
| Intake vs. Exit Pressure | Derived imbalance signal with its uncertainty |
| Model Comparison & Accuracy | Full comparison matrix and the selection decision |
| Scenario Comparison | Model-versus-model at equal footing |
| Methodology & Data | Provenance, hashes, limitations, refresh policy |

Three conventions are applied consistently, and each exists to prevent a
specific misreading:

* **Horizons are labelled in both units.** "h=14 (~20 days)", never "14 days" —
  the reporting calendar is not the calendar, and a user who assumes otherwise
  will mis-plan by nearly a week.
* **Imputed values are marked wherever they are plotted.** A user must be able
  to tell an observation from an interpolation at a glance.
* **The capacity proxy carries its disclaimer on every page it appears on**, not
  once in a methodology footnote.

The two forecast pages share one renderer, so the mandated core modules cannot
drift apart in behaviour or presentation.

---

## 19. Deployment

**Platform.** Streamlit Community Cloud, from the `master` branch, entry point
`app/Home.py`.

**Artifact strategy.** Since the app is a pure consumer of pre-generated files,
a deployment needs no build step — but it does need those files present in the
clone. The `.gitignore` is therefore an allow-list: the artifacts the dashboard
actually reads are committed; the regenerable intermediates are not.

**Provenance verification.** A dashboard that reads pre-generated files has a
specific silent failure mode: it keeps serving happily while showing a data
vintage that no longer matches the repository. Nothing errors; the numbers are
simply stale. To close this, the app publishes its provenance sidecar as a
static file, and `scripts/smoke_test.py` fetches it from the live URL and
compares the raw-CSV and derived-series SHA-256 byte for byte against the local
artifact. The script exits non-zero on any failure, so it can gate a deploy.

**Refresh policy.** `manual only -- replace the CSV and re-run this script`.
Nothing retrains on a schedule. This is a deliberate choice: an unattended
retrain on a series with a regime shift of the magnitude in Section 8 would
silently change every conclusion in this paper without anyone reviewing it.

**Environment.** A clean-environment install was verified as part of deployment
readiness, and it caught a real defect: a fresh resolve pulls pandas 3.x where
development ran on pandas 2.x, and one page assigned `pd.NA` into a plain
boolean column — tolerated by pandas 2, a `TypeError` in pandas 3. Fixed by
declaring the nullable `boolean` dtype explicitly.

---

## 20. Limitations

These are ordered by how much they should change a reader's confidence in the
conclusions. The first two are severe enough that they constrain what the system
may legitimately be used for.

### L1 — The warning horizon does not reach the decision horizon

Median lead time is 1.0 reporting periods for care load and 2.5 for discharges —
one to three days. Facility, staffing, and sponsor-vetting decisions take weeks.
**The system cannot support capacity planning**, which is the use its framing
most naturally invites. This is not a model-quality problem and cannot be fixed
by a better model: the longest horizon evaluated, h=14, is only ~20 days of
wall-clock time. See Section 17.1.

### L2 — The measured flows do not explain the measured stock (omitted variable)

The accounting identity `care[t] = care[t−1] + transferred_in[t] −
discharged[t]` should hold if the published series described one closed system.
Tested on the 682 periods where every term is genuinely observed, it holds
exactly in 18 of them — 2.64% — with a median absolute discrepancy of 24.0
children and a **mean signed discrepancy of +42.0 children per period**.

The sign is the informative part. The stock systematically rises by more than
the published flows account for, which means **at least one material intake
channel into HHS care is absent from this dataset** — the discrepancy is a
persistent bias, not measurement noise, which would average toward zero.

The consequence is direct and unhedgeable. Every model in this study forecasts
the observable series and none can represent the missing channel. A shift in
that unmeasured channel would move care load in a way no model here could
anticipate from the data it is given, and would degrade every forecast
simultaneously — the errors would not be independent across models, so an
ensemble offers no protection either. Any use of these forecasts must assume the
unmeasured channel continues to behave as it has. **That assumption is
untestable with this dataset.**

### L3 — Interval calibration is unverified

Per-cell coverage samples are 12–15 observations and the Wilson bands span tens
of percentage points. The intervals are not contradicted by the evidence; that
is not the same as being calibrated. 2 of 6 cells are already inconsistent with
the nominal level.

### L4 — No official capacity threshold exists

Every capacity-stress statement rests on a statistical proxy derived from the
program's own recent history. No reading indicates that actual capacity has been
or will be breached, because the data contains no capacity figure to compare
against. Because the proxy is relative, it also re-bases as the regime moves.

### L5 — Small effective sample after the regime shift

The governing evidence scope holds 12–15 scored points per cell. Most "no
distinguishable difference" verdicts in Section 15 mean *not proven better*, not
*proven equal*. Some would likely resolve with more post-shift data.

### L6 — The discharge early-warning signal misses nearly half its targets

False-negative rate 0.476 at the frozen operating point (Section 17.3). It
should not be relied on as a discharge-surge detector.

### L7 — The stability KPI is a documented substitution

The roadmap defines forecast stability as agreement between forecasts of the
same target date made from different origins. That is not computable here: 0 of
195 test positions are reached from more than one origin, because the fold step
(10) exceeds the horizon spacing. The substituted definition — p90/median of
absolute error across holdout origins — is recorded in the artifact itself
rather than presented as if it were the original metric.

### L8 — Single national aggregate, no covariates

The data is one national series with no regional, facility-level, policy, or
seasonal-migration covariates. Forecasts cannot be localised, and nothing here
attributes cause.

---

## 21. Recommendations

**For anyone using these forecasts**

1. **Use the care-load forecast for near-term operational tempo, not capacity
   planning.** Shift rostering, transport, and discharge-queue prioritisation
   sit within the horizon this system can actually see. Facility and staffing
   decisions do not (L1).
2. **Read the interval, not the point.** Where the champion is `naive` the point
   forecast is flat by construction; all horizon information is in the width of
   the band.
3. **Treat the imbalance sign as unresolved.** In Section 16.2 the standard
   deviation exceeds the net pressure itself; the direction shown is a central
   estimate whose interval includes the opposite sign.
4. **Do not act on the discharge early-warning signal alone** (L6,
   false-negative rate 0.476).
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
   L2 and L4 are the two limitations that no modelling choice can remove, and
   both are properties of what is published rather than of the analysis.

## 22. Future Work

Ordered by expected value, not by novelty. The highest-value items are
unglamorous, and the glamorous ones are unlikely to help.

1. **Close the reconciliation gap (L2).** Identifying the unmeasured intake
   channel would do more for forecast quality than any modelling change in this
   list. It is the binding constraint.
2. **Extend the horizon to the decision timescale (L1).** A system that
   forecasts 4–8 weeks out would be usable for the decisions this one cannot
   support. Whether the data sustains that is an open question and should be
   tested honestly — a long-horizon forecast with an interval spanning the
   plausible range is not an improvement.
3. **Seek leading indicators upstream.** Apprehension and CBP-custody series
   lead HHS care mechanically. Exploiting the lag structure explicitly — rather
   than as generic lagged features — is the most promising route to genuine lead
   time.
4. **Accumulate post-shift data and re-run selection.** Cheap, automatic, and
   will settle several currently-undecidable comparisons (L5).
5. **Revisit hierarchical or regional modelling if such data is ever
   published.** Not possible with a single national aggregate (L8).
6. **Probabilistic models fitted to the count structure** (e.g. negative
   binomial for the discharge flow) would give intervals that respect the
   non-negative integer support natively, instead of relying on a display-time
   clip.

**What is unlikely to help:** more model families, or deeper ones. Section 15
shows that under the current sample size the paired bootstrap cannot distinguish
the eight candidates already evaluated. Adding a ninth would add a row to the
table and nothing to the decision.

---

## 23. Conclusion

This study delivered a validated multi-target, multi-horizon forecasting system
for the HHS UAC Program, with prediction intervals, an early-warning layer, and
a deployed dashboard — all five primary objectives met.

The substantive findings are three, and two of them are constraints rather than
capabilities.

**Simple baselines win.** In 6 of 6 cells the champion is a baseline. Neither
the statistical nor the machine-learning family produced an advantage that
survived a paired bootstrap on the operationally relevant regime. The care-load
series is close to a random walk at short horizons, and the regime shift left
roughly a hundred usable rows to learn from. Reported as the result, with the
full matrix in Section 15, rather than resolved by promoting a model the
evidence does not support.

**The system cannot do capacity planning.** A median lead time of 1.0 reporting
periods against decisions that take weeks is not a gap a better model closes. It
is a property of the horizon the data supports. The useful scope is near-term
operational tempo, and the paper says so wherever the forecasts appear.

**The data describes an open system.** The flows reconcile to the stock in 2.64%
of testable periods, with the stock systematically running +42.0 children per
period above what the published flows explain. A material intake channel is
missing from the dataset, no model here can represent it, and every forecast is
conditional on its continued behaviour.

The methodological contribution is the discipline rather than the models: one
evaluation harness for every family, leakage invariants enforced in code and
tested, a selection rule frozen before the results were seen, baselines treated
as first-class candidates, and a paper generated from the artifacts so that no
figure in it can drift from the run that produced it. That discipline is what
makes the negative result trustworthy — and a negative result you can trust is
worth more than a positive one you cannot.

---

## 24. References

**Primary sources**

1. U.S. Department of Health and Human Services, *Unaccompanied Alien Children
   Program* data release. `HHS_Unaccompanied_Alien_Children_Program (1).csv`.
   SHA-256 `061af0a97a1b3bda7a36f0ce8df08b6994847b0a4c402828cf2ef835d4947198`.
2. Unified Mentor, *Project Allotment Portal* — project brief and requirements.

**Methods**

3. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
   (Walk-forward evaluation; MASE; seasonal-naive benchmarking.)
4. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast
   accuracy. *International Journal of Forecasting*, 22(4). (MASE; why MAPE
   fails near zero.)
5. Wilson, E. B. (1927). Probable inference, the law of succession, and
   statistical inference. *Journal of the American Statistical Association*,
   22(158). (Score interval used for coverage bands.)
6. Efron, B., & Tibshirani, R. J. *An Introduction to the Bootstrap*. (Paired
   bootstrap for practical equivalence.)
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
8. Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in
   random forest variable importance measures. *BMC Bioinformatics*, 8(25).
   (Basis for the collinearity caveat in Section 10.)
9. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for
   time series predictor evaluation. *Information Sciences*, 191.

**Software**

10. `pandas`, `numpy`, `statsmodels` (SARIMAX, ETS, STL), `scikit-learn`
    (`RandomForestRegressor`, `HistGradientBoostingRegressor`), `scipy`,
    `streamlit`, `plotly`, `pytest`.

**Project artifacts** — every figure in this paper is traceable to one of these:

11. `forecasts/` — `baseline_metrics.csv`, `baseline_predictions.csv`,
    `champion_selection.csv`, `comparison_matrix.csv`,
    `early_warning_backtest.csv`, `early_warning_sensitivity.csv`,
    `ensemble_predictions.csv`, `feature_importance.csv`,
    `forward_forecasts.csv`, `full_model_comparison.csv`,
    `holdout_evaluation.csv`, `imbalance_forecast.csv`,
    `imbalance_residual_correlation.csv`, `interval_coverage.csv`,
    `kpi_summary.csv`, `ml_metrics.csv`, `ml_predictions.csv`,
    `oos_residuals.csv`, `statistical_metrics.csv`,
    `statistical_predictions.csv`, `walk_forward_folds.csv`, `provenance.json`
12. `models/model_registry.json` — frozen selection decisions with bootstrap
    evidence
13. `docs/` — EDA findings, discrepancy log, requirements matrix, selection
    rationale, per-day metric reports
14. `reports/paper_evidence.json` — the claim-by-claim audit trail for this
    document

---

## 25. Appendix

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

Deterministic given the same input CSV: seed 42 throughout, and every artifact
carries the SHA-256 of the data it derives from.

### B. Frozen parameters

| Parameter | Value |
|---|---|
| `TRAINING_CAP_DATE` | 2025-02-05 |
| `FINAL_TEST_WINDOW` | 60 |
| `WALK_FORWARD_STEP` | 10 |
| `MIN_INITIAL_TRAINING` | 50 |
| `SEASONAL_PERIOD_M` | 5 |
| `FORECAST_HORIZONS` | 1, 7, 14 |
| `EARLY_WARNING_PERCENTILE` | 90 |
| `MIN_RESIDUALS_FOR_INTERVAL` | 10 |
| `PRACTICAL_EQUIVALENCE_RESAMPLES` | 10,000 |
| `RANDOM_SEED` | 42 |
| `SELECTION_SCOPE` | `post_cutoff_common_support` |
| `SELECTION_WINDOW_RULE` | `capped` |

### C. Walk-forward fold structure (first 5 of 65)

| Fold | Origin pos | Origin date | Train cutoff pos | Test pos h=1 | Test pos h=14 |
|---|---|---|---|---|---|
| 0 | 49 | 2023-03-22 | 49 | 50 | 63 |
| 1 | 59 | 2023-04-05 | 59 | 60 | 73 |
| 2 | 69 | 2023-04-19 | 69 | 70 | 83 |
| 3 | 79 | 2023-05-03 | 79 | 80 | 93 |
| 4 | 89 | 2023-05-17 | 89 | 90 | 103 |

Note `train_cutoff_pos` versus `origin_pos`. Where they differ, the origin's own
value is interpolated and the fold trains only to the last genuinely observed
position — the leakage guard described in Section 7.

### D. Out-of-sample residual pool

5,124 scored residuals across 7 models, 2 targets, and 3 horizons. This pool is
what the prediction intervals in Section 14.2 are built from — out-of-sample by
construction.

### E. Evidence ledger

Every numeral in this paper was issued by `Evidence.cite` in
`src/reporting/build_paper.py` and is recorded in `reports/paper_evidence.json`
with its source artifact and the locator within it. `tests/test_day13.py` scans
the finished document and fails if it contains a number the ledger never issued.
