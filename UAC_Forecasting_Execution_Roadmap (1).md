# Predictive Forecasting of Care Load & Placement Demand
## Master Execution Roadmap — 15-Day Plan (Solo Delivery)

**Program:** Unified Mentor — Project Allotment Portal
**Referenced context:** U.S. Department of Health and Human Services (HHS) / Unaccompanied Alien Children (UAC) Program
**Prepared as:** Senior technical lead blueprint — planning document only (no implementation performed)
**Sources of truth inspected:** (1) `Unified_Mentor___Project_Allotment_Portal.mht` — official project instructions page, (2) `HHS_Unaccompanied_Alien_Children_Program.csv` — actual dataset (1,170 rows × 6 columns as shipped)

---

## SOURCE VERIFICATION NOTE — READ FIRST

Both source files were opened and inspected directly (not assumed) before this roadmap was written. The official instructions were extracted from the `.mht` archive's HTML content; the dataset was loaded and profiled with `pandas`. The findings below are the evidence base for every decision in this document. Where the two sources agree, that's stated plainly. Where they don't — or where either source is silent on something a real implementation needs — it is flagged explicitly rather than papered over.

### Critical discrepancies / gaps found (evidence-based, not assumed)

| # | Finding | Evidence | Why it matters | How this roadmap handles it |
|---|---|---|---|---|
| 1 | The CSV has **1,170 rows but only 720 contain data**. Rows 721–1,170 (450 rows) are **100% blank** across all 6 columns. | Direct inspection: `df.notna().any(axis=1)` → 720 True, 450 False, with the blank block starting immediately after the last populated row and running to the end of the file. | The documented row/column profile in the file metadata (1,170 rows) is **not** the usable dataset size. Any EDA or model built on the raw row count without truncation will silently corrupt statistics (mean/std, date parsing, row counts). | Day 2 cleaning step explicitly truncates to the 720 real rows before anything else touches the data. This is the very first data-quality gate. |
| 2 | The documentation instructs to **"ensure continuity of daily observations"** and "handle missing days," implying a calendar-daily series. The actual data is **not calendar-daily**. | Day-of-week counts across all 720 real rows: Mon 145, Tue 149, Wed 147, Thu 147, **Fri 2, Sat 0**, Sun 130. Reporting is overwhelmingly **Sunday–Thursday**, not Monday–Friday and not 7-day. | If the pipeline naively reindexes to a full 7-day calendar grid, ~35% of the "daily" grid (Fri/Sat) would be 100% synthetic/interpolated data — a serious, avoidable distortion. | Day 2–4 treat the **true reporting schedule (Sun–Thu)** as the base frequency, not the calendar week. See Part 3 and Part 4 for the exact reindexing logic and its justification. |
| 3 | `Children in HHS Care` is typed as **String**, not Integer like its neighbors. | Every one of the 720 real values contains a thousands-separator comma (e.g. `"2,484"`), because every value is ≥ 1,000. Simple `int()`/`float()` casting fails without stripping the comma first. | This is the **primary forecasting target**. If it's left as text, every downstream step (EDA stats, feature engineering, modeling) breaks or silently miscomputes. | Day 2 cleaning explicitly strips thousands separators and casts to numeric, with a unit test asserting zero parse failures. |
| 4 | The other four numeric columns load as `float64`, not integers, even though every real value is a whole number. | This is caused by finding #1 — the 450 blank tail rows inject `NaN`, which forces pandas to upcast the whole column to float. Once the blank rows are dropped, all values are integers. | Minor in isolation, but it's a good canary: it proves the blank-row problem (#1) if a builder loads the file naively and wonders why "integer" counts look like floats. | Documented as a known/explained artifact, not a mystery, in the data dictionary produced Day 2. |
| 5 | The **KPI table in the official documentation is internally malformed.** | Raw HTML shows two consecutive `<tr>` rows: `Forecast Stability Index → Model robustness`, then `Model robustness → Long-term reliability`. The second row's KPI *name* is literally the first row's *description*, and its own description ("Long-term reliability") has no clear KPI name attached. | This is a genuine authoring defect in the source, not a data problem. Silently "fixing" it by inventing a plausible-sounding KPI name would violate the "don't invent requirements" rule. | Treated as **4 well-defined KPIs** (Forecast Accuracy %, Surge Lead Time, Capacity Breach Probability, Forecast Stability Index) plus **1 flagged, ambiguous 5th item** related to "long-term reliability" that is implemented only as P2/optional, with the ambiguity documented in the report rather than guessed at. See Part 2 and Part 15. |
| 6 | **No numeric capacity threshold** (e.g., "shelter capacity is X children") appears anywhere in either source. | Full-text extraction of the instructions page and full-column review of the CSV — no such figure exists in either. | The documented KPI "Capacity Breach Probability" and the general "capacity-stress" framing presuppose *some* reference point for "too many." Inventing one would be a fabricated requirement. | Part 7 defines a **relative/statistical proxy** (percentile- and rate-of-change-based), explicitly labeled as an engineering recommendation standing in for an absent official threshold — never presented as an authoritative capacity figure. |
| 7 | The column name `Children apprehended and placed in CBP custody*` carries a **footnote asterisk with no footnote text anywhere** in the extracted source. | Confirmed by full-text search of the extracted documentation; no asterisk/footnote definition exists on the page. | An unexplained asterisk on a government-style figure often flags a scope caveat (e.g., partial counting, later reclassification) in the original source. Guessing at its meaning would be fabrication. | The column is used strictly per its **plain documented definition** ("Daily intake volume") with the asterisk explicitly noted as an unresolved documentation gap in the report's Limitations section — not explained away. |
| 8 | `Children in HHS Care` swings from **1,972 to 11,516** (≈5.8×) over the observed window — a large, non-stationary regime shift, not noise around a stable mean. | Full-column statistics: min 1,972 (Aug 21, 2025), max 11,516 (Dec 20, 2023), mean 6,061, std 2,833. The series falls from its Dec-2023 peak to an Aug-2025 trough, then edges back up to ~2,400–2,500 by Dec 2025. | The documentation's framing (short-term surges, day-to-day planning) is real, but the data also contains a multi-year structural decline that plain baselines/ARIMA-without-differencing would handle poorly. | Part 5/Part 6 require explicit stationarity testing (ADF/KPSS) and differencing/regime-awareness before model selection — not assumed away, not ignored. |

These eight items recur throughout the roadmap wherever they're operationally relevant, tagged **[FLAG]**. Everywhere else, the two sources agree cleanly (see Part 3 for the full column-by-column reconciliation).

---

# PART 1 — COMPLETE PROJECT UNDERSTANDING

## 1.1 Project Identity

- **Official project title (from the portal):** *Predictive Forecasting of Care Load & Placement Demand*
- **Domain:** Public-sector / health & human-services operations analytics, applied as a time-series forecasting problem.
- **Organizational framing:** The project is assigned through the Unified Mentor project-allotment portal (`projects.unifiedmentor.com`) and is contextually framed around the U.S. Department of Health and Human Services and its Unaccompanied Alien Children (UAC) Program (`hhs.gov` is referenced as the contextual organization on the instructions page). This is a mentorship/portfolio-style project: the deliverables should be built **as if** for real HHS operational stakeholders, using real published-style data, but the engagement itself is via Unified Mentor, not a live HHS contract. That framing should be reflected honestly in the report (e.g., "prepared as an independent analytical exercise using HHS UAC program data" rather than implying an official HHS work product).
- **Intended stakeholders/users (per documentation):**
  - **Primary:** HHS decision-makers / healthcare and shelter-capacity planners — the audience the dashboard and executive summary are written for.
  - **Secondary:** The Unified Mentor reviewer/mentor evaluating the submission against the stated requirements.

## 1.2 Real-World Problem

The UAC Program is described as operating in a high-uncertainty environment: shifts in border activity, policy enforcement, or humanitarian crises can rapidly increase the number of children entering federal care. The current state, per the documentation, is **descriptive only** — historical counts are reported, but nothing forward-looking is produced from them.

**Why descriptive reporting is insufficient:** A daily count tells decision-makers what already happened, not what is about to happen. By the time a surge is visible in the raw numbers, the lead time needed to act — opening shelter capacity, scheduling medical staff, assigning caseworkers — has already been partly or fully lost. The dataset itself makes this concrete: HHS care load moved across a **~5.8× range** (1,972 to 11,516) over the ~3 years of available data. A planning process driven only by "what is the count today" has no mechanism to anticipate a move of that magnitude before it happens.

**What decisions need to become proactive (per documentation):**
- How many children will be under HHS care in the coming days/weeks (staffing and bed-capacity planning)
- Whether discharge (sponsor-placement) capacity will be sufficient to offset incoming transfers (throughput planning)
- When shelters, medical staff, and caseworkers should be scaled up **in advance** of a surge, not in response to one

**Risks when forecasting is unavailable (per documentation):** overcrowding risk, staff burnout, and increased length of stay for children — i.e., operational and welfare consequences, not just reporting inconvenience.

## 1.3 Exact Problem We Are Solving

Translating the narrative objectives into precise analytical problems, grounded in what the actual columns can support:

| Documented objective | Analytical translation | Grounded in |
|---|---|---|
| "Forecast the number of children in HHS care" | **Care-load forecasting**: multi-horizon time-series forecast of the `Children in HHS Care` stock variable, with uncertainty bands. | Doc objective (primary) + column exists as a clean, high-variance target |
| "Predict short-term discharge demand" | **Discharge-demand forecasting**: multi-horizon time-series forecast of the `Children discharged from HHS Care` flow variable. | Doc objective (primary) + column exists directly |
| "Estimate future imbalance between intake and exits" | **Intake/exit imbalance estimation**: a *derived* signal (documented explicitly as "Transfers − Discharges," i.e. `Children transferred out of CBP custody` minus `Children discharged from HHS Care`), tracked historically and projected forward from the two forecasts above — not a new raw target. | Doc's own "Flow-Based Signals" language in the methodology section |
| "Provide early warnings for healthcare planners" | **Capacity-stress / early-warning intelligence**: translate the forecasts and their uncertainty into a lead-time signal using relative (not officially-provided) thresholds. | Doc secondary objective; threshold gap noted in Finding #6 above |

This is fundamentally a **multi-target, multi-horizon time-series forecasting problem** with a derived-signal/early-warning layer on top. It is explicitly **not**: a classification problem, an anomaly-detection-as-primary-task problem, or a causal/policy-inference problem. The roadmap does not substitute any of those for the documented objective.

## 1.4 What We Have Been Given

Everything below is drawn from direct inspection, not the file's self-reported metadata alone.

| Aspect | What was found |
|---|---|
| **Official requirements** | Background/context, problem statement, 3 primary objectives, 3 secondary objectives, a 6-column dataset description, a step-by-step analytical methodology (time-series prep → feature engineering → train/test strategy → models → evaluation), 4–5 KPIs, Streamlit "Core Modules" (4) and "User Capabilities" (3), and 3 named deliverables (research paper, Streamlit dashboard, executive summary). |
| **Dataset (raw file)** | 1,170 rows × 6 columns, of which only **720 rows carry data** (see Finding #1). |
| **Time coverage** | **January 12, 2023 → December 21, 2025** (1,074 calendar days spanned; 720 actual observations). |
| **Granularity** | Irregular, ~5 observations/week, **predominantly Sunday–Thursday** (see Finding #2) — not calendar-daily, not Monday–Friday business-day. |
| **Target variable candidates** | `Children in HHS Care` (stock/level — Target 1) and `Children discharged from HHS Care` (flow — Target 2); see Part 3.3 for the formal target formulation. |
| **Potential predictors** | `Children apprehended and placed in CBP custody*`, `Children in CBP custody`, `Children transferred out of CBP custody` — all usable only in lagged/rolling form (see leakage notes, Part 3). |
| **Data limitations found** | 450 blank trailing rows; comma-formatted target column; irregular non-daily cadence with 49 gaps relative to the implied Sun–Thu schedule (including a 10-day gap at the series' start and a gap around Dec 25, 2023 consistent with a holiday); no capacity threshold; unresolved footnote asterisk; no exogenous variables (no policy-event markers, no geography, no age/gender breakdowns); flow columns don't exactly reconcile against the HHS Care stock (only ~2.5% exact match — see Part 3), meaning the 5 published columns do not fully explain day-to-day stock changes on their own. |
| **Missing information** | No explicit submission deadline on the instructions page itself (confirm via the portal's Calendar/Submit Project pages); no grading rubric visible in the extracted content; no explicit formula for any of the 4 named KPIs (names and one-line purposes only). |

## 1.5 What We Ultimately Have to Deliver

**A. Data/analytics layer** — A validated, cleaned, reproducible time-series dataset (720 real rows, correctly typed, correctly indexed on its true reporting schedule) plus a documented EDA.

**B. Forecasting/model layer** — For both Target 1 (HHS Care) and Target 2 (Discharged): 2 baseline models, 2 statistical models (Exponential Smoothing, ARIMA/SARIMA), 2 ML models (Random Forest, Gradient Boosting) — all six are named explicitly in the documentation, so all six are treated as required, not illustrative (see Part 2 and Part 15).

**C. Evaluation/validation layer** — A leakage-safe, walk-forward, multi-horizon (1/7/14-period) validation harness; a full model-comparison matrix; a documented, evidence-based champion-model selection per target.

**D. Decision-support/dashboard layer** — A Streamlit application covering the 4 documented core modules and 3 documented user capabilities, plus a small number of clearly justified supporting pages (Part 8).

**E. Deployment layer** — A publicly reachable, live-deployed instance of the dashboard (not a local-only prototype), reading from pre-generated forecast artifacts.

**F. Research/reporting layer** — A research paper (EDA, methodology, results, insights, recommendations) per the 25-section structure in Part 12, populated only with real, produced results.

**G. Final submission package** — Repository + deployed URL + research paper + executive summary, assembled and submitted through the Unified Mentor "Submit Project" flow.


---

# PART 2 — REQUIREMENTS DECOMPOSITION

Tags used: **[DOC]** = explicitly stated in official documentation · **[IMPLICIT]** = not written, but professionally necessary to satisfy a [DOC] requirement · **[ENG-REC]** = engineering recommendation filling a gap the sources leave open.

| # | Requirement | Source | Mandatory / Recommended | Implementation Implication | Final Evidence |
|---|---|---|---|---|---|
| 1 | Forecast children in HHS Care | [DOC] Objectives | Mandatory | Primary model Target 1, multi-horizon | Forecast chart + accuracy table for Target 1 |
| 2 | Predict short-term discharge demand | [DOC] Objectives | Mandatory | Primary model Target 2, multi-horizon | Forecast chart + accuracy table for Target 2 |
| 3 | Estimate future intake/exit imbalance | [DOC] Objectives | Mandatory | Derived signal = Transferred Out − Discharged, historical + near-term projected | Imbalance/pressure chart |
| 4 | Early warnings for healthcare planners | [DOC] Secondary Objectives | Mandatory | Early-warning module using relative thresholds (Finding #6) | Risk view + Surge Lead Time KPI |
| 5 | Quantify forecast uncertainty | [DOC] Secondary Objectives | Mandatory | Native CIs for statistical models; residual/quantile-based intervals for ML models | CI bands on every forecast chart + CI table in report |
| 6 | Compare statistical vs ML approaches | [DOC] Secondary Objectives | Mandatory | Both families implemented on the same validation harness | Model comparison table/page |
| 7 | Time-series preparation (datetime index, continuity, missing-day handling, decomposition) | [DOC] Methodology Step 1 | Mandatory | Must reconcile "daily" language with actual Sun–Thu cadence (Finding #2) — reindex to true schedule, not calendar week | Cleaned/reindexed dataset + decomposition plot |
| 8 | Feature engineering: lags (t-1/t-7/t-14), rolling 7/14-day mean+variance, flow signal, calendar effects | [DOC] Methodology Step 2 | Mandatory | Implement as named; resolve calendar-day-vs-row-based ambiguity using time-based rolling windows (Part 4) | Feature dictionary + processed feature tables |
| 9 | Train/test strategy: strict time-based split, no random sampling | [DOC] Methodology Step 3 | Mandatory | Chronological split only, justified explicitly given the regime shift (Finding #8) | Split-date documentation |
| 10 | Walk-forward validation | [DOC] Methodology Step 3 | Mandatory | Expanding-window, rolling-origin CV | Fold table + per-fold metrics |
| 11 | Multi-horizon evaluation | [DOC] Methodology Step 3 | Mandatory | Horizons = {1, 7, 14} periods ahead, inferred directly from the documented t-1/t-7/t-14 features | Horizon-specific error table |
| 12 | Baseline models: naive persistence, moving average | [DOC] Forecasting Models | Mandatory | Implemented first, before any complexity | Baseline metrics (the floor every other model must beat) |
| 13 | Statistical models: ARIMA/SARIMA, Exponential Smoothing | [DOC] Forecasting Models | Mandatory (both named explicitly) | Fit on cleaned/reindexed series; order selection evidence-based from ACF/PACF/stationarity tests | Model diagnostics + metrics |
| 14 | ML models: Random Forest Regressor, Gradient Boosting Regressor | [DOC] Forecasting Models | Mandatory (both named explicitly) | Fit on engineered features, leakage-checked | Metrics + feature importances |
| 15 | Metrics: MAE, RMSE, MAPE, horizon error | [DOC] Model Evaluation | Mandatory | MAPE flagged unstable for flow columns with true zero-days (Apprehended, Transferred Out, Discharged each have zero-days) → sMAPE/MASE reported alongside MAPE for those series [ENG-REC] | Metrics table (all 4 + safe alternative where flagged) |
| 16 | KPIs: Forecast Accuracy (%), Surge Lead Time, Capacity Breach Probability, Forecast Stability Index | [DOC] KPIs | Mandatory (4 clear items) | No formulas given in source → explicit formulas must be defined and documented [ENG-REC] (Part 7) | KPI computation appendix + dashboard cards |
| 16b | 5th KPI row ("Model robustness" / "Long-term reliability") | [DOC] KPIs — **malformed**, see Finding #5 | Flagged / P2 only | Do not invent a name; implement only if time allows, documented as ambiguous | Explicit note in report's Limitations section |
| 17 | Streamlit Core Modules: Future Care Load Forecast Chart, Discharge Demand Forecast Panel, Model Selection & Comparison, Confidence Interval Visualization | [DOC] Streamlit Requirements | Mandatory | One dashboard section per module, minimum | Live dashboard, 4 modules present |
| 18 | Streamlit User Capabilities: forecast horizon selector, model toggle, scenario comparison view | [DOC] Streamlit Requirements | Mandatory | Interactive controls bound to real precomputed forecast artifacts | Working controls in deployed app |
| 19 | Deliverables: research paper, Streamlit dashboard, executive summary | [DOC] Deliverables and Submission | Mandatory | 3 concrete, separately reviewable artifacts | Files/links in submission package |
| 20 | Deployment ("live analytics" language) | [DOC] implies via "live analytics" | Mandatory (technically necessary) | Must be hosted, not local-only | Public URL, smoke-tested |
| 21 | Submission via portal | [DOC] nav shows "Submit Project" | Mandatory | Package deliverables per portal's submission flow | Submission confirmation |
| 22 | Reproducible environment (requirements file, README, no hardcoded local paths) | [IMPLICIT] required to satisfy #20 credibly | Recommended (professional necessity) | `requirements.txt`, README with exact run steps | Clean-environment install succeeds |
| 23 | Data validation tests | [IMPLICIT] required to satisfy #7/#9 credibly | Recommended (professional necessity) | Schema/date/duplicate/continuity tests | Passing test suite |
| 24 | Version-controlled repository | [IMPLICIT] standard for "internship-quality submission" | Recommended (professional necessity) | GitHub repo, incremental commits | Repo link in submission |
| 25 | Config-driven code (no magic numbers/paths scattered in code) | [IMPLICIT] | Recommended (professional necessity) | Central `config.py` (Part 9) | Code review checklist item |
| 26 | Model/forecast artifact persistence (no retrain-on-page-load) | [IMPLICIT] required for a responsive, reliable dashboard | Recommended (professional necessity) | Batch-generate forecasts offline; dashboard reads flat files | Forecast files versioned alongside models |
| 27 | Documented assumptions/limitations | [IMPLICIT] given the 8 discrepancies found | Recommended (professional necessity) | Explicit Limitations section in report | Traceable to this roadmap's Findings table |

---

# PART 3 — DATASET UNDERSTANDING

*(Per the master instructions, this is a conceptual profile to inform the roadmap — not the full EDA, which is scheduled for Day 3.)*

## 3.1 Dataset Profile

| Property | Value | Notes |
|---|---|---|
| Rows (raw file) | 1,170 | Includes 450 fully blank trailing rows — **[FLAG #1]** |
| Rows (usable) | **720** | All 6 fields populated on every one of these rows |
| Columns | 6 | 1 date, 5 numeric |
| Date range | Jan 12, 2023 – Dec 21, 2025 | 1,074 calendar days spanned |
| Frequency | Irregular, ~5 obs/week | Predominantly Sun–Thu — **[FLAG #2]** |
| Day-of-week counts | Mon 145 · Tue 149 · Wed 147 · Thu 147 · Fri 2 · Sat 0 · Sun 130 | Sums to 720 |
| Missingness (within 720 real rows) | 0 cells | Fully populated once the blank tail is dropped |
| Missing relative to implied Sun–Thu schedule | 49 of 767 expected slots (≈6.4%) | Includes a 10-day gap at series start (Jan 12→22, 2023) and a gap around Dec 25, 2023 |
| Duplicate dates | 0 | |
| Duplicate full rows | 0 | |
| Data types (as shipped) | `Date`=text; 4 columns float64; `Children in HHS Care`=text | Float typing is an artifact of Finding #1; text typing of the target is Finding #3 |
| Sort order (raw file) | Descending by date | Must be re-sorted ascending for time-series work |
| Outliers (target, 3×IQR rule) | 0 flagged | No obviously corrupted extreme values |
| Zero values | Apprehended: 2 days · Transferred Out: 3 days · Discharged: 1 day · CBP Custody: 0 · HHS Care: 0 | Plausible operational zeros, not obviously errors — confirm in Day 3 EDA whether clustered around holidays |
| Negative values | 0 across all columns | Physically plausible throughout |

## 3.2 Variable Interpretation

| Column (exact name) | Represents | Operational meaning (per doc) | Analytical role | Usable for forecasting? | Leakage risk | Variable type |
|---|---|---|---|---|---|---|
| `Date` | Reporting date | — | Time index | N/A | None | Index |
| `Children apprehended and placed in CBP custody*` | New CBP intake | "Daily intake volume" | Predictor (upstream, leading) | Yes — lagged/rolling only | High if used at the *same* row as same-day target during backtesting without proper temporal shifting | Flow (inflow, upstream) |
| `Children in CBP custody` | CBP-side active caseload | "Active CBP care load" | Predictor (contextual stock) | Yes — lagged/rolling only | Same as above | Stock (CBP side) |
| `Children transferred out of CBP custody` | CBP→HHS transfer volume | "Flow into HHS system" | Predictor (primary driver of HHS Care) + component of the imbalance signal | Yes — lagged/rolling only | Same as above | Flow (inflow to HHS) |
| `Children in HHS Care` | Active HHS caseload | "Active HHS care load" | **Target 1** | — | — | Stock/level |
| `Children discharged from HHS Care` | Sponsor placements | "Successful sponsor placements" | **Target 2**; also usable as a *lagged* predictor of Target 1 | — | Leakage if used same-day (unlagged) as a predictor for Target 1 | Flow (outflow from HHS) |

Note on the flow columns: a simple accounting check (`HHS Care[t] ≈ HHS Care[t-1] + Transferred Out[t] − Discharged[t]`) only holds **exactly on 2.5% of rows** (median residual ≈ +6, but a heavy tail out to −657/+833). This means the four flow/stock predictor columns are **informative but not a complete explanation** of day-to-day changes in HHS Care — there are evidently other flows (e.g., reclassifications, age-outs, data revisions) not captured in this 6-column extract. This is useful to know for two reasons: (1) engineered "net flow" features are genuinely predictive, not spuriously perfect proxies for the target, so they carry real signal without being leakage by definition; (2) it tempers any expectation that a purely arithmetic/rules-based model could replace statistical/ML forecasting — it can't, the identity doesn't close cleanly enough for that.

Raw-level correlations are high across the board (e.g., Discharged–HHS Care r≈0.92, Apprehended–CBP Custody r≈0.95), but given the strong multi-year trend in the data (Finding #8), some of this is trend-inflated rather than short-term causal coupling. Day 3 EDA re-checks correlation on **differenced/detrended** series before treating any of these relationships as feature-engineering justification.

## 3.3 Target Formulation

- **Primary Target 1 — `Children in HHS Care`** (care-load / level forecasting). This is the headline objective ("forecast the number of children in HHS care") and the dataset supports it directly and cleanly once cleaned.
- **Primary Target 2 — `Children discharged from HHS Care`** (discharge-demand forecasting). The second explicit objective, again a direct column.
- **Derived signal — Intake/exit imbalance** (`Transferred Out − Discharged`). This is **not** a third independently-optimized forecasting target in the P0 scope. It's tracked historically and produced going forward by combining short-horizon forecasts/behavior of its two components — consistent with the documentation's own "Flow-Based Signals" framing, and consistent with "avoid overengineering": a third fully separate model family for a quantity that's algebraically composed of two things already being forecast would add cost without clear incremental value. A fully independent third model is listed as P2 (Part 15), not required.
- **`Children apprehended...`** and **`Children in CBP custody`** are **not** forecasting targets — they are not named as objectives anywhere in the documentation. They are used exclusively as engineered (lagged/rolling) predictors for Targets 1 and 2, per the documentation's own feature-engineering section.
- **Two separate model tracks are recommended** (one per primary target) sharing the same cleaning, feature-engineering, and validation harness — rather than a single combined multi-output model — because a stock variable (HHS Care) and a flow variable (Discharged) have different dynamics and scales, and the documentation itself treats them as two separate objectives, not one.

---

# PART 4 — ANALYTICAL ARCHITECTURE

**Pipeline:** Raw Data → Validation → Cleaning → Time-Series Construction → EDA → Feature Engineering → Baselines → Statistical Models → ML Models → Validation → Model Selection → Forecast Generation → Risk/Capacity Signals → Dashboard

| Stage | Purpose | Inputs | Outputs | Recommended technique | Why needed | Success measure |
|---|---|---|---|---|---|---|
| **Validation** | Confirm the file matches expectations before trusting it | Raw CSV | Pass/fail validation report | Row-count check, schema check, dtype check | Catches Finding #1 (blank rows) and Finding #3 (comma text) before they propagate | Zero unexpected schema deviations |
| **Cleaning** | Produce a trustworthy numeric dataset | Validated raw CSV | Clean DataFrame, 720 rows, correct dtypes | Truncate blank tail; strip commas; cast to int; parse `Date` with explicit format string; sort ascending | Every downstream step depends on this being correct | 0 nulls, 0 duplicate dates, 0 parse failures |
| **Time-Series Construction** | Establish the correct time index/frequency | Clean DataFrame | Indexed series on the **true Sun–Thu reporting schedule** (not a full 7-day calendar grid) | Build the expected-date index from the 5 observed weekdays; align real observations; explicitly mark the 49 missing expected slots; interpolate or mask only those (per documentation's own instruction), not a synthetic full week | Reconciles the doc's "handle missing days" instruction with the *actual* reporting cadence (Finding #2) — this is the single most consequential technical decision in the pipeline | Reindexed series has a known, documented count of real vs. imputed points |
| **EDA** | Understand trend/seasonality/stationarity before choosing models | Reindexed series | Findings doc + figures | Decomposition (trend/seasonal/residual), ACF/PACF, ADF/KPSS stationarity tests, weekly-pattern and regime-shift visualization | The 5.8× regime shift (Finding #8) must be characterized with statistics, not assumed away | Documented stationarity conclusion + seasonal period decision, each backed by a test statistic |
| **Feature Engineering** | Turn the raw series into model-ready predictors | Reindexed series + EDA conclusions | Feature tables (1 per target) | Lags t-1/t-7/t-14, rolling 7/14-day mean+variance (time-based, not row-based — see rationale below), net-flow signal, calendar features (day-of-week, month, and a simple holiday-proximity flag as an [ENG-REC] reading of "if available") | Directly implements the documented methodology step | Leakage test passes: no feature uses same-or-future information relative to its target row |
| **Baselines** | Establish the floor every later model must beat | Feature-free series | Naive persistence, seasonal-naive, moving-average forecasts + metrics | Simple last-value / same-weekday-last-week / trailing-mean rules | Documentation explicitly requires this before anything else, and it's the only honest way to know if complexity is earning its keep | Baseline MAE/RMSE/MAPE logged per target/horizon |
| **Statistical Models** | Capture trend/seasonality with classical time-series methods | Reindexed series | Exponential Smoothing + ARIMA/SARIMA forecasts, native CIs | Order selection from ACF/PACF + differencing informed by the stationarity tests | Named explicitly in the documentation | Beats baseline on held-out folds, or the gap is explained |
| **ML Models** | Capture nonlinear/interaction effects across engineered features | Feature tables | Random Forest + Gradient Boosting forecasts, residual-based intervals | Direct multi-horizon (one model per horizon) as the primary strategy | Named explicitly in the documentation | Beats baseline and is compared fairly against statistical models on the same folds |
| **Validation** | Produce a leakage-safe, apples-to-apples comparison | All fitted models | Walk-forward metrics matrix | Expanding-window rolling-origin CV, chronological only | Documentation explicitly requires this and explicitly rules out random splitting | No fold has training data later than its test data |
| **Model Selection** | Choose a champion, on evidence | Metrics matrix | Documented selection per target (and possibly per horizon) | Best aggregate error that also beats baseline and shows acceptable bias/stability | Prevents "most complex model wins by default" | Selection rationale cites specific metric-table rows |
| **Forecast Generation** | Produce the artifacts the dashboard will actually read | Champion model(s) | Flat forecast files (per target/horizon, with CIs) | One offline batch script, run on a schedule/on demand — not per dashboard page-load | Keeps the dashboard fast and guarantees dashboard/model consistency (Part 17 risk) | Forecast files match expected schema and date range |
| **Risk/Capacity Signals** | Translate forecasts into operational meaning | Forecast files | Early-warning flags, Surge Lead Time, Capacity Breach Probability (proxy) | Relative/statistical thresholds (Part 7), explicitly labeled as a proxy given Finding #6 | Satisfies the "early warning" objective without fabricating an absolute threshold | Backtested lead-time numbers, not asserted ones |
| **Dashboard** | Deliver decision support to the named stakeholders | All of the above | Deployed Streamlit app | Reads pre-generated artifacts only | Matches documented Streamlit requirements | 4 core modules + 3 user capabilities all functional |

### Why the reindexing decision matters (worked example)

The documentation's instruction to "ensure continuity of daily observations" and "handle missing days via interpolation or masking" is written in language that assumes a daily series. Two ways to satisfy that instruction were considered:

1. **Naive full calendar-day reindex** (7 days/week): would require synthesizing ~355 additional "days" (1,075 calendar days − 720 real ones) that were *never reported* — the majority of them Fridays and Saturdays, which the data shows are essentially never part of this program's reporting cadence at all. This would manufacture a large fraction of fake data.
2. **True-schedule reindex** (the 5 weekdays actually used, Sun–Thu): requires filling only the 49 slots that are genuinely missing *relative to the program's own real reporting pattern* — about 6.4% of the true schedule, concentrated around identifiable events (series start, a late-Dec-2023 gap consistent with a holiday).

Option 2 is adopted as the evidence-based choice. It satisfies the documentation's literal instruction ("handle missing days via interpolation or masking") while staying faithful to what the data actually is.

---

# PART 5 — FORECASTING STRATEGY

### Level 1 — Baselines

| Model | Definition | Assumptions | Features needed |
|---|---|---|---|
| Naive / persistence | Forecast = last observed value | No trend/seasonality captured | None |
| Seasonal naive | Forecast = value from the same weekday one reporting-week earlier (date-based lookup, not a fixed row offset, given the irregular cadence) | Weekly pattern is stable | Calendar alignment only |
| Moving average | Forecast = trailing mean over a defined window (e.g., time-based 7-calendar-day window) | Series is locally flat | Rolling window only |

### Level 2 — Statistical

| Model | When tested | Assumptions | Features needed | Notes |
|---|---|---|---|---|
| Exponential Smoothing (Holt / Holt-Winters) | After Level 1 is beaten by something, or as a required comparison point regardless (documentation names it explicitly) | Additive/multiplicative trend + seasonality; needs a defined, consistent period | Cleaned, reindexed series only | Fit on the true-schedule reindexed series (Part 4) so a consistent seasonal period can be defined |
| ARIMA / SARIMA | Same | Stationarity (after differencing); linear autocorrelation structure | Cleaned, reindexed series only | Order (p,d,q)(P,D,Q,m) selected from ACF/PACF + ADF/KPSS results from Day-3 EDA — **not** guessed. Given the confirmed regime shift (Finding #8), differencing (d≥1) is expected to be necessary; this is confirmed, not assumed, in EDA. |

### Level 3 — Machine Learning

| Model | When tested | Assumptions | Features needed | Notes |
|---|---|---|---|---|
| Random Forest Regressor | After the feature table exists (Day 4+) | Minimal distributional assumptions; can overfit with too many correlated lag/rolling features on ~700 rows | Full engineered feature table | Modest depth/estimator counts to control overfitting on a moderate-sized dataset |
| Gradient Boosting Regressor | Same | Same, plus sensitive to learning-rate/depth choices | Full engineered feature table | Light, time-budgeted tuning only — no exhaustive grid search (avoid overengineering) |

**Multi-step forecasting approach:** **Direct multi-horizon** (a separate model fit per horizon — 1, 7, 14 periods ahead) is the primary strategy, because it matches the documentation's explicit "multi-horizon evaluation" requirement cleanly and avoids compounding recursive error. Recursive one-step-ahead forecasting (feeding predictions back in as inputs) is noted as a viable alternative but is **not** required — P2 only, implemented if time allows and if it's shown to add value on the validation harness.

**Uncertainty representation:**
- Statistical models: native confidence intervals from the fitted model.
- ML models (Random Forest / Gradient Boosting): no native CIs. **[ENG-REC]** Use horizon-specific historical residual distributions from walk-forward validation to build empirical prediction intervals (e.g., residual quantiles added/subtracted from the point forecast). This is required to satisfy the documented "Confidence Interval Visualization" module for the ML track, not optional polish.

**Explicit non-assumption:** No model is assumed to be "the best" in advance. Complexity is only kept if Part 6's validation evidence shows it earns its place over the simpler alternative — this applies level-by-level (baseline → statistical → ML), and also across targets/horizons independently, since the winning model may differ between Target 1 and Target 2, or between a 1-period and a 14-period horizon.

---

# PART 6 — VALIDATION AND MODEL SELECTION

### Why random train/test splitting is inappropriate here

Two independent, evidence-based reasons, both grounded in this specific dataset:

1. **Temporal leakage.** A forecasting model's real job is to predict the future from the past. A random split lets the model train on dates that fall *after* some of its test dates, which no real deployment could ever do — it silently overstates accuracy.
2. **The regime shift makes this worse, not just theoretically wrong.** Because `Children in HHS Care` falls ~5.8× from its Dec-2023 peak to its Aug-2025 trough (Finding #8), a random split would scatter both high-load (2023) and low-load (2025) observations across *both* train and test sets. The model would effectively be tested on regimes it has already "seen" in training, which is exactly the scenario a real deployment will never get — the real task is forecasting a regime the model has only partially observed. A chronological split is the only way to measure something honest here.

### Validation design

| Element | Definition |
|---|---|
| **Split type** | Strict chronological split. No shuffling, no random sampling, at any stage. |
| **Final held-out test window** | The most recent portion of the series (exact cut date finalized after Day-3 EDA, sized to cover multiple full weekly cycles across all three horizons — an **[ENG-REC]** default, not a documentation-specified number) |
| **Walk-forward validation** | Expanding-window, rolling-origin: the training window grows forward in time across multiple folds, with the model re-evaluated (and, where feasible, refit) at each new origin |
| **Forecast horizons** | **1, 7, and 14 periods ahead** — inferred directly from the documentation's own t-1/t-7/t-14 feature language, adopted consistently as the evaluation grid |
| **Evaluation windows** | Folds span **both** the high-load (2023) and low-load (2025) portions of history, so performance is checked in both regimes, not just the most recent (easier) one |
| **Baseline comparison** | Every statistical/ML model must beat the naive and seasonal-naive baselines on held-out folds to be considered viable — a simple "skill vs. naive" gate before anything is called an improvement |
| **Model comparison** | All models (2 baselines + 2 statistical + 2 ML) evaluated on the *same* folds and horizons, per target, so results are directly comparable |
| **Error analysis** | Per-horizon breakdown, bias (signed mean error), forecast stability across refits, and performance specifically during the top-quartile-load period of history (an evidence-based stand-in for "surge periods," since no official surge definition exists) |

### Metrics

| Metric | Purpose | Note for this dataset |
|---|---|---|
| MAE | Absolute forecast accuracy | Reported for every model/target/horizon |
| RMSE | Penalizes large errors | Reported for every model/target/horizon |
| MAPE | Relative error understanding | Safe for Target 1 (HHS Care never reaches 0, min 1,972). **Flagged unstable** for series with true zero-days (Apprehended: 2, Transferred Out: 3, Discharged: 1) — sMAPE or MASE reported alongside MAPE wherever a series has zero-days, per the documentation's own allowance for "an appropriate alternative when MAPE is problematic" |
| Horizon-specific error | Short vs. medium-term reliability | Broken out at h=1, h=7, h=14 separately, never only as one blended number |
| Forecast stability | Model robustness | **[ENG-REC]** defined as the variance of forecasts made for the same target date from different, rolling forecast origins — low variance = stable |
| Bias | Systematic over/under-forecasting | Signed mean error, checked separately for the high-load and low-load regimes |
| Surge/high-load performance | Whether the model holds up when it matters most | Evaluated specifically on the top-quartile-load segment of history (evidence-based proxy, since no official "surge" definition exists in either source) |

### How the winning model is selected

For each target (and, where results diverge meaningfully, for each horizon): the model with the best aggregate error across walk-forward folds **that also** (a) clears the baseline-beating gate, (b) shows acceptable bias (no large systematic over/under-forecast), and (c) shows acceptable stability, is selected as champion. The selection is written up with the specific metric-table numbers that justified it — never asserted without that evidence trail, and never defaulting to "the most complex model" without those numbers behind it.

---

# PART 7 — EARLY-WARNING / OPERATIONAL INTELLIGENCE

| Signal | Derivation |
|---|---|
| Expected future care load | Target 1 point forecast + CI, at each horizon |
| Expected discharge demand | Target 2 point forecast + CI, at each horizon |
| Intake vs. discharge pressure | Forward-looking view of (forecasted/near-term Transferred Out) − (forecasted/near-term Discharged) — the same "net pressure" definition used in feature engineering, projected forward instead of only computed historically |
| Confidence/uncertainty | Interval width at each horizon, shown numerically and as a visual band |

### Capacity-stress signal — handling the missing threshold honestly

**No numeric capacity threshold exists in either source (Finding #6).** Inventing one (e.g., "shelter capacity is 8,000 children") would be a fabricated requirement and is explicitly disallowed by the master brief. Instead:

- **[ENG-REC] Relative/statistical proxy thresholds**, computed from the data's own history rather than an external number:
  - *Percentile-based:* flag "elevated" when a forecast exceeds, e.g., the trailing-window 90th percentile of recent observed load; flag "high" beyond a wider band. Exact percentile cut-points are tuned during backtesting, not fixed in advance.
  - *Rate-of-change-based:* flag when the forecasted increase over the horizon exceeds a data-derived threshold (e.g., a multiple of the series' typical period-over-period volatility).
- **Every place this proxy appears** — dashboard, report, executive summary — states plainly that it is a **relative, data-derived proxy standing in for an absent official capacity figure**, not a validated operational threshold. This is a hard requirement of this roadmap, not a stylistic suggestion: presenting the proxy as if it were an authoritative number would misrepresent the analysis to the named stakeholders.
- **Surge Lead Time (KPI):** defined via backtesting — the number of periods between when the early-warning signal first fires and when the actual series subsequently crosses that same relative threshold. This number must come from running the backtest, not from an assumed target.

---

# PART 8 — STREAMLIT PRODUCT DESIGN

Every page below is tagged **[DOC]** (directly required by the documented Core Modules / User Capabilities) or **[REC]** (not named explicitly, but needed for the [DOC] pages to make sense — e.g., you can't have a "Model Selection & Comparison" module without somewhere the raw historical data and methodology are explained). No page is included purely for visual padding; the total page count is deliberately capped at 8 to respect the 15-day budget and the "avoid overengineering" instruction.

| # | Page | Tag | Purpose | Main visualization | KPI cards | User controls | Interpretation focus |
|---|---|---|---|---|---|---|---|
| 1 | **Executive Overview** | [REC] | Single-screen orientation for a non-technical visitor | Small multiples: current HHS Care + Discharged trend (last ~90 days) | Forecast Accuracy (%), Surge Lead Time, Capacity Breach Probability (proxy), Forecast Stability Index | Date-range quick filters | "Where do things stand, and how much can we trust the forecast" |
| 2 | **Historical Trends** | [REC] | Full context for the regime shift (Finding #8) and the reporting cadence (Finding #2) | Full-history line charts, all 5 variables; visible peak (Dec 2023) → trough (Aug 2025) → mild rebound | — | Date range, variable picker | "Why the series looks the way it does" |
| 3 | **Care Load Forecast** | **[DOC]** — "Future Care Load Forecast Chart" | Primary forecasting deliverable for Target 1 | Forecast line + historical tail + CI band | Point forecast, CI width, model used | **Horizon selector, model toggle** | "What's expected, and how confident are we" |
| 4 | **Discharge Demand Forecast** | **[DOC]** — "Discharge Demand Forecast Panel" | Primary forecasting deliverable for Target 2 | Same pattern as page 3, for Discharged | Point forecast, CI width, model used | **Horizon selector, model toggle** | "Will discharge throughput keep pace" |
| 5 | **Intake vs. Exit Pressure (Early Warning)** | **[DOC]** — objectives (imbalance) + secondary (early warning) | Operational risk view | Net-flow (Transferred Out − Discharged) history + near-term projection, relative threshold bands overlaid | Surge Lead Time, current pressure state | Threshold sensitivity (percentile) slider | "Is pressure building, and how much lead time do we have" — with the proxy-threshold caveat always visible |
| 6 | **Model Comparison & Accuracy** | **[DOC]** — "Model Selection & Comparison" + "compare statistical vs ML" | Evidence for model selection | Metrics table (MAE/RMSE/MAPE or sMAPE, per model/horizon/target) + baseline-beaten indicator | Champion model per target | Target picker, horizon picker | "Which model wins, and why" |
| 7 | **Scenario Comparison** | **[DOC]** — "Scenario comparison view" | Side-by-side comparison | Overlay of 2+ models and/or 2+ horizons on one chart | — | Multi-select models/horizons | "How much do model choice and horizon choice actually matter" |
| 8 | **Methodology / Data Information** | [REC] | Transparency and traceability | Data dictionary table + pipeline diagram (static) | — | — | Documents every limitation from the Source Verification Note (450 blank rows, Sun–Thu cadence, no threshold, KPI ambiguity, asterisk gap) directly in-product, not just in the paper |

**Confidence Interval Visualization** (a named Core Module) is satisfied by the CI bands built into pages 3 and 4, plus an explicit CI-width readout — it is not a separate page, since a standalone "CI page" with nothing else on it would be padding.

**Mapping check — 100% documented-module coverage:**
- Future Care Load Forecast Chart → Page 3
- Discharge Demand Forecast Panel → Page 4
- Model Selection & Comparison → Page 6
- Confidence Interval Visualization → Pages 3 & 4 (built-in)
- Forecast horizon selector → Pages 3, 4, 7
- Model toggle → Pages 3, 4, 6, 7
- Scenario comparison view → Page 7

Every dashboard page reads exclusively from **pre-generated forecast/metrics artifacts** (Part 9) — no model is trained live inside a page callback. This is a deliberate architectural constraint, not a simplification for its own sake: it keeps the app fast, keeps behavior identical between local and deployed environments, and removes an entire class of "dashboard shows something the model never actually produced" failure (Part 17).

---

# PART 9 — PROFESSIONAL SOFTWARE ARCHITECTURE

### Directory structure

```
uac-forecasting/
├── data/
│   ├── raw/                      # original CSV, untouched, read-only
│   ├── interim/                  # cleaned + reindexed series (Part 4 output)
│   └── processed/                # per-target engineered feature tables
├── notebooks/                    # exploration only — nothing here is imported by the app
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_statistical_models.ipynb
│   ├── 06_ml_models.ipynb
│   └── 07_validation_and_selection.ipynb
├── src/                          # reusable, production-facing logic
│   ├── config.py                 # paths, horizons, split dates, threshold params — single source of truth
│   ├── data/
│   │   ├── load.py
│   │   ├── clean.py
│   │   └── validate.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── statistical.py
│   │   ├── ml.py
│   │   └── registry.py           # records which model wins per target/horizon, and why
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── signals/
│   │   └── early_warning.py
│   └── forecast/
│       └── generate.py           # the one offline batch script that produces dashboard-ready artifacts
├── models/                       # serialized trained artifacts (per target/horizon)
├── forecasts/                    # generated forecast + metrics flat files consumed by the dashboard
├── app/
│   ├── Home.py
│   └── pages/                    # one file per page from Part 8
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_metrics.py
├── reports/
│   ├── research_paper.md
│   └── executive_summary.md
├── requirements.txt
├── README.md
└── .streamlit/config.toml
```

### Layer responsibilities

| Layer | Lives in | Responsibility |
|---|---|---|
| Data | `data/`, `src/data/` | Ingestion, cleaning, validation, reindexing — everything from Part 4's Validation/Cleaning/Time-Series Construction stages |
| Configuration | `src/config.py` | Every path, split date, horizon list, threshold parameter, and model hyperparameter set lives here — nothing hardcoded elsewhere |
| EDA | `notebooks/02_eda.ipynb` | Exploration only; conclusions get written into `src/` decisions (e.g., differencing order), not left stranded in a notebook |
| Feature engineering | `src/features/` | Implements Part 4/5's lag, rolling, flow, and calendar features with leakage-safe shifting |
| Modeling | `src/models/` | Baselines, statistical, ML — one file per family, common interface so the validation harness can loop over all of them identically |
| Evaluation | `src/evaluation/` | Walk-forward harness + metric functions, unit-tested |
| Forecast generation | `src/forecast/generate.py` | The single offline entry point that produces everything `forecasts/` contains |
| Visualization/App | `app/` | Reads only from `forecasts/` and `models/` metadata — contains no training or cleaning logic |
| Tests | `tests/` | Data, feature, and metric correctness (Part 11) |
| Documentation | `README.md`, `reports/` | Setup instructions, architecture summary, research paper, executive summary |
| Deployment configuration | `.streamlit/config.toml`, `requirements.txt` | Everything needed for a clean-environment deploy (Part 10) |

### What must NOT be hardcoded

- File paths (always resolved relative to a project root defined once in `config.py`)
- Train/test split dates and walk-forward fold boundaries
- The horizon list ({1, 7, 14})
- Relative-threshold percentiles used for the early-warning proxy
- Column name strings repeated across files (define once, import everywhere)
- Model hyperparameters (kept in a small, explicit params structure, not scattered inline across notebooks and scripts)

### How trained artifacts, config, and forecasts are handled

- Fitted models (and any fitted preprocessing, e.g. scalers) are persisted to `models/` via a standard serialization format, named clearly per target and horizon.
- `src/forecast/generate.py` is the **only** place forecasts get produced. It reads the champion models from the registry, generates forecasts + intervals for every target/horizon, and writes flat files to `forecasts/`.
- The Streamlit app **never** trains or refits a model as part of serving a page. It reads `forecasts/` and `models/` metadata only. An optional "regenerate forecast" action can exist as an explicit, clearly-labeled advanced control — never the default load path — if time allows (P2).
- This separation is what makes the "dashboard becomes disconnected from the actual model" risk (Part 17) detectable and preventable: there is exactly one producer of forecast artifacts, and the app is a pure consumer.

---

# PART 10 — DEPLOYMENT PLAN

The simplest professional option that satisfies "live analytics" within the 15-day budget is used — no infrastructure is added that the project doesn't need.

| Concern | Plan | Tag |
|---|---|---|
| **Application hosting** | Streamlit Community Cloud, connected directly to the GitHub repository | **[ENG-REC]** — docs require "live" dashboard but don't name a host |
| **Dependency management** | `requirements.txt` pinned to the exact versions used in development; kept minimal (pandas, numpy, statsmodels, scikit-learn, streamlit, plotting library — no deep-learning framework, since none is required) | |
| **Environment configuration** | No secrets/API keys are required for this project (public dataset, no external services) — this materially simplifies deployment and is worth stating explicitly rather than adding unnecessary `secrets.toml` machinery | |
| **Model artifact handling** | Small serialized models + precomputed forecast files committed directly into the repository — the dataset (720 rows) and resulting artifacts are small enough that no external object storage is needed | |
| **Dataset handling** | Raw CSV kept as the single committed source of truth; interim/processed files are regenerated by a reproducible script rather than hand-edited | |
| **Reproducibility** | README documents: clone → create environment → `pip install -r requirements.txt` → run the artifact-generation script → `streamlit run app/Home.py`; Python version pinned | |
| **Production smoke testing** | After every deploy, manually walk every page and control once against the Part 11 Application Tests checklist | |
| **Public/demo URL** | Included in the submission package and the README | |
| **README instructions** | Setup, run-locally steps, architecture summary, and — importantly — the same limitations/assumptions log carried through this whole roadmap (450 blank rows, Sun–Thu cadence, no official capacity threshold, KPI table ambiguity, unresolved footnote), so a reviewer sees them without having to ask | |

The goal, per the master brief, is a **working deployed application** — the Day 12 checkpoint (Part 14) exists specifically so deployment isn't attempted for the first time in the project's final hours.

---

# PART 11 — TESTING AND QUALITY ASSURANCE

### Data tests

| Test | What it checks | Tied to |
|---|---|---|
| Schema | Exactly 6 columns, expected names/order | Baseline structural check |
| Row-count truncation | File loads to exactly 720 usable rows, blank tail dropped | Finding #1 |
| Date parsing | 100% of `Date` values parse with the `Month DD, YYYY` format, 0 failures | Verified during audit; must stay 0 after any pipeline change |
| Missing values | 0 nulls in the 720 real rows post-cleaning | Baseline data-quality gate |
| Duplicates | 0 duplicate dates, 0 duplicate full rows | Verified during audit |
| Numeric validity | No negative values in any of the 5 numeric columns; `Children in HHS Care` casts cleanly after comma-stripping | Finding #3 |
| Time continuity | Reindexed series matches the expected Sun–Thu schedule count, with exactly the 49 known gaps flagged (not silently dropped or silently invented) | Finding #2 |

### ML tests

| Test | What it checks |
|---|---|
| Leakage | Every engineered feature's timestamp is strictly earlier than its target row's timestamp — enforced as an automated assertion, not a manual spot-check |
| Feature correctness | A handful of engineered rows hand-verified against the raw series (e.g., a known lag-7 value matches the actual observation from 7 periods prior) |
| Train/test separation | For every walk-forward fold, `max(train dates) < min(test dates)`, asserted programmatically |
| Forecast horizon correctness | Generated forecast dates are exactly h periods ahead of their origin, per the defined Sun–Thu schedule — not off-by-one, not calendar-day when it should be schedule-day |
| Metric correctness | MAE/RMSE/MAPE/sMAPE functions unit-tested against hand-computed toy examples with known answers |

### Application tests

- Dashboard loads without error, end to end
- Every page's forecast/metrics data renders (non-empty) on first load
- Model selector changes the displayed chart/metrics correctly, for every option
- Horizon selector changes the displayed chart/metrics correctly, for every option
- All charts render with valid, non-empty data — no blank chart states
- No broken interaction across any control combination exercised during manual QA
- Edge cases explicitly checked: shortest horizon, longest horizon, every model option, every page

### Deployment tests

- Clean-environment install (fresh virtual environment) completes without manual intervention
- `streamlit run app/Home.py` starts successfully in that clean environment
- No local-only dependencies (absolute local paths, machine-specific packages) remain anywhere in the codebase
- Data/model paths resolve correctly relative to the repository root in the deployed environment, not just on the development machine

---

# PART 12 — FINAL REPORT / RESEARCH PAPER STRUCTURE

Every section below is populated **only** with evidence actually produced during Days 1–13 of execution. No numbers are pre-filled here.

| # | Section | What must appear | Evidence source |
|---|---|---|---|
| 1 | Title | Project title, author, date, program (Unified Mentor) | Fixed |
| 2 | Executive Summary | 1-page distillation of Part 13 below | Written last, from final real results |
| 3 | Background | Restated UAC/HHS operating context (Part 1.2) | Documentation |
| 4 | Problem Statement | Restated descriptive-vs-predictive gap (Part 1.2) | Documentation |
| 5 | Objectives | The 3 primary + 3 secondary objectives, verbatim in meaning (Part 1.3) | Documentation |
| 6 | Dataset | Row/column counts (raw vs. usable), date range, cadence, the 8 findings from the Source Verification Note | This roadmap's audit |
| 7 | Data Preparation | Cleaning steps performed (truncation, comma-stripping, dtype casting, reindexing to the true schedule), with before/after row counts | Day 2 output |
| 8 | Exploratory Analysis | Decomposition, ACF/PACF, stationarity test results, weekly-pattern and regime-shift figures | Day 3 output |
| 9 | Methodology | The full pipeline from Part 4, as actually executed | Days 2–9 |
| 10 | Feature Engineering | Final feature list per target, with the calendar-day-vs-row-based lag/rolling decision explained | Day 4 output |
| 11 | Forecasting Models | All 6 models (2 baseline, 2 statistical, 2 ML) as actually configured/fit | Days 5–7 |
| 12 | Experimental Design | Split dates, fold structure, horizons — as actually used | Day 5 output |
| 13 | Validation Strategy | Walk-forward mechanics, why random splitting was rejected (Part 6), as actually implemented | Day 5 output |
| 14 | Results | Full metrics tables per model/target/horizon — real numbers only | Days 6–8 |
| 15 | Model Comparison | Statistical vs. ML comparison, baseline-beaten status, champion selection rationale | Day 8 output |
| 16 | Forecast Analysis | What the champion models actually forecast going forward, with CIs | Day 9 output |
| 17 | Early-Warning / Operational Insights | Surge Lead Time and Capacity Breach Probability results from backtesting, with the proxy-threshold caveat stated explicitly | Day 9 output |
| 18 | Dashboard | Screenshots/description of the 8 pages, mapped to documented requirements (Part 8's mapping table) | Days 10–11 |
| 19 | Deployment | Architecture summary, hosting choice and rationale, deployed URL | Day 12 output |
| 20 | Limitations | All 8 findings from the Source Verification Note, plus any new ones surfaced during execution | Ongoing log |
| 21 | Recommendations | Operationally grounded suggestions for HHS-style stakeholders, tied to what the models actually showed | Derived from Days 6–9 results |
| 22 | Future Work | Explicitly the P2 items from Part 15 that were deferred, and why | Part 15 |
| 23 | Conclusion | Restates whether the objectives (Part 1.3) were met, with evidence | Final synthesis |
| 24 | References | Documentation source, dataset source (the Google Drive link named in the portal, and the CSV as authoritative), any libraries/methods cited | Source Verification Note |
| 25 | Appendix | Full metrics tables, data dictionary, config values used | All prior days |

**Hard rule carried through every section above:** no accuracy figure, KPI value, lead-time number, or "the model achieved X%" claim is written until it has actually been produced by the Day 5–9 implementation. Placeholder structure is fine to draft early (Day 1 can stub the section headers); populated numbers are not.

---

# PART 13 — EXECUTIVE SUMMARY (SPEC)

Written **last**, after real results exist (Day 14), for a non-technical HHS-style operational stakeholder. One page. No equations, no library names, no code. It must answer, in this order, mirroring the documentation's own framing:

1. **What problem exists** — descriptive-only reporting leaves capacity planning reactive (Part 1.2), stated in one or two sentences.
2. **What was built** — a forecasting system covering care load and discharge demand, with an early-warning view, delivered as a live dashboard.
3. **What data was used** — the HHS UAC program's own published-style figures, ~3 years, ~720 reporting days (plain-language version of Part 3.1) — including a one-line, non-alarming mention that the raw file needed cleaning (blank rows, formatting) before use, for transparency.
4. **What the forecasting system achieved** — real accuracy figures from Day 6–8, stated in plain terms (e.g., "typically within X children of the actual count Y periods ahead") — **filled in only from actual results**.
5. **How early it can identify pressure/surges** — the real, backtested Surge Lead Time number, with the proxy-threshold caveat stated in one plain sentence.
6. **What operational value it provides** — tied directly to the 3 original decision questions from Part 1.2 (how many children, will discharge keep pace, when to scale up).
7. **Major limitations** — plain-language versions of the Source Verification Note findings that matter operationally (no official capacity number was available; the data has some gaps and formatting quirks that were cleaned; forecasts are less certain further into the future).
8. **How stakeholders should use it** — as a planning input alongside existing judgment, not as an unattended automated trigger — especially given the proxy nature of the capacity signal.

Tone: decision-oriented, plain language, honest about uncertainty and limitations — never overstating what a ~720-row dataset and a 15-day solo build can support.

---

# PART 14 — EXACT 15-DAY EXECUTION ROADMAP

Two buffer touchpoints are built in (Day 8, Day 14) rather than left to the end. Testing and deployment happen on Day 12 — not in the final hours. Documentation is spread across Days 13–14, not compressed into Day 15. Day 15 is reserved for QA and submission only, deliberately kept light.

### Day 1 — Project Understanding & Source Audit
**Primary objective:** Lock down a shared, evidence-based understanding of both sources before writing any pipeline code.
**Tasks:**
- Re-read the official documentation end to end; extract the requirements matrix (Part 2)
- Run the full data audit reproduced in this roadmap's Source Verification Note (row counts, dtypes, date range, day-of-week cadence, missingness, duplicates, comma formatting, flow-identity check)
- Log every discrepancy found (the 8 findings above, plus any new ones)
- Set up the repository skeleton (Part 9 directory tree), git init, virtual environment, initial `requirements.txt`
**Expected output:** A written requirements matrix and discrepancy log matching Parts 1–3 of this roadmap.
**Files/artifacts created:** `README.md` (stub), `requirements.txt` (stub), `docs/requirements_matrix.md`, `docs/discrepancy_log.md`, empty repo skeleton
**Validation checkpoint:** Can reload the CSV and independently reproduce: 720 real rows, Sun–Thu-dominant cadence, comma-formatted target column.
**Definition of done:** Both sources are fully understood and logged; no open questions about what the project is asking for remain.

### Day 2 — Data Cleaning & Validated Master Series
**Primary objective:** Produce the one trustworthy dataset every later step depends on.
**Tasks:**
- Truncate the 450 blank trailing rows
- Parse `Date` with the correct explicit format
- Strip thousands-separators from `Children in HHS Care` and cast all 5 numeric columns to integer
- Sort ascending; build the true Sun–Thu expected-schedule index; align real observations to it; explicitly flag the 49 missing expected slots (do not silently interpolate without marking which points are real vs. imputed)
- Write the Part 11 data tests
**Expected output:** Cleaned, correctly-typed, correctly-indexed master series with an explicit `is_imputed` flag column.
**Files/artifacts created:** `src/data/load.py`, `clean.py`, `validate.py`; `data/interim/master_series.parquet`; `tests/test_data.py`
**Validation checkpoint:** All Part 11 data tests pass; row count and gap count match the audit numbers exactly.
**Definition of done:** A single command reproduces the clean dataset from the raw CSV, with tests green.

### Day 3 — Exploratory Data Analysis
**Primary objective:** Understand trend, seasonality, and stationarity before any model is chosen.
**Tasks:**
- Seasonal decomposition (trend/seasonal/residual) on the reindexed series
- ACF/PACF plots; ADF and KPSS stationarity tests, given the confirmed regime shift
- Visualize the weekly reporting pattern and the Dec-2023-peak → Aug-2025-trough → mild-rebound shape
- Review the zero-value days (Apprehended, Transferred Out, Discharged) for any holiday clustering
- Re-check correlations on differenced series, not just raw levels
**Expected output:** A findings document that will directly drive Day 6's SARIMA order selection and Day 4's rolling-window design.
**Files/artifacts created:** `notebooks/02_eda.ipynb`, `reports/figures/*`, `docs/eda_findings.md`
**Validation checkpoint:** Stationarity conclusion is backed by an actual test statistic, not a visual guess; seasonal period decision is written down explicitly.
**Definition of done:** Every modeling decision in Parts 5–6 that depends on EDA now has a concrete answer instead of a placeholder.

### Day 4 — Feature Engineering
**Primary objective:** Turn the cleaned series into leakage-safe, model-ready feature tables.
**Tasks:**
- Implement lags (t-1, t-7, t-14) and rolling 7/14-period mean+variance using **time-based** windows (not naive row-count windows), per the Part 4 rationale
- Implement the net-flow (Transferred Out − Discharged) signal
- Implement calendar features (day-of-week, month, a simple holiday-proximity flag as the documented "if available" allowance)
- Build separate feature tables for Target 1 and Target 2, each with strictly backward-looking predictors only
- Write and run the leakage test
**Expected output:** Two model-ready feature tables.
**Files/artifacts created:** `src/features/build_features.py`, `data/processed/features_target1.parquet`, `features_target2.parquet`, `tests/test_features.py`
**Validation checkpoint:** Leakage test passes; row counts reconcile after the initial rows lost to the longest lag window are accounted for.
**Definition of done:** Feature dictionary documented; both tables ready to feed models.

### Day 5 — Baselines & Walk-Forward Harness
**Primary objective:** Establish the floor every later model must beat, and build the one validation harness every subsequent day will reuse.
**Tasks:**
- Implement naive persistence, seasonal-naive, and moving-average baselines for both targets
- Build the expanding-window, rolling-origin walk-forward harness with the {1, 7, 14} horizon grid
- Implement and unit-test MAE/RMSE/MAPE/sMAPE
- Run baselines through the harness
**Expected output:** Baseline metrics matrix (per target, per horizon) and a reusable harness.
**Files/artifacts created:** `src/models/baselines.py`, `src/evaluation/walk_forward.py`, `metrics.py`, `tests/test_metrics.py`
**Validation checkpoint:** Harness produces strictly chronological folds (assert `max(train) < min(test)` on every fold); baseline numbers are sane on visual spot-check.
**Definition of done:** A logged baseline metrics table exists that every later model will be compared against.

### Day 6 — Statistical Models
**Primary objective:** Fit and evaluate Exponential Smoothing and ARIMA/SARIMA on both targets.
**Tasks:**
- Select ARIMA/SARIMA orders using Day 3's ACF/PACF and stationarity results (differencing expected, given the confirmed regime shift)
- Fit Exponential Smoothing and ARIMA/SARIMA per target
- Generate multi-horizon forecasts with native confidence intervals
- Run both through the Day 5 harness for direct comparability with baselines
**Expected output:** Statistical-model metrics matrix, comparable to baselines.
**Files/artifacts created:** `src/models/statistical.py`, `models/stat_target1_*.pkl`, `models/stat_target2_*.pkl`, `notebooks/05_statistical_models.ipynb`
**Validation checkpoint:** Statistical-vs-baseline comparison table complete for both targets, all horizons — whichever direction the evidence points.
**Definition of done:** No open modeling questions remain about the statistical family; results are logged either way.

### Day 7 — Machine Learning Models
**Primary objective:** Fit and evaluate Random Forest and Gradient Boosting on both targets.
**Tasks:**
- Fit Random Forest and Gradient Boosting per target, using the Day 4 feature tables
- Use direct multi-horizon modeling (one model per horizon) as the primary strategy
- Build horizon-specific residual-based prediction intervals (Part 5) since these models have no native CIs
- Run both through the Day 5 harness
- Keep tuning light and time-boxed — no exhaustive search
**Expected output:** ML-model metrics matrix, comparable to both baselines and statistical models.
**Files/artifacts created:** `src/models/ml.py`, `models/ml_target*_h*.pkl`, `notebooks/06_ml_models.ipynb`
**Validation checkpoint:** Full 6-model comparison matrix (2 baseline + 2 statistical + 2 ML) exists for both targets, all 3 horizons.
**Definition of done:** Every model named in the documentation has been fit, validated, and logged — the complete evidence base for model selection now exists.

### Day 8 — Model Selection + Early-Warning Design (Buffer Day #1)
**Primary objective:** Convert the Day 5–7 evidence into a formal champion-model decision, and start the early-warning layer. Also the first scheduled buffer if any prior day slipped.
**Tasks:**
- Apply the Part 6 selection rule (beats baseline, best aggregate error, acceptable bias/stability, checked in both regimes) per target/horizon
- Write the selection rationale, citing specific metric-table numbers
- Begin the early-warning signal design: relative/statistical threshold proxy (Part 7), explicitly labeled as a proxy
- If Days 1–7 slipped, use the slack in this day to catch up before continuing
**Expected output:** A documented model registry and a first working version of the early-warning logic.
**Files/artifacts created:** `src/models/registry.py`, `docs/model_selection_rationale.md`, `src/signals/early_warning.py` (first pass)
**Validation checkpoint:** Every selection claim traces to a specific row in the Day 5–7 metrics tables.
**Definition of done:** The forecasting layer is "frozen" — no more open modeling decisions before the dashboard is built.

### Day 9 — Batch Forecast Generation & Early-Warning Finalization
**Primary objective:** Produce every artifact the dashboard will read, so no model ever trains inside a page.
**Tasks:**
- Build `src/forecast/generate.py`, the single offline script producing all forecast + CI + metrics flat files
- Finalize Surge Lead Time, Capacity Breach Probability (proxy), and Forecast Stability Index computations, with explicit documented formulas (Part 7)
- Run the pipeline end to end once
**Expected output:** Complete `forecasts/` directory ready for the dashboard.
**Files/artifacts created:** `src/forecast/generate.py`, `src/signals/early_warning.py` (final), `forecasts/*`
**Validation checkpoint:** Forecast files match the expected schema and date ranges; all KPIs compute without error.
**Definition of done:** The dashboard build (Days 10–11) requires zero live model training — everything it needs already exists on disk.

### Day 10 — Streamlit: Core Forecast Pages
**Primary objective:** Build the two explicitly mandatory forecast pages.
**Tasks:**
- Build Page 3 (Care Load Forecast) and Page 4 (Discharge Demand Forecast) from Part 8, each with horizon selector, model toggle, and CI band
**Expected output:** Two fully working dashboard pages, running locally.
**Files/artifacts created:** `app/Home.py` (skeleton), `app/pages/2_Care_Load_Forecast.py`, `app/pages/3_Discharge_Demand_Forecast.py`
**Validation checkpoint:** Manual click-through confirms every control changes the chart correctly.
**Definition of done:** Both documented Core Modules for forecasting are functional locally.

### Day 11 — Streamlit: Remaining Pages
**Primary objective:** Complete the full 8-page application.
**Tasks:**
- Build Executive Overview, Historical Trends, Intake vs. Exit Pressure, Model Comparison, Scenario Comparison, and Methodology pages
- Wire KPI cards on the Overview page to the real, computed KPI values
**Expected output:** Full 8-page app complete and internally consistent.
**Files/artifacts created:** Remaining `app/pages/*.py`
**Validation checkpoint:** Full manual pass against the Part 11 Application Tests checklist.
**Definition of done:** 100% of documented Core Modules and User Capabilities are present and working locally (Part 8's mapping table fully checked off).

### Day 12 — Testing & Deployment
**Primary objective:** Prove the application works outside the development machine, and put it live.
**Tasks:**
- Run the full Part 11 test suite (data, ML, application); fix any failures
- Finalize `requirements.txt`; remove any local-only paths/dependencies
- Push the repository to GitHub (daily incremental commits should already exist from Day 1 onward — this is the final push, not the first)
- Deploy to Streamlit Community Cloud; run the Part 11 Deployment Tests against the live URL
**Expected output:** A passing test suite and a live, publicly reachable dashboard.
**Files/artifacts created:** `tests/*` finalized, `.streamlit/config.toml`, final `requirements.txt`
**Validation checkpoint:** The deployed app is smoke-tested identically to the local QA pass, from a clean browser session.
**Definition of done:** The public URL works end to end, matching local behavior exactly.

### Day 13 — Research Paper Draft
**Primary objective:** Write the full research paper using only real, already-produced results.
**Tasks:**
- Populate all 25 sections from Part 12 using the actual outputs of Days 2–9 (EDA figures, metrics tables, selection rationale, forecast analysis, early-warning results)
- Explicitly include every finding from the Source Verification Note in the Limitations section
**Expected output:** A complete draft research paper.
**Files/artifacts created:** `reports/research_paper.md`
**Validation checkpoint:** Every section from Part 12 is present; every number in the paper is cross-checked against an actual output file — zero figures written from memory or assumption.
**Definition of done:** The paper is complete except for final polish.

### Day 14 — Executive Summary, Documentation Polish (Buffer Day #2)
**Primary objective:** Finish the last required artifact and absorb any remaining slippage.
**Tasks:**
- Write the 1-page executive summary per Part 13, using final real metrics
- Finalize the README (setup, run steps, architecture, limitations, links)
- Polish code comments/docstrings across `src/`
- Re-run the full smoke test one more time
- Use any remaining slack to absorb delays from earlier days
**Expected output:** Finished executive summary and a fully polished repository.
**Files/artifacts created:** `reports/executive_summary.md`, final `README.md`
**Validation checkpoint:** A self-review against the full Part 16 checklist.
**Definition of done:** Every item on the Part 16 checklist can honestly be marked complete.

### Day 15 — Final QA & Submission
**Primary objective:** Confirm everything works, and submit.
**Tasks:**
- Run the Part 16 Final Project Checklist top to bottom
- Re-verify the deployed URL one final time
- Verify every link in the submission package resolves (repo, deployed app, paper, executive summary)
- Assemble and submit the package via the Unified Mentor "Submit Project" flow
**Expected output:** Submitted project.
**Files/artifacts created:** None new — this is a verification and submission day by design, intentionally kept light.
**Validation checkpoint:** Checklist 100% complete before submitting.
**Definition of done:** Project submitted and confirmed received.

---

# PART 15 — PRIORITY SYSTEM

### P0 — Must work (required for a successful submission)
- Cleaned, validated, correctly-reindexed dataset (Day 2)
- Both baselines, both statistical models (Exp. Smoothing, ARIMA/SARIMA), both ML models (Random Forest, Gradient Boosting) — **all six are named explicitly in the documentation**, so all six are P0, not illustrative examples
- Walk-forward validation, multi-horizon {1,7,14} evaluation, MAE/RMSE/MAPE(or safe alternative) for both primary targets
- Documented, evidence-based champion-model selection
- All 4 documented Streamlit Core Modules + all 3 User Capabilities, live and functional
- Live deployment (public URL)
- Research paper (all 25 sections, populated with real results)
- Executive summary
- Submission through the Unified Mentor portal

### P1 — Important (strongly improves professional quality)
- KPI dashboard cards (Forecast Accuracy %, Surge Lead Time, Capacity Breach Probability proxy, Forecast Stability Index)
- Automated test suite (Part 11) fully wired into a pre-deploy check
- Scenario Comparison page polish (Page 7)
- Deeper EDA visuals (decomposition, ACF/PACF plots) included in the paper, not just produced privately
- Early-warning/imbalance signal refinement beyond the minimum
- README polish that goes beyond the minimum (architecture diagrams, expanded setup troubleshooting)

### P2 — Nice to have (only after P0/P1 are complete)
- A fully independent third forecasting model for the intake/exit imbalance signal (Part 3.3 explicitly defers this)
- The ambiguous 5th KPI row (Finding #5) — only if a defensible interpretation can be confirmed, e.g. via the live portal
- Deep hyperparameter tuning beyond a light, time-boxed search
- Additional exogenous features not named in the documentation (e.g., an external policy-event calendar)
- Recursive (as opposed to direct) multi-horizon forecasting as an additional comparison
- Alternative deployment hosts beyond the primary choice
- Advanced UI theming beyond a clean, functional default

This ordering exists specifically to prevent overengineering: nothing in P1/P2 is attempted while any P0 item is incomplete.

---

# PART 16 — FINAL PROJECT CHECKLIST

**Data**
- [ ] 450 blank trailing rows removed; 720 real rows confirmed
- [ ] `Children in HHS Care` cleaned (commas stripped) and numeric
- [ ] All 5 numeric columns correctly typed (integer, not float-by-accident)
- [ ] Dataset reindexed to the true Sun–Thu schedule, with the 49 gap slots explicitly flagged
- [ ] 0 duplicate dates, 0 duplicate rows, 0 unexplained missing values
- [ ] Data tests passing

**Analysis**
- [ ] Decomposition, ACF/PACF, stationarity tests completed and documented
- [ ] Regime shift (peak → trough → rebound) explicitly characterized with real numbers
- [ ] Correlation reviewed on both raw and differenced series

**Forecasting**
- [ ] Both baselines implemented for both targets
- [ ] Both statistical models implemented for both targets
- [ ] Both ML models implemented for both targets
- [ ] All models evaluated at all 3 horizons

**Validation**
- [ ] Strict chronological split used throughout — no random split anywhere
- [ ] Walk-forward harness covers both the high-load and low-load regimes
- [ ] Every model compared against baseline on identical folds
- [ ] Champion model(s) selected with a documented, metric-cited rationale

**Dashboard**
- [ ] All 4 documented Core Modules present and functional
- [ ] All 3 documented User Capabilities present and functional
- [ ] Every page reads only from pre-generated artifacts (no live training)
- [ ] Capacity/threshold language on every relevant page clearly marked as a data-derived proxy, not an official figure

**Deployment**
- [ ] Public URL live and reachable
- [ ] Deployed behavior matches local behavior exactly
- [ ] No secrets/credentials required or mismanaged (none are needed for this project)

**Testing**
- [ ] Data tests, ML tests, application tests, deployment tests all passing

**Documentation**
- [ ] README covers setup, run steps, architecture, and every known limitation
- [ ] Config-driven code — no hardcoded paths/dates/thresholds found on review

**Research Paper**
- [ ] All 25 sections present
- [ ] Every figure/claim traceable to an actual produced artifact — zero fabricated numbers

**Executive Summary**
- [ ] 1 page, non-technical, answers all 8 questions from Part 13
- [ ] States the capacity-threshold limitation explicitly

**GitHub / Repository**
- [ ] Clean directory structure matching Part 9
- [ ] Incremental commit history (not one final dump commit)
- [ ] `requirements.txt` accurate and minimal

**Submission Package**
- [ ] Repository link
- [ ] Deployed app link
- [ ] Research paper
- [ ] Executive summary
- [ ] Submitted via the Unified Mentor "Submit Project" flow, deadline/format reconfirmed against the portal's Calendar/FAQs (not assumed, since the instructions page itself states no deadline)

This checklist is the objective answer to *"is this project actually finished?"* — every box traces back to either a [DOC] requirement or a specific finding from this roadmap's audit.

---

# PART 17 — RISKS AND FAILURE PREVENTION

| Risk | Detection | Mitigation | Fallback |
|---|---|---|---|
| Poor data quality (blank rows, comma formatting) | Day-1/2 audit — **already performed in this roadmap** | Explicit cleaning step with automated tests (Part 11) | Quarantine any newly-discovered bad rows rather than halting the pipeline |
| Insufficient time-series observations | Check held-out test window size against remaining training history each time it's adjusted | Keep test windows proportionate to the 720-row history | Reduce the number of walk-forward folds rather than shrinking training history below a sane minimum |
| Missing/irregular dates | Schedule-reconciliation test (Part 11) | Reindex to the *true* Sun–Thu schedule (Part 4), not a naive calendar week | Mask (exclude) rather than interpolate any gap where interpolation visibly distorts a model |
| Leakage | Automated leakage test (Part 11) | Strict backward-only shifting in feature construction | If any leak is found, rebuild the feature pipeline from scratch rather than patching around the symptom |
| Overfitting (RF/GBR on ~700 rows with many engineered features) | Compare training error vs. walk-forward test error | Modest tree depth/estimator counts; light, time-boxed tuning only | Fall back to the simpler statistical model as champion if ML doesn't generalize |
| Weak validation | Self/peer review of split logic before Day 6 | Programmatic chronological-order assertions on every fold | Non-negotiable — there is no acceptable fallback for a broken split |
| Model instability | Forecast Stability Index across refits/folds | Prefer the more stable model when errors are close between candidates | Widen the reported confidence interval and disclose lower confidence, rather than hiding instability |
| Forecasts that look good but aren't useful | Domain-plausibility sanity checks (no negative forecasts, no implausible horizon-14 swings) | Post-processing guardrails + a manual review pass | State the limitation openly in the paper rather than silently accepting an implausible output |
| Arbitrary capacity thresholds | Already found: **no threshold exists in either source (Finding #6)** | Relative/statistical proxy thresholds, explicitly labeled as such everywhere they appear | If the proxy proves unreliable in backtesting, present qualitative risk tiers only, without a numeric "probability" claim |
| Dashboard disconnected from the actual model | Version/consistency check between `models/` artifacts and `forecasts/` outputs the app reads | Single batch-generation script (Day 9) is the sole producer of dashboard data | Regenerate artifacts and redeploy immediately if any mismatch is found |
| Deployment failures | Day-12 clean-environment smoke test | Pinned dependencies; no local-only paths (Part 11 Deployment Tests) | Fall back to a secondary host (e.g., Render/Hugging Face Spaces) only if the primary choice is genuinely blocked |
| Spending too much time on UI | Track actual vs. planned time on Days 10–11 | Page count capped at 8, each justified against a requirement (Part 8) | Cut unstarted P1/P2 dashboard polish before cutting into Days 12–15 |
| Spending too much time on advanced models | Track actual vs. planned time on Days 6–7 | Modeling scope capped at exactly the 6 named models (Part 15 P0) | Stop tuning and move to Day 8 even if results feel "improvable" — the harness already gives an honest answer |
| Weak final report | Day 13 is a dedicated day with a pre-defined 25-section structure (Part 12) | Populate only with real evidence, never write-ahead placeholders that get forgotten | If Day 13 runs short, use Day 14's buffer rather than compressing into Day 15 |
| Fabricated/unsupported conclusions | Day-15 traceability pass: every claim in the paper mapped back to a specific produced file/number | The "no fabricated results" rule is enforced throughout this roadmap, not just at the end | Delete or caveat any claim that can't be traced, rather than leaving it in |

---

# PART 18 — DEFINITION OF THE FINISHED PRODUCT

The finished project is:

**A cleaned, validated UAC time-series dataset + a leakage-safe, walk-forward-validated forecasting pipeline (2 baselines, 2 statistical models, 2 ML models, across 2 primary targets and 3 horizons) + a documented, evidence-based model-selection outcome + an early-warning layer built on honestly-labeled relative thresholds + an 8-page Streamlit dashboard covering every documented Core Module and User Capability + a live, publicly deployed instance of that dashboard + a reproducible, tested, version-controlled repository + a complete 25-section research paper + a 1-page executive summary + a final submission package delivered through the Unified Mentor portal.**

### Minimum acceptance criteria

1. The dataset pipeline reproduces exactly 720 clean rows from the raw CSV, with 0 nulls/duplicates, every time it's run.
2. All 6 documented models exist, are validated on identical walk-forward folds, and every one of them is compared against the naive baseline.
3. A champion model is selected per target with a written rationale citing specific metric numbers — not asserted.
4. Every documented Streamlit Core Module and User Capability is present, functional, and reachable at a live public URL — not only on localhost.
5. Every place a capacity/threshold figure appears states plainly that it is a data-derived proxy, because no official threshold exists in either source.
6. The research paper's 25 sections are complete and every quantitative claim in it traces to a real, produced artifact.
7. The executive summary is understandable by a non-technical reader and honestly states the project's limitations.
8. The repository installs and runs cleanly in a fresh environment, per the README, without manual patching.
9. All 8 findings from the Source Verification Note are disclosed in the final documentation — none are silently resolved or hidden.
10. The submission package (repo + deployed URL + paper + executive summary) is assembled and submitted through the Unified Mentor portal.

If all ten hold, the project is finished. If any one doesn't, that item — and only that item — is what remains to be done; this checklist is deliberately built so "done" is never a matter of opinion.
