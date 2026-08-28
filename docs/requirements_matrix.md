# Requirements Matrix — UAC Forecasting Project

**Source:** Unified Mentor Project Allotment Portal (project ID 18744)  
**Extracted:** Day 1 Source Audit  
**Tags:** [DOC] = from documentation · [IMPLICIT] = professionally necessary · [ENG-REC] = engineering recommendation

## Mandatory Requirements (P0)

| # | Requirement | Source | Evidence |
|---|---|---|---|
| 1 | Forecast children in HHS Care | [DOC] Primary Objectives | Column exists as `Children in HHS Care` |
| 2 | Predict short-term discharge demand | [DOC] Primary Objectives | Column exists as `Children discharged from HHS Care` |
| 3 | Estimate future intake/exit imbalance | [DOC] Primary Objectives | Derived: `Transferred Out − Discharged` |
| 4 | Early warnings for healthcare planners | [DOC] Secondary Objectives | Requires relative/statistical thresholds (no official capacity figure exists) |
| 5 | Quantify forecast uncertainty | [DOC] Secondary Objectives | Native CIs for statistical models; residual-quantile intervals for ML |
| 6 | Compare statistical vs ML approaches | [DOC] Secondary Objectives | Both families on identical validation harness |
| 7 | Time-series preparation | [DOC] Methodology Step 1 | Datetime index, continuity on Sun–Thu schedule, missing-day handling |
| 8 | Feature engineering | [DOC] Methodology Step 2 | Lags t-1/t-7/t-14, rolling 7/14 mean+variance, flow signal, calendar effects |
| 9 | Strict time-based train/test split | [DOC] Methodology Step 3 | No random sampling — chronological only |
| 10 | Walk-forward validation | [DOC] Methodology Step 3 | Expanding-window, rolling-origin CV |
| 11 | Multi-horizon evaluation | [DOC] Methodology Step 3 | Horizons = {1, 7, 14} periods ahead |
| 12 | Baseline models | [DOC] Forecasting Models | Naive persistence, moving average |
| 13 | Statistical models | [DOC] Forecasting Models | ARIMA/SARIMA, Exponential Smoothing |
| 14 | ML models | [DOC] Forecasting Models | Random Forest, Gradient Boosting |
| 15 | Metrics: MAE, RMSE, MAPE | [DOC] Model Evaluation | MAPE flagged unstable for flow columns with zeros → sMAPE/MASE alongside |
| 16 | KPIs: Forecast Accuracy %, Surge Lead Time, Capacity Breach Probability, Forecast Stability Index | [DOC] KPIs | No formulas given — defined in implementation |
| 17 | Streamlit: 4 Core Modules | [DOC] Dashboard | Care Load Forecast, Discharge Demand, Model Comparison, CI Visualization |
| 18 | Streamlit: 3 User Capabilities | [DOC] Dashboard | Horizon selector, model toggle, scenario comparison |
| 19 | 3 Deliverables | [DOC] Submission | Research paper, Streamlit dashboard, executive summary |
| 20 | Live deployment | [DOC] "live analytics" | Publicly reachable URL |
| 21 | Portal submission | [DOC] Navigation | Via Unified Mentor "Submit Project" flow |

## Recommended Requirements (Professional Necessity)

| # | Requirement | Source |
|---|---|---|
| 22 | Reproducible environment | [IMPLICIT] — `requirements.txt`, README |
| 23 | Data validation tests | [IMPLICIT] — schema/date/duplicate checks |
| 24 | Version-controlled repository | [IMPLICIT] — GitHub with incremental commits |
| 25 | Config-driven code | [IMPLICIT] — central `config.py`, no magic numbers |
| 26 | Model/forecast artifact persistence | [IMPLICIT] — no retrain-on-page-load |
| 27 | Documented assumptions/limitations | [IMPLICIT] — given 8 source discrepancies |

## Flagged / P2

| # | Item | Status |
|---|---|---|
| 16b | 5th KPI row ("Model robustness" / "Long-term reliability") | Malformed in source — implement only if time allows, documented as ambiguous |

## Model Families (All Required)

Per the documentation and addendum:

1. **Naive persistence** — baseline
2. **Seasonal naive** — baseline
3. **Moving average** — baseline
4. **Exponential Smoothing** → `ETSModel` (`missing='drop'` for Discharged)
5. **ARIMA/SARIMA** → `SARIMAX`
6. **Random Forest** → `RandomForestRegressor` (NaN rows dropped, count logged)
7. **Gradient Boosting** → `HistGradientBoostingRegressor` (native NaN support)
8. **Post-hoc ensemble** — average of champion statistical + champion ML (evaluated, not pre-declared winner)

## Targets

- **Target 1:** `Children in HHS Care` (stock/level variable)
- **Target 2:** `Children discharged from HHS Care` (flow variable)
- **Derived signal:** Intake/exit imbalance = `Transferred Out − Discharged`

## Validation Protocol (Locked)

- Final test window: 60 most recent real observations
- Walk-forward step: 10 periods
- Minimum initial training: 50 periods
- Training cap date: 2025-02-05 (with ~60–80 row fallback)
- Baseline-beats gate: every model must beat naive/seasonal-naive
- Early-warning threshold: trailing 90th percentile
