# Discrepancy Log — UAC Forecasting Project

**Source:** Day 1 Data Audit  
**Reference:** Source Verification Note in UAC_Forecasting_Execution_Roadmap

## 8 Confirmed Discrepancies

### Finding #1 — Blank Trailing Rows
- **Evidence:** Raw CSV has 1,170 rows but only 720 contain data. Rows 721–1,170 (450 rows) are 100% blank.
- **Impact:** Naive loading inflates statistics and corrupts dtype inference (float64 instead of int).
- **Resolution:** Day 2 cleaning truncates to 720 real rows as the first data-quality gate.

### Finding #2 — Non-Calendar-Daily Cadence
- **Evidence:** Day-of-week counts: Mon 145, Tue 149, Wed 147, Thu 147, Fri 2, Sat 0, Sun 130. Reporting is Sun–Thu.
- **Impact:** Calendar-daily reindexing would create ~35% synthetic data (Fri/Sat never reported).
- **Resolution:** Reindex to true Sun–Thu reporting schedule. Only 49 genuine gaps filled — not 355+ fake calendar days.
- **Off-template Fridays:** 2024-09-13, 2025-04-11 — retained as standalone rows at their real dates.

### Finding #3 — HHS Care Column Is String Type
- **Evidence:** `Children in HHS Care` contains thousands-separator commas (e.g., "2,484"). dtype = object.
- **Impact:** All downstream computation fails without explicit comma-stripping and numeric cast.
- **Resolution:** Day 2 cleaning strips commas, casts to int, with unit test asserting zero parse failures.

### Finding #4 — Float Dtypes From Blank Rows
- **Evidence:** 4 numeric columns load as float64 (not int) because NaN from blank rows forces upcast.
- **Impact:** Minor, but a canary confirming Finding #1.
- **Resolution:** Once blank rows are dropped, all values are integers. Documented in data dictionary.

### Finding #5 — Malformed KPI Table
- **Evidence:** Two consecutive `<tr>` rows: "Forecast Stability Index → Model robustness", then "Model robustness → Long-term reliability". Second row's KPI name is the first row's description.
- **Impact:** 5th KPI cannot be unambiguously defined.
- **Resolution:** Treated as 4 well-defined KPIs + 1 flagged ambiguous item (P2/optional). Ambiguity documented in report.

### Finding #6 — No Capacity Threshold
- **Evidence:** No numeric shelter capacity figure exists in either source (full-text search confirmed).
- **Impact:** "Capacity Breach Probability" KPI requires some reference point.
- **Resolution:** Relative/statistical proxy (trailing 90th percentile), explicitly labeled as data-derived proxy everywhere it appears. Never presented as an official capacity figure.

### Finding #7 — Unresolved Footnote Asterisk
- **Evidence:** `Children apprehended and placed in CBP custody*` — asterisk has no footnote text in extracted documentation.
- **Impact:** Possible scope caveat in original data source cannot be verified.
- **Resolution:** Column used per its plain documented definition ("Daily intake volume"). Asterisk noted as unresolved in Limitations section.

### Finding #8 — Major Regime Shift
- **Evidence:** `Children in HHS Care` swings from 1,972 (Aug 2025) to 11,516 (Dec 2023) — ~5.8× range.
- **Impact:** Non-stationary; plain baselines/undifferenced ARIMA handle poorly.
- **Resolution:** Explicit stationarity testing (ADF/KPSS), training-window cap at 2025-02-05 (from structural-break analysis), differencing before model selection.

## Additional Observations (Day 1 Audit)

### Flow Identity Non-Closure
- **Evidence:** `HHS_Care[t] ≈ HHS_Care[t-1] + Transferred[t] − Discharged[t]` holds exactly on only ~2.5% of rows.
- **Impact:** The 5 published columns do not fully explain day-to-day stock changes. Other flows exist (reclassifications, age-outs, data revisions).
- **Implication:** Net-flow features carry real signal without being leakage by definition. Purely arithmetic model cannot replace statistical/ML forecasting.

### Zero-Value Days
- **Evidence:** Apprehended: 2 zero-days, Transferred Out: 3, Discharged: 1. CBP Custody/HHS Care: 0 zeros.
- **Impact:** MAPE unstable for flow columns with zeros.
- **Resolution:** sMAPE/MASE reported alongside MAPE for affected series.

---

*All findings above are computed from actual data inspection, not assumed. Exact numbers verified by running `src/data/audit.py` against the raw CSV.*
