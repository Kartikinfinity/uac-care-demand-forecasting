# EDA Findings — UAC Forecasting Project

**Generated:** Day 3 automated EDA pipeline
**Purpose:** Drive modeling decisions for Days 4-6
**All numbers computed from actual data, never fabricated.**

---

## 1. Stationarity Tests

| Series | ADF stat | ADF p | ADF conclusion | KPSS stat | KPSS p | KPSS conclusion | Joint |
|---|---|---|---|---|---|---|---|
| HHS_Care (raw) | -1.0931 | 0.7178 | non-stationary | 0.5691 | 0.0100 | non-stationary | NON-STATIONARY (both tests agree) |
| HHS_Care (1st diff) | -5.1938 | 0.0000 | stationary | 0.0923 | 0.1000 | stationary | STATIONARY (both tests agree) |
| Discharged (raw) | -0.7065 | 0.8451 | non-stationary | 0.4954 | 0.0100 | non-stationary | NON-STATIONARY (both tests agree) |
| Discharged (1st diff) | -7.7324 | 0.0000 | stationary | 0.0472 | 0.1000 | stationary | STATIONARY (both tests agree) |

**Decision for Day 6:** Both targets require differencing (d=1) to achieve stationarity for SARIMA. The first-differenced series should be tested to confirm stationarity before fixing SARIMA orders.

## 2. Structural Break Scan

### HHS_Care
- Dominant break: **2025-01-02** (CUSUM = 896987)
- Secondary break: **2024-07-02** (CUSUM = 885432)

### Discharged
- Dominant break: **2024-07-28** (CUSUM = 35874)
- Secondary break: **2024-07-04** (CUSUM = 35623)

**Locked decision:** Training-window cap at **2025-02-05**, frozen by `PRE_BUILD_TECHNICAL_ADDENDUM.md` Section 2 -- NOT derived from this scan.

This scan is corroborating evidence only. It locates the dominant CUSUM deviation at the dates listed above, which sit near but are not identical to the frozen cap date. The scan therefore supports the existence of a regime change in this period; it does not by itself select 2025-02-05, and this document does not claim that it does.

## 3. Within-Week Seasonality

| Target | Test | H stat | p-value | Seasonal? | Diff H stat | Diff p-value | Diff Seasonal? |
|---|---|---|---|---|---|---|---|
| HHS_Care | Kruskal-Wallis | 2.0826 | 0.837600 | False | 152.3054 | 0.000000 | True |
| Discharged | Kruskal-Wallis | 32.2839 | 0.000005 | True | 313.9764 | 0.000000 | True |

**SARIMA seasonal order decision:**

- **HHS Care:** Within-week seasonality detected in differenced series. Use SARIMA with seasonal period m=5 (5-day reporting week).
- **Discharged:** Within-week seasonality detected in differenced series. Use SARIMA with seasonal period m=5.

## 4. Seasonal Decomposition (STL, period=5)

| Target | Seasonal Strength | Trend Strength |
|---|---|---|
| HHS_Care | 0.4114 | 0.9993 |
| Discharged | 0.4454 | 0.9253 |

Seasonal strength close to 0 = weak seasonality; close to 1 = strong. Trend strength close to 1 = dominant trend component.

## 5. Cross-Series Correlations

### Transferred Out vs Discharged (imbalance signal components)

- Raw levels: **0.6573**
- First difference: **0.0739**
- Detrended: **0.0989**

**Decision for Day 8:** The addendum pre-registered expectation of near-independence in first differences (confirmed — diff correlation = 0.0739). Use the full Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B) formula unless first-diff correlation is negligible.

## 6. Zero-Value Days

### Children apprehended and placed in CBP custody*: 2 zero-value day(s)

- 2025-07-15 (Tuesday, July)
- 2025-08-10 (Sunday, August)

### Children transferred out of CBP custody: 3 zero-value day(s)

- 2025-03-17 (Monday, March)
- 2025-07-09 (Wednesday, July)
- 2025-07-16 (Wednesday, July)

### Children discharged from HHS Care: 1 zero-value day(s)

- 2025-11-30 (Sunday, November)

**Holiday clustering assessment:** Review whether zero-value dates coincide with US federal holidays. Sparse zeros (2-3 per column) do not indicate systematic holiday effects; MAPE remains unstable for these columns — sMAPE/MASE needed alongside.

## 7. Summary of Modeling Decisions from EDA

| Decision | Value | Evidence |
|---|---|---|
| Differencing order (d) | 1 for both targets | ADF/KPSS tests above |
| Seasonal period (HHS Care) | m=5 | Kruskal-Wallis p=0.000000 |
| Seasonal period (Discharged) | m=5 | Kruskal-Wallis p=0.000000 |
| Training cap date | 2025-02-05 | Frozen by addendum Sec. 2; scan corroborates a regime change nearby |
| Rolling window design | Period-based (not calendar) | Already locked in addendum |
| MAPE reliability | Unstable for flow columns | Small denominators (flow minima near 0) plus zero-value days |
| Imbalance signal independence | Near-independent | Diff correlation = 0.0739 |

---

## Figures Generated

All figures saved to `reports/figures/`:

- `structural_break_hhs_care.png`
- `structural_break_discharged.png`
- `weekly_seasonality_hhs_care.png`
- `weekly_seasonality_discharged.png`
- `acf_pacf_hhs_care.png`
- `acf_pacf_discharged.png`
- `stl_decomp_hhs_care.png`
- `stl_decomp_discharged.png`
- `regime_shift_overview.png`
- `correlation_heatmaps.png`