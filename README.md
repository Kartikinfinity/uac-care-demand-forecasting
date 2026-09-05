# Predictive Forecasting of Care Load & Placement Demand

**Program:** Unified Mentor — Project Allotment Portal  
**Context:** U.S. Department of Health and Human Services (HHS) / Unaccompanied Alien Children (UAC) Program  
**Prepared as:** Independent analytical exercise using HHS UAC program data

## Overview

This project builds a multi-target, multi-horizon time-series forecasting system for the UAC Program's care load and discharge demand, with an early-warning capacity-stress layer.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # runtime + test tooling

# Rebuild every artifact from the raw CSV, in order
python -m src.data.clean               # -> data/interim/master_series.parquet
python -m src.eda                      # -> docs/eda_findings.md, reports/figures/
python -m src.features.build_features  # -> data/processed/features_target*.parquet
python -m src.evaluation.run_baselines   # -> forecasts/baseline_*.csv, docs/day5_baseline_metrics.md
python -m src.evaluation.run_statistical # -> forecasts/statistical_*.csv, docs/day6_statistical_metrics.md, models/stat_*.pkl
python -m src.evaluation.run_ml          # -> forecasts/full_model_comparison.csv, oos_residuals.csv, docs/day7_ml_metrics.md, models/ml_*.pkl
python -m src.evaluation.run_selection   # -> models/model_registry.json, docs/model_selection_rationale.md, forecasts/champion_selection.csv
python -m src.forecast.generate          # -> forecasts/{forward_forecasts,interval_coverage,holdout_evaluation,
                                         #      imbalance_forecast,early_warning_backtest,kpi_summary}.csv + provenance.json

# Run the test suite
python -m pytest -q

# Run the dashboard (available from Day 10)
streamlit run app/Home.py
```

### Data location

The raw CSV ships at the **repository root** as
`HHS_Unaccompanied_Alien_Children_Program (1).csv`, and `src/config.py`
resolves it from there. The roadmap's directory sketch shows it under
`data/raw/`; the file was delivered at the root and is read-only either way.
This deviation is recorded rather than silently reconciled, so the documented
path matches the path the code actually uses.

## Project Structure

```
├── HHS_..._Program (1).csv    # Original CSV (source of truth, read-only)
├── data/raw/                  # reserved (see "Data location" above)
├── data/interim/              # Cleaned master series
├── data/processed/            # Model-ready feature tables
├── src/
│   ├── config.py              # All parameters (no hardcoded values elsewhere)
│   ├── data/                  # Loading, cleaning, validation
│   ├── features/              # Feature engineering
│   ├── models/                # Model implementations
│   ├── evaluation/            # Walk-forward validation harness
│   └── forecast/              # Batch artifact generation
├── models/                    # Serialized trained models
├── forecasts/                 # Pre-generated forecast artifacts
├── app/                       # Streamlit dashboard
├── tests/                     # Automated test suite
├── reports/                   # Research paper, executive summary
├── docs/                      # Requirements matrix, discrepancy log
├── notebooks/                 # EDA notebooks
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # + test tooling
└── README.md
```

## Known Limitations & Data Discrepancies

1. Raw CSV contains 1,170 rows but only 720 carry data (450 blank trailing rows)
2. Reporting cadence is Sun–Thu, not calendar-daily (Fri: 2 obs, Sat: 0)
3. `Children in HHS Care` column is string-typed due to thousands-separator commas
4. No official capacity threshold exists — all capacity-stress signals use a data-derived statistical proxy
5. KPI table in official documentation has a malformed 5th row
6. Column `Children apprehended...` carries an unresolved footnote asterisk
7. Flow columns do not exactly reconcile against HHS Care stock (~2.5% exact match)
8. `Children in HHS Care` exhibits a ~5.8× regime shift (11,516 → 1,972)

## Refresh Policy

This dashboard reads from pre-generated forecast artifacts. It is **not** a continuously live system.
To update forecasts with new data, replace the CSV, run `generate.py`, and redeploy.

## Hosting

Deployed on Streamlit Community Cloud. Free-tier instances may experience cold-start delays of ~30 seconds on first load.
