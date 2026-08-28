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
pip install -r requirements.txt

# Generate forecast artifacts
python src/forecast/generate.py

# Run the dashboard
streamlit run app/Home.py
```

## Project Structure

```
├── data/raw/                  # Original CSV (source of truth)
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
├── requirements.txt
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
