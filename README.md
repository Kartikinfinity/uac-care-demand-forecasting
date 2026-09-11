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
python -m src.reporting.feature_importance # -> forecasts/feature_importance.csv
python -m src.reporting.comparison_matrix  # -> forecasts/comparison_matrix.csv (8 models)
python -m src.reporting.build_paper        # -> reports/research_paper.md + reports/paper_evidence.json

# Run the test suite
python -m pytest -q

# Run the dashboard (reads pre-generated artifacts only, never trains)
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

## Research Paper

`reports/research_paper.md` is **generated**, not written by hand. Every numeral
in it is fetched from an artifact by `src/reporting/build_paper.py` and recorded
in `reports/paper_evidence.json` with the file and locator it came from.
`tests/test_day13.py` then scans the finished document and fails if it contains a
number the ledger never issued — so a figure typed from memory cannot survive the
build. Regenerate it with:

```bash
python -m src.reporting.build_paper
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
To update forecasts with new data, replace the CSV, re-run the pipeline above, and redeploy.
Nothing retrains on a schedule, and no page trains anything at any time.

## Deployment

The app is a pure consumer of pre-generated artifacts, so a deployment needs no
build step — but it does need those artifacts present in the clone. The ~128 KB
the dashboard actually reads is therefore committed (see `.gitignore` for the
allow-list and the reasoning); the ~3.4 MB of regenerable intermediates is not.

**Deploy to Streamlit Community Cloud**

1. Push to GitHub (`master`).
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   this repository, branch `master`, main file **`app/Home.py`**.
3. Set the Python version in the advanced settings to the newest the platform
   offers. Development and the clean-environment verification both ran on 3.14
   and every dependency resolved; the lower bounds in `requirements.txt` are
   satisfiable on 3.11+.

   A fresh install resolves **pandas 3.x** while development ran on pandas 2.3.
   That difference already broke one page (`pd.NA` into a plain `bool` column),
   caught by the clean-environment test and fixed. If you pin a different Python
   or pandas version, re-run the smoke test.
4. Deploy, then verify against the live URL:

```bash
python scripts/smoke_test.py --url https://<your-app>.streamlit.app
```

**What the smoke test proves.** It checks that every one of the eight pages
responds, and — the part that matters — that the *deployed* app is serving the
same data version this repository holds. The app publishes its provenance
sidecar as a static file, so the test fetches
`/app/static/provenance.json` from the live URL and compares the raw-CSV and
master-series SHA-256 against the local artifact. A dashboard reading stale
pre-generated files fails silently otherwise: nothing errors, the numbers are
simply from a different vintage than the code implies. Exit code is non-zero if
any check fails, so it can gate a deploy.

Run it with `--skip-live` to check only that the committed artifacts describe the
committed data.

## Hosting

Streamlit Community Cloud. Free-tier instances sleep after inactivity and may
take ~30 seconds to wake on first load.
