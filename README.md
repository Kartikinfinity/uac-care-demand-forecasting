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

## Deliverables

| Deliverable | Path | Audience |
|---|---|---|
| Live dashboard | <https://uac-care-demand-forecasting-yjwnfnfaw8gqjcpvevlqjo.streamlit.app> | Operational |
| Executive summary | `reports/executive_summary.md` | Non-technical stakeholder, one page |
| Research paper | `reports/research_paper.md` | Technical, 25 sections |
| Evidence ledgers | `reports/paper_evidence.json`, `reports/executive_summary_evidence.json` | Anyone checking a figure |

**Both documents are generated, not written by hand.** Every numeral in them is
fetched from an artifact by `src/reporting/` and recorded in the matching
evidence ledger with the file and locator it came from. `tests/test_day13.py`
scans the finished paper and fails if it contains a number the ledger never
issued, so a figure typed from memory cannot survive the build; the executive
summary is additionally asserted byte-identical on regeneration. Rebuild both
with:

```bash
python -m src.reporting.build_paper
```

```bash
python -m src.reporting.build_executive_summary
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

## Reproducibility, Seed & Versioning

**Seed.** Every stochastic step uses `RANDOM_SEED = 42`, set once in
`src/config.py` and threaded through model fitting and the paired bootstrap
(`PRACTICAL_EQUIVALENCE_SEED` is the same value). Re-running the pipeline on the
same input CSV reproduces every artifact, and two of the generated documents are
asserted byte-identical on regeneration by the test suite. There is no hidden
source of randomness: nothing samples at serve time, and the dashboard never
fits a model.

**Data versioning.** There is no version number on the dataset, so the data is
identified by content instead. `src/data/validate.py` hashes the raw CSV and the
derived master series with SHA-256 on every run, and writes both digests plus a
`data_as_of` date into `data/interim/provenance.json`. Every downstream
artifact — forecasts, registry, research paper, executive summary — carries the
same pair of digests, so any output can be traced to the exact bytes it came
from.

Current vintage:

| | |
|---|---|
| `data_as_of` | 2025-12-21 |
| Raw CSV SHA-256 | `061af0a97a1b3bda7a36f0ce8df08b6994847b0a4c402828cf2ef835d4947198` |
| Master series SHA-256 | `3c6b093d73744140ac7d1d68d308e9f3f0f070fdc3d26e8c6e3d03bf24243c6d` |

**Checking a deployment matches.** The app serves its provenance sidecar as a
static file, and `scripts/smoke_test.py` compares the deployed digests against
the local ones. That is what catches the failure mode this architecture is prone
to: a dashboard serving a stale pre-generated artifact without erroring. If you
replace the CSV, every digest changes and the mismatch is reported rather than
silently absorbed.

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

**A note on Streamlit Cloud and HTTP 303.** The cloud answers a cookieless client
with `303 → share.streamlit.io/-/auth/app` on the first request to *any* path,
**including public apps**. A browser follows the hop, picks up a session cookie
and comes back; a script does not, and sees the redirect forever. A 303 therefore
says nothing about whether an app is private. The smoke test handles this by
falling back to the cloud's internal `/~/+/<path>` route, which skips the auth
hop — and it must use that route for the provenance fetch too, because a
cookie-jar client is handed the SPA shell (HTML) for
`/app/static/provenance.json` rather than the JSON.

If the live checks fail outright, the likely cause is that the free-tier instance
has **gone to sleep**. Open the URL in a browser once, click *"Yes, get this app
back up!"*, wait for it to finish booting, then re-run.

**Live deployment:**
<https://uac-care-demand-forecasting-yjwnfnfaw8gqjcpvevlqjo.streamlit.app>

## Hosting

Streamlit Community Cloud. Free-tier instances sleep after inactivity and may
take ~30 seconds to wake on first load.
