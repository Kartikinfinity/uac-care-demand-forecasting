"""
config.py — Central configuration for UAC Forecasting Pipeline

Every parameter fixed in the PRE_BUILD_TECHNICAL_ADDENDUM is defined here.
Nothing is hardcoded elsewhere in the codebase.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────
# PROJECT PATHS (resolved relative to project root)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw data file
RAW_CSV_FILENAME = "HHS_Unaccompanied_Alien_Children_Program (1).csv"
RAW_CSV_PATH = PROJECT_ROOT / RAW_CSV_FILENAME

# Interim artifacts
MASTER_SERIES_PATH = DATA_INTERIM_DIR / "master_series.parquet"

# ──────────────────────────────────────────────────────────────────────
# COLUMN NAMES (defined once, imported everywhere)
# ──────────────────────────────────────────────────────────────────────
COL_DATE = "Date"
COL_APPREHENDED = "Children apprehended and placed in CBP custody*"
COL_CBP_CUSTODY = "Children in CBP custody"
COL_TRANSFERRED = "Children transferred out of CBP custody"
COL_HHS_CARE = "Children in HHS Care"
COL_DISCHARGED = "Children discharged from HHS Care"

# Ordered list of all numeric columns
NUMERIC_COLS = [
    COL_APPREHENDED,
    COL_CBP_CUSTODY,
    COL_TRANSFERRED,
    COL_HHS_CARE,
    COL_DISCHARGED,
]

# Stock columns (interpolated at gap slots)
STOCK_COLS = [COL_HHS_CARE, COL_CBP_CUSTODY]

# Flow columns (left as true-missing at gap slots — NEVER zero-filled or interpolated)
FLOW_COLS = [COL_APPREHENDED, COL_TRANSFERRED, COL_DISCHARGED]

# Target columns
TARGET_1 = COL_HHS_CARE       # Stock target
TARGET_2 = COL_DISCHARGED     # Flow target

# ──────────────────────────────────────────────────────────────────────
# DATA CONSTRUCTION PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# Expected row counts per the addendum Section 2
EXPECTED_REAL_ROWS = 720
EXPECTED_TEMPLATE_POSITIONS = 767  # Sun–Thu canonical template
OFF_TEMPLATE_FRIDAYS = ["2024-09-13", "2025-04-11"]  # 2 confirmed standalone Friday rows
EXPECTED_TOTAL_POSITIONS = 769  # 767 + 2 off-template Fridays
EXPECTED_GAP_COUNT = 49  # 769 - 720 = 49 gaps

# Raw file has 450 blank trailing rows
EXPECTED_RAW_ROWS = 1170
EXPECTED_BLANK_ROWS = 450

# Reporting schedule: Sunday through Thursday (the true cadence)
REPORTING_WEEKDAYS = [6, 0, 1, 2, 3]  # Sun=6, Mon=0, Tue=1, Wed=2, Thu=3

# Date format in the raw CSV
DATE_FORMAT = "%B %d, %Y"  # e.g., "December 21, 2025"

# ──────────────────────────────────────────────────────────────────────
# REGIME / TRAINING-WINDOW PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# Training-window cap date (from structural-break analysis)
TRAINING_CAP_DATE = "2025-02-05"

# Minimum usable rows before fallback to full-expanding window
MIN_USABLE_ROWS_FLOOR = 60  # lower bound of the ~60–80 range
MAX_USABLE_ROWS_FLOOR = 80  # upper bound

# ──────────────────────────────────────────────────────────────────────
# VALIDATION PROTOCOL PARAMETERS (Section 5 of addendum)
# ──────────────────────────────────────────────────────────────────────
FINAL_TEST_WINDOW = 60       # Most recent 60 real observations
WALK_FORWARD_STEP = 10       # Period step between successive fold origins
MIN_INITIAL_TRAINING = 50    # Minimum initial training size

# Forecast horizons
FORECAST_HORIZONS = [1, 7, 14]

# ──────────────────────────────────────────────────────────────────────
# EARLY-WARNING PARAMETERS (Section 8 of addendum)
# ──────────────────────────────────────────────────────────────────────
EARLY_WARNING_PERCENTILE = 90  # 90th percentile, trailing-only

# ──────────────────────────────────────────────────────────────────────
# MODEL PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# These will be populated during Days 5-7 with evidence-based selections.
# Stubs here for structure.
MODEL_FAMILIES = [
    "naive",
    "seasonal_naive",
    "moving_average",
    "exponential_smoothing",  # ETSModel with missing='drop' for Discharged
    "sarima",                 # SARIMAX
    "random_forest",          # RandomForestRegressor
    "gradient_boosting",      # HistGradientBoostingRegressor
    "ensemble",               # Post-hoc average of champion stat + champion ML
]
