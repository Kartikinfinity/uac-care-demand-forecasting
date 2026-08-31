import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import timedelta

import holidays

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    MASTER_SERIES_PATH, NUMERIC_COLS, TARGET_1, TARGET_2,
    COL_TRANSFERRED, COL_DISCHARGED, DATA_PROCESSED_DIR, COL_DATE,
    LAG_PERIODS, ROLLING_WINDOWS, ROLLING_MIN_PERIODS_MEAN, ROLLING_MIN_PERIODS_VAR,
)

PROCESSED_DIR = DATA_PROCESSED_DIR
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-based calendar features without leaking future info."""
    df_feat = df.copy()
    
    # Extract basic calendar elements
    df_feat['day_of_week'] = df_feat['parsed_date'].dt.dayofweek
    df_feat['month'] = df_feat['parsed_date'].dt.month
    
    # Holiday proximity flag (US federal holidays)
    us_holidays = holidays.US(years=df_feat['parsed_date'].dt.year.unique())
    
    # Holiday PROXIMITY, deliberately calendar-based: this is the only feature
    # that is genuinely about wall-clock dates rather than reporting periods, so
    # invariant 1 (no calendar arithmetic) does not apply to it. Lags, rolling
    # windows and horizons all remain strict period-position offsets.
    #
    # Membership is tested on datetime.date objects. Adding a pandas Timedelta
    # to a Timestamp and testing THAT against the holidays mapping raised a
    # NumPy deprecation on every call (3,588 warnings per test run) and is slated
    # to become an error.
    def is_near_holiday(d):
        base = d.date()
        return int(any((base + timedelta(days=offset)) in us_holidays
                       for offset in range(-2, 3)))

    df_feat['is_near_holiday'] = df_feat['parsed_date'].apply(is_near_holiday)
    
    return df_feat

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly period-position lags and rolling statistics.
    Invariant: Never include the current row in its own rolling-window feature.
    """
    df_feat = df.copy()
    
    # 1. Net Flow
    df_feat['net_flow'] = df_feat[COL_TRANSFERRED] - df_feat[COL_DISCHARGED]
    
    cols_to_lag = NUMERIC_COLS + ['net_flow']
    
    # 2. Lags -- integer offsets on the period-position index, never calendar
    #    arithmetic (invariant 1). `.shift(k)` on a positionally-ordered frame
    #    is exactly a k-period-position offset.
    for col in cols_to_lag:
        for lag in LAG_PERIODS:
            df_feat[f'lag_{lag}_{col}'] = df_feat[col].shift(lag)

    # 3. Rolling windows.
    #    The series is shifted by one period FIRST, so the current row can never
    #    enter its own window (invariant 3).
    #
    #    On min_periods: these are permissive by design (see config). Flow
    #    columns are true-missing at gap slots and pandas' rolling reductions
    #    skip NaN, so a `rolling_7` value spans 7 period-positions but may rest
    #    on fewer than 7 observations. The name describes the window SPAN, not
    #    the sample size. An `n_obs` column is emitted alongside every rolling
    #    statistic so the true sample size is visible to any consumer rather
    #    than implied by the column name.
    for col in cols_to_lag:
        for w in ROLLING_WINDOWS:
            shifted_series = df_feat[col].shift(1)
            df_feat[f'rolling_{w}_mean_{col}'] = shifted_series.rolling(
                window=w, min_periods=ROLLING_MIN_PERIODS_MEAN
            ).mean()
            df_feat[f'rolling_{w}_var_{col}'] = shifted_series.rolling(
                window=w, min_periods=ROLLING_MIN_PERIODS_VAR
            ).var()
            df_feat[f'rolling_{w}_nobs_{col}'] = shifted_series.rolling(
                window=w, min_periods=1
            ).count()

    return df_feat

def build_features():
    print("Loading master series...")
    df = pd.read_parquet(MASTER_SERIES_PATH)
    
    # Base feature generation
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    
    # Generate Target 1 Features (HHS Care)
    print("Building Target 1 (HHS Care) features...")
    df_t1 = df.copy()
    df_t1.to_parquet(PROCESSED_DIR / 'features_target1.parquet', index=False)
    
    # Generate Target 2 Features (Discharged)
    print("Building Target 2 (Discharged) features...")
    df_t2 = df.copy()
    df_t2.to_parquet(PROCESSED_DIR / 'features_target2.parquet', index=False)
    
    # RF Drop-NaN Logging (Addendum Sec 2 & 11)
    # Count rows with NaNs in feature space (which ML models like RF can't handle natively)
    # For Target 1: Features are lags/rolling. The target itself is not dropped.
    # Exclude the raw flow columns since we use their lags for features, but 
    # if a flow column lag is NaN, the feature row is NaN.
    # We check how many feature rows (starting after max lag = 14) have NaNs.
    df_valid = df.iloc[14:]  # Drop first 14 rows due to lag 14
    
    # RF Feature columns (exclude raw non-target columns and date)
    exclude = [COL_DATE, 'parsed_date'] + NUMERIC_COLS + ['net_flow'] 
    rf_cols = [c for c in df.columns if c not in exclude]
    
    nan_rows_t1 = df_valid[rf_cols].isna().any(axis=1).sum()
    print(f"RF NaN Feature Rows (Target 1 context): {nan_rows_t1} / {len(df_valid)} dropped due to Flow gaps")
    
    print("Feature building complete.")

if __name__ == '__main__':
    build_features()
