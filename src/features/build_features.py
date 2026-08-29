import pandas as pd
import numpy as np
from pathlib import Path
import sys
import holidays

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    MASTER_SERIES_PATH, NUMERIC_COLS, TARGET_1, TARGET_2,
    COL_TRANSFERRED, COL_DISCHARGED, DATA_PROCESSED_DIR, COL_DATE
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
    
    # Check if current date, date+1, date+2, date-1, date-2 is a holiday
    # We use a 2-day proximity
    def is_near_holiday(d):
        for offset in range(-2, 3):
            if (d + pd.Timedelta(days=offset)) in us_holidays:
                return 1
        return 0

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
    
    # 2. Lags (t-1, t-7, t-14)
    lags = [1, 7, 14]
    for col in cols_to_lag:
        for lag in lags:
            df_feat[f'lag_{lag}_{col}'] = df_feat[col].shift(lag)
            
    # 3. Rolling Windows (7, 14)
    windows = [7, 14]
    for col in cols_to_lag:
        for w in windows:
            # Shift by 1 first to prevent current-row leakage
            shifted_series = df_feat[col].shift(1)
            # Use min_periods=1 to allow early predictions, or strict w?
            # Roadmap implies we will evaluate on robust data, but min_periods=1 is safe.
            # We'll use strict w to be rigorously correct about window sizes.
            df_feat[f'rolling_{w}_mean_{col}'] = shifted_series.rolling(window=w, min_periods=1).mean()
            df_feat[f'rolling_{w}_var_{col}'] = shifted_series.rolling(window=w, min_periods=2).var()
            
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
    total_rows = len(df)
    
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
