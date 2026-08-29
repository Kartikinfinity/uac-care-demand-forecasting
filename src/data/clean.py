import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    RAW_CSV_PATH, MASTER_SERIES_PATH, COL_DATE, NUMERIC_COLS, 
    STOCK_COLS, FLOW_COLS, EXPECTED_REAL_ROWS, DATE_FORMAT,
    REPORTING_WEEKDAYS, OFF_TEMPLATE_FRIDAYS, EXPECTED_TOTAL_POSITIONS
)
from src.data.load import load_raw_data

def clean_and_reindex_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe and reindexes to canonical schedule.
    Implementation invariants observed:
    - Never zero-fill or interpolate a flow column at a gap slot (true-missing only).
    """
    # 1. Truncate blank trailing rows
    has_data = df.notna().any(axis=1)
    df_clean = df[has_data].copy()
    assert len(df_clean) == EXPECTED_REAL_ROWS, f"Expected {EXPECTED_REAL_ROWS} real rows, got {len(df_clean)}"
    
    # 2. Parse dates
    df_clean['parsed_date'] = pd.to_datetime(df_clean[COL_DATE], format=DATE_FORMAT)
    
    # 3. Strip commas and cast numeric columns to float, then Int64 (to handle NaNs for flows)
    for col in NUMERIC_COLS:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '').astype(float)
        # Using nullable integer type 'Int64' to allow NaNs safely
        df_clean[col] = df_clean[col].astype('Int64')
        
    # 4. Sort ascending
    df_clean = df_clean.sort_values('parsed_date').reset_index(drop=True)
    
    # 5. Build true schedule index
    date_min = df_clean['parsed_date'].min()
    date_max = df_clean['parsed_date'].max()
    
    # Base calendar
    full_range = pd.date_range(start=date_min, end=date_max, freq='D')
    # Filter to Sun-Thu
    expected_schedule = full_range[full_range.dayofweek.isin(REPORTING_WEEKDAYS)]
    
    # Add off-template Fridays
    off_template = pd.to_datetime(OFF_TEMPLATE_FRIDAYS)
    
    # Combine and sort master index
    master_index = expected_schedule.union(off_template).sort_values()
    assert len(master_index) == EXPECTED_TOTAL_POSITIONS, f"Expected {EXPECTED_TOTAL_POSITIONS} slots, got {len(master_index)}"
    
    # 6. Reindex and align
    df_clean = df_clean.set_index('parsed_date')
    
    # Flag which dates were in the original data
    original_dates = set(df_clean.index)
    
    # Reindex
    df_master = df_clean.reindex(master_index)
    
    # Reset index to make date a column again
    df_master = df_master.reset_index().rename(columns={'index': 'parsed_date'})
    
    # Create imputation flags
    df_master['is_imputed'] = ~df_master['parsed_date'].isin(original_dates)
    for col in STOCK_COLS:
        df_master[f'is_imputed_{col}'] = df_master['is_imputed']
    
    # Restore original Date string for consistency
    df_master[COL_DATE] = df_master['parsed_date'].dt.strftime(DATE_FORMAT)
    
    # 7. Interpolate STOCK columns ONLY
    for col in STOCK_COLS:
        # Linear interpolation (requires temporary cast to float)
        df_master[col] = df_master[col].astype(float).interpolate(method='linear').round().astype('Int64')
        
    # FLOW columns remain NaN where is_imputed is True
    # (they are already NaN because of reindex, and we don't touch them)
    
    return df_master

def generate_master_series():
    """Load, clean, reindex, and save to Parquet."""
    print("Loading raw data...")
    df_raw = load_raw_data()
    print("Cleaning and reindexing...")
    df_master = clean_and_reindex_data(df_raw)
    
    # Ensure interim dir exists
    MASTER_SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {MASTER_SERIES_PATH}...")
    df_master.to_parquet(MASTER_SERIES_PATH, index=False)
    print("Master series generated successfully.")
    
    return df_master

if __name__ == '__main__':
    generate_master_series()
