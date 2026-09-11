"""
clean.py -- Build the one trustworthy master series (Day 2).

Everything downstream reads the output of this module, so every judgement made
about the data is made here and recorded rather than scattered.

Four things happen, in order:

  1. **Truncate the blank tail.** The delivered file carries 450 empty trailing
     rows after the 720 real observations.
  2. **Parse and type.** `Children in HHS Care` arrives string-typed because its
     values carry thousands-separator commas; it is parsed to a nullable integer
     rather than coerced through float, so no value is silently rounded.
  3. **Reindex onto the reporting calendar.** Reporting runs Sunday-Thursday, so
     the series is placed on the true Sun-Thu schedule rather than a daily one,
     which would invent weekend rows that were never meant to exist. Positions
     with no published observation become explicit gap slots.
  4. **Fill gaps according to what the series IS.** This is the invariant that
     matters most:

        * A STOCK (`Children in HHS Care`, `Children in CBP custody`) exists
          continuously and is merely unobserved on a non-reporting day, so
          interpolating it estimates something real.
        * A FLOW (discharges, transfers, apprehensions) counts events WITHIN a
          period. On a day with no report there is no count to estimate, and
          interpolating one would fabricate events that never happened.

     Flows are therefore left genuinely missing, without exception.

Every imputed value carries a per-column `is_imputed_*` flag, and those flags are
load-bearing rather than documentary: the walk-forward harness uses them to stop
training on an interpolated origin (which blends values from both sides of
itself, including future ones) and to exclude interpolated actuals from scoring
(which would measure agreement with the interpolation, not accuracy).
"""
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
from src.data.validate import (
    validate_master_series,
    build_provenance,
    write_provenance,
)

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
    
    # Fail-fast gate + data hash (addendum Sec. 9, Day 2). Validation runs BEFORE
    # anything is written, so a series that violates the frozen specification can
    # never reach an artifact that downstream days would trust.
    print("Validating master series against the frozen specification...")
    validate_master_series(df_master)

    print(f"Saving to {MASTER_SERIES_PATH}...")
    df_master.to_parquet(MASTER_SERIES_PATH, index=False)

    record = build_provenance(RAW_CSV_PATH, df_master)
    write_provenance(record)
    print(f"Provenance written: raw_csv_sha256={record['raw_csv_sha256'][:12]}... "
          f"master_series_sha256={record['master_series_sha256'][:12]}... "
          f"data_as_of={record['data_as_of']}")
    print("Master series generated successfully.")

    return df_master

if __name__ == '__main__':
    generate_master_series()
