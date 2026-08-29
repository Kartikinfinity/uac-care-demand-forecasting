import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.features.build_features import build_features
from src.config import DATA_PROCESSED_DIR, COL_HHS_CARE, COL_DISCHARGED

@pytest.fixture(scope="module")
def feature_tables():
    # Build features to ensure parquet files exist
    build_features()
    t1_path = DATA_PROCESSED_DIR / 'features_target1.parquet'
    t2_path = DATA_PROCESSED_DIR / 'features_target2.parquet'
    
    assert t1_path.exists()
    assert t2_path.exists()
    
    df1 = pd.read_parquet(t1_path)
    df2 = pd.read_parquet(t2_path)
    return df1, df2

def test_feature_lag_strict_backward(feature_tables):
    """Ensure lag-7 correctness is row-based and exactly 7 periods backward."""
    df1, _ = feature_tables
    
    # Pick a random robust row index > 14
    idx = 50
    
    # The lag 7 value at row 50 should EXACTLY equal the target value at row 43
    actual_val = df1.loc[idx - 7, COL_HHS_CARE]
    lag_val = df1.loc[idx, f'lag_7_{COL_HHS_CARE}']
    
    assert actual_val == lag_val, f"Leakage/Misalignment in lag! {actual_val} != {lag_val}"

def test_no_current_row_leakage_in_rolling(feature_tables):
    """Invariant: Never include the current row in its own rolling-window feature."""
    df1, _ = feature_tables
    
    idx = 60
    # Rolling 7 mean at row 60 should be the mean of rows 53, 54, 55, 56, 57, 58, 59
    # It must NOT include row 60.
    expected_mean = df1.loc[idx-7 : idx-1, COL_HHS_CARE].mean()
    rolling_val = df1.loc[idx, f'rolling_7_mean_{COL_HHS_CARE}']
    
    # Use np.isclose for float comparison
    assert np.isclose(expected_mean, rolling_val), f"Leakage in rolling feature! {expected_mean} != {rolling_val}"

def test_holiday_proximity_logic(feature_tables):
    """Verify that July 4 is flagged correctly."""
    df1, _ = feature_tables
    
    # 2024-07-04 is a Thursday. 
    # Because of our 2-day proximity, 2024-07-02, 07-03, 07-04, 07-07, 07-08 might be flagged depending on weekends.
    df1['parsed_date'] = pd.to_datetime(df1['parsed_date'])
    july4 = df1[df1['parsed_date'] == '2024-07-04']
    
    if not july4.empty:
        assert july4.iloc[0]['is_near_holiday'] == 1
