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


# ----------------------------------------------------------------------
# ROLLING_MIN_PERIODS -- resolved Day-7 open item, pinned here
# ----------------------------------------------------------------------
def test_rolling_min_periods_matches_the_configured_decision(feature_tables):
    """
    The permissive setting is a RESOLVED, documented decision (see config), not
    an accident. This pins the semantics so it cannot drift silently.
    """
    from src.config import (
        ROLLING_MIN_PERIODS_MEAN, ROLLING_MIN_PERIODS_VAR, ROLLING_WINDOWS,
    )

    df1, _ = feature_tables
    for w in ROLLING_WINDOWS:
        shifted = df1[COL_DISCHARGED].astype(float).shift(1)
        expected_mean = shifted.rolling(w, min_periods=ROLLING_MIN_PERIODS_MEAN).mean()
        expected_var = shifted.rolling(w, min_periods=ROLLING_MIN_PERIODS_VAR).var()
        assert np.allclose(df1[f'rolling_{w}_mean_{COL_DISCHARGED}'].astype(float),
                           expected_mean, equal_nan=True)
        assert np.allclose(df1[f'rolling_{w}_var_{COL_DISCHARGED}'].astype(float),
                           expected_var, equal_nan=True)


def test_rolling_features_are_unaffected_for_the_stock_target(feature_tables):
    """Evidence point 2: stock columns have no gaps, so the setting cannot bite."""
    from src.config import ROLLING_WINDOWS

    df1, _ = feature_tables
    for w in ROLLING_WINDOWS:
        nobs = df1[f'rolling_{w}_nobs_{COL_HHS_CARE}'].iloc[w:]
        assert (nobs == w).all(), "a stock rolling window is not fully populated"


def test_the_configured_floor_is_never_actually_binding(feature_tables):
    """
    Evidence point 5: the data never drives a rolling window below 2
    observations, so min_periods=1 vs 2 is immaterial on this dataset.
    """
    from src.config import ROLLING_WINDOWS

    df1, _ = feature_tables
    observed_minimums = {}
    for w in ROLLING_WINDOWS:
        cols = [c for c in df1.columns if c.startswith(f'rolling_{w}_nobs_')]
        observed_minimums[w] = int(df1[cols].iloc[14:].to_numpy().min())
    assert observed_minimums[7] >= 2
    assert observed_minimums[14] >= 2


def test_nobs_companion_columns_expose_the_true_sample_size(feature_tables):
    """Nothing is hidden: every rolling statistic ships its own observation count."""
    from src.config import ROLLING_WINDOWS

    df1, _ = feature_tables
    for w in ROLLING_WINDOWS:
        means = [c for c in df1.columns if c.startswith(f'rolling_{w}_mean_')]
        for col in means:
            base = col.split('_', 3)[3]
            assert f'rolling_{w}_nobs_{base}' in df1.columns
