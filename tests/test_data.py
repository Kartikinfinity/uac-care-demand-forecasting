import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
from src.data.load import load_raw_data
from src.data.clean import clean_and_reindex_data

@pytest.fixture
def raw_data():
    return load_raw_data()

@pytest.fixture
def clean_data(raw_data):
    return clean_and_reindex_data(raw_data)

def test_schema_and_row_count(raw_data):
    # Schema check
    assert len(raw_data.columns) == 6
    assert list(raw_data.columns) == [COL_DATE] + NUMERIC_COLS
    
    # Row-count truncation (720 usable before reindex)
    has_data = raw_data.notna().any(axis=1)
    assert has_data.sum() == EXPECTED_REAL_ROWS

def test_missing_and_duplicates_in_usable(raw_data):
    has_data = raw_data.notna().any(axis=1)
    df_real = raw_data[has_data].copy()
    
    # Missing values
    assert df_real.isnull().sum().sum() == 0
    
    # Duplicates
    assert df_real.duplicated().sum() == 0
    assert df_real[COL_DATE].duplicated().sum() == 0

def test_numeric_validity_and_types(clean_data):
    # No negative values
    for col in NUMERIC_COLS:
        assert (clean_data[col].dropna() < 0).sum() == 0, f"Negative values found in {col}"
    
    # Stock columns shouldn't have NaNs after interpolation
    for col in STOCK_COLS:
        assert clean_data[col].isnull().sum() == 0
        
    # Flow columns should have exactly EXPECTED_GAP_COUNT NaNs
    for col in FLOW_COLS:
        assert clean_data[col].isnull().sum() == EXPECTED_GAP_COUNT

def test_time_continuity(clean_data):
    # Reindexed series has expected row count
    assert len(clean_data) == EXPECTED_TOTAL_POSITIONS
    
    # Check that is_imputed flags exactly the gaps
    assert clean_data['is_imputed'].sum() == EXPECTED_GAP_COUNT
    
    # Check that dates are strictly ascending
    assert clean_data['parsed_date'].is_monotonic_increasing
