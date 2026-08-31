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


# ----------------------------------------------------------------------
# Explicitly required by PRE_BUILD_TECHNICAL_ADDENDUM Section 3
# ----------------------------------------------------------------------
def test_the_two_named_off_template_fridays_are_present_as_standalone_rows(clean_data):
    """
    Addendum Section 2 builds the index as 767 canonical Sun-Thu template
    positions PLUS the 2 confirmed off-template Fridays, retained at their real
    dates. Section 3 requires this as a named test.
    """
    dates = pd.to_datetime(clean_data['parsed_date'])
    for friday in OFF_TEMPLATE_FRIDAYS:
        ts = pd.Timestamp(friday)
        assert (dates == ts).sum() == 1, f"{friday} missing or duplicated"
        row = clean_data[dates == ts].iloc[0]
        assert ts.dayofweek == 4, "the named date is not actually a Friday"
        assert not row['is_imputed'], f"{friday} must be a real observation, not a gap slot"

    on_template = dates.dt.dayofweek.isin(REPORTING_WEEKDAYS)
    assert int((~on_template).sum()) == len(OFF_TEMPLATE_FRIDAYS)
    assert int(on_template.sum()) == EXPECTED_TEMPLATE_POSITIONS


def test_period_position_counts_are_769_and_720(clean_data):
    assert len(clean_data) == EXPECTED_TOTAL_POSITIONS == 769
    assert int((~clean_data['is_imputed']).sum()) == EXPECTED_REAL_ROWS == 720
    assert int(clean_data['is_imputed'].sum()) == EXPECTED_GAP_COUNT == 49


def test_column_specific_imputation_flags_exist_and_are_correct(clean_data):
    """Addendum Section 2: stock columns flagged per-column as is_imputed_<column>."""
    for col in STOCK_COLS:
        flag = f'is_imputed_{col}'
        assert flag in clean_data.columns, f"missing required flag {flag}"
        # A stock value is invented exactly where the reporting slot has no row.
        assert clean_data[flag].equals(clean_data['is_imputed'])
    for col in FLOW_COLS:
        assert f'is_imputed_{col}' not in clean_data.columns, (
            "flow columns are true-missing and must not carry an imputation flag"
        )


def test_reindexing_loses_no_original_observation(raw_data, clean_data):
    """A date present in the raw CSV but absent from the template would be silently dropped."""
    real = raw_data[raw_data.notna().any(axis=1)]
    original = set(pd.to_datetime(real[COL_DATE], format=DATE_FORMAT))
    reindexed = set(pd.to_datetime(clean_data['parsed_date']))
    assert original - reindexed == set(), "reindexing dropped real observations"
    assert len(original) == EXPECTED_REAL_ROWS


def test_real_rows_carry_the_exact_published_values(raw_data, clean_data):
    """Cleaning must not alter a single published figure."""
    real = raw_data[raw_data.notna().any(axis=1)].copy()
    real['parsed_date'] = pd.to_datetime(real[COL_DATE], format=DATE_FORMAT)
    merged = clean_data.merge(real, on='parsed_date', suffixes=('_clean', '_raw'))
    assert len(merged) == EXPECTED_REAL_ROWS
    for col in NUMERIC_COLS:
        got = merged[f'{col}_clean'].astype(float)
        want = merged[f'{col}_raw'].astype(str).str.replace(',', '').astype(float)
        assert (got == want).all(), f"cleaning changed published values in {col}"


def test_flow_columns_are_never_filled_at_a_gap(clean_data):
    """Invariant 2: never zero-fill or interpolate a flow column at a gap slot."""
    for col in FLOW_COLS:
        assert clean_data.loc[clean_data['is_imputed'], col].isna().all()
        assert clean_data.loc[~clean_data['is_imputed'], col].notna().all()


def test_interpolated_stock_values_are_a_linear_blend_of_their_neighbours(clean_data):
    """
    Documents WHY an interpolated slot must never end a training window or be
    scored as an actual: its value is derived from the next REAL observation
    after it, which is future information at that position.
    """
    import numpy as np
    imputed = clean_data['is_imputed'].to_numpy(dtype=bool)
    y = clean_data[TARGET_1].astype(float).to_numpy()
    gaps = np.flatnonzero(imputed)
    checked = 0
    for p in gaps:
        before = np.flatnonzero(~imputed[:p])
        after = np.flatnonzero(~imputed[p:])
        if before.size == 0 or after.size == 0:
            continue
        lo = int(before[-1]); hi = p + int(after[0])
        expected = y[lo] + (y[hi] - y[lo]) * (p - lo) / (hi - lo)
        assert abs(y[p] - expected) <= 1.0, f"pos {p} is not a linear blend"
        assert hi > p, "the blend uses an observation AFTER the gap"
        checked += 1
    assert checked > 0


# ----------------------------------------------------------------------
# Provenance + fail-fast gate (addendum Sec. 3 and Sec. 9 Day-2 deliverables)
# ----------------------------------------------------------------------
def test_provenance_record_exists_and_matches_the_current_data(clean_data):
    from src.data.validate import hash_file, hash_frame, read_provenance

    rec = read_provenance()
    assert rec["raw_csv_sha256"] == hash_file(RAW_CSV_PATH)
    assert rec["master_series_sha256"] == hash_frame(clean_data)
    assert rec["n_period_positions"] == EXPECTED_TOTAL_POSITIONS
    assert rec["n_real_observations"] == EXPECTED_REAL_ROWS
    assert rec["n_gap_slots"] == EXPECTED_GAP_COUNT
    assert rec["data_as_of"] == "2025-12-21"
    assert rec["generated_at_utc"]


def test_frame_hash_is_deterministic_across_recomputation(clean_data):
    """A provenance hash that drifts between identical runs is useless."""
    from src.data.validate import hash_frame

    assert hash_frame(clean_data) == hash_frame(clean_data.copy())


def test_frame_hash_changes_when_a_single_value_changes(clean_data):
    from src.data.validate import hash_frame

    tampered = clean_data.copy()
    tampered.loc[0, TARGET_1] = int(tampered.loc[0, TARGET_1]) + 1
    assert hash_frame(tampered) != hash_frame(clean_data)


@pytest.mark.parametrize("break_it,reason", [
    (lambda d: d.iloc[:-1], "row count"),
    (lambda d: d.assign(**{COL_DISCHARGED: d[COL_DISCHARGED].fillna(0)}), "zero-filled flow column"),
    (lambda d: d.iloc[::-1], "reversed chronology"),
])
def test_fail_fast_gate_rejects_a_violating_series(clean_data, break_it, reason):
    """The gate must stop the pipeline, not let a bad artifact through."""
    from src.data.validate import validate_master_series

    with pytest.raises(ValueError):
        validate_master_series(break_it(clean_data.copy()))


def test_fail_fast_gate_accepts_the_real_series(clean_data):
    from src.data.validate import validate_master_series

    validate_master_series(clean_data)  # must not raise
