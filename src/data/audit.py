"""
Day 1 — Data Audit Script

Reproduces every finding from the Source Verification Note.
All numbers are computed from the actual CSV, never fabricated.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def run_audit(csv_path=None):
    """Run the complete data audit and return findings dict."""
    if csv_path is None:
        csv_path = project_root / "HHS_Unaccompanied_Alien_Children_Program (1).csv"
    
    print("=" * 70)
    print("UAC FORECASTING PROJECT — DAY 1 DATA AUDIT")
    print("=" * 70)
    print(f"\nSource file: {csv_path}")
    print(f"File exists: {csv_path.exists()}")
    
    findings = {}
    
    # ── 1. Load raw data ──
    df_raw = pd.read_csv(csv_path)
    findings['raw_rows'] = len(df_raw)
    findings['raw_cols'] = len(df_raw.columns)
    findings['column_names'] = list(df_raw.columns)
    
    print(f"\n{'─'*70}")
    print("1. RAW FILE STRUCTURE")
    print(f"{'─'*70}")
    print(f"   Total rows in file: {findings['raw_rows']}")
    print(f"   Total columns: {findings['raw_cols']}")
    print(f"   Column names: {findings['column_names']}")
    print(f"   Column dtypes:")
    for col in df_raw.columns:
        print(f"      {col}: {df_raw[col].dtype}")
    findings['raw_dtypes'] = {col: str(df_raw[col].dtype) for col in df_raw.columns}
    
    # ── 2. Identify usable vs blank rows ──
    has_data = df_raw.notna().any(axis=1)
    usable_rows = has_data.sum()
    blank_rows = (~has_data).sum()
    findings['usable_rows'] = int(usable_rows)
    findings['blank_rows'] = int(blank_rows)
    
    # Check that blank rows form a contiguous tail
    first_blank_idx = None
    for i in range(len(has_data)):
        if not has_data.iloc[i]:
            first_blank_idx = i
            break
    if first_blank_idx is not None:
        all_blank_after = (~has_data.iloc[first_blank_idx:]).all()
    else:
        all_blank_after = True
    findings['blank_tail_contiguous'] = bool(all_blank_after)
    
    print(f"\n{'─'*70}")
    print("2. USABLE vs BLANK ROWS")
    print(f"{'─'*70}")
    print(f"   Rows with data: {usable_rows}")
    print(f"   Blank trailing rows: {blank_rows}")
    print(f"   Blank block is contiguous tail: {all_blank_after}")
    if first_blank_idx is not None:
        print(f"   First blank row index: {first_blank_idx}")
    
    # ── 3. Work with usable rows only ──
    df = df_raw.iloc[:int(usable_rows)].copy()
    
    # ── 4. Date analysis ──
    # Parse dates
    df['parsed_date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')
    date_min = df['parsed_date'].min()
    date_max = df['parsed_date'].max()
    findings['date_min'] = str(date_min.date())
    findings['date_max'] = str(date_max.date())
    findings['calendar_days_spanned'] = (date_max - date_min).days + 1
    
    print(f"\n{'─'*70}")
    print("3. DATE ANALYSIS")
    print(f"{'─'*70}")
    print(f"   Date range: {date_min.date()} to {date_max.date()}")
    print(f"   Calendar days spanned: {findings['calendar_days_spanned']}")
    
    # Sort order check
    is_descending = df['parsed_date'].is_monotonic_decreasing
    findings['raw_sort_descending'] = bool(is_descending)
    print(f"   Raw file sort order: {'Descending' if is_descending else 'NOT descending'}")
    
    # Day-of-week distribution
    dow_counts = df['parsed_date'].dt.dayofweek.value_counts().sort_index()
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    findings['dow_distribution'] = {dow_names[k]: int(v) for k, v in dow_counts.items()}
    
    print(f"   Day-of-week distribution:")
    for dow_idx in range(7):
        name = dow_names[dow_idx]
        count = int(dow_counts.get(dow_idx, 0))
        print(f"      {name}: {count}")
    
    # ── 5. Missing values in usable rows ──
    nulls_per_col = df.drop(columns=['parsed_date']).isnull().sum()
    total_nulls = int(nulls_per_col.sum())
    findings['nulls_in_usable'] = total_nulls
    
    print(f"\n{'─'*70}")
    print("4. MISSING VALUES (within {usable_rows} usable rows)")
    print(f"{'─'*70}")
    print(f"   Total nulls: {total_nulls}")
    for col, n in nulls_per_col.items():
        print(f"      {col}: {n}")
    
    # ── 6. Duplicates ──
    dup_dates = df['parsed_date'].duplicated().sum()
    dup_rows = df.drop(columns=['parsed_date']).duplicated().sum()
    findings['duplicate_dates'] = int(dup_dates)
    findings['duplicate_rows'] = int(dup_rows)
    
    print(f"\n{'─'*70}")
    print("5. DUPLICATES")
    print(f"{'─'*70}")
    print(f"   Duplicate dates: {dup_dates}")
    print(f"   Duplicate full rows: {dup_rows}")
    
    # ── 7. Children in HHS Care dtype check ──
    hhs_col = 'Children in HHS Care'
    hhs_dtype = str(df_raw[hhs_col].dtype)
    has_commas = df[hhs_col].astype(str).str.contains(',').any()
    findings['hhs_care_dtype'] = hhs_dtype
    findings['hhs_care_has_commas'] = bool(has_commas)
    
    print(f"\n{'─'*70}")
    print("6. CHILDREN IN HHS CARE — TYPE CHECK")
    print(f"{'─'*70}")
    print(f"   Raw dtype: {hhs_dtype}")
    print(f"   Contains thousands-separator commas: {has_commas}")
    
    # Parse HHS Care for numeric analysis
    hhs_numeric = df[hhs_col].astype(str).str.replace(',', '').astype(float)
    findings['hhs_care_min'] = float(hhs_numeric.min())
    findings['hhs_care_max'] = float(hhs_numeric.max())
    findings['hhs_care_mean'] = float(hhs_numeric.mean())
    findings['hhs_care_std'] = float(hhs_numeric.std())
    findings['hhs_care_range_ratio'] = float(hhs_numeric.max() / hhs_numeric.min())
    
    print(f"   Min: {hhs_numeric.min():.0f}")
    print(f"   Max: {hhs_numeric.max():.0f}")
    print(f"   Mean: {hhs_numeric.mean():.0f}")
    print(f"   Std: {hhs_numeric.std():.0f}")
    print(f"   Range ratio (max/min): {hhs_numeric.max()/hhs_numeric.min():.1f}×")
    
    # ── 8. Numeric column analysis ──
    # Sort ascending for temporal analysis
    df_sorted = df.sort_values('parsed_date').reset_index(drop=True)
    
    # Parse all numeric columns
    numeric_cols = [
        'Children apprehended and placed in CBP custody*',
        'Children in CBP custody',
        'Children transferred out of CBP custody',
        'Children in HHS Care',
        'Children discharged from HHS Care'
    ]
    
    for col in numeric_cols:
        df_sorted[col + '_num'] = df_sorted[col].astype(str).str.replace(',', '').astype(float)
    
    print(f"\n{'─'*70}")
    print("7. NUMERIC COLUMN ANALYSIS")
    print(f"{'─'*70}")
    
    zero_counts = {}
    neg_counts = {}
    for col in numeric_cols:
        num_col = col + '_num'
        zeros = int((df_sorted[num_col] == 0).sum())
        negs = int((df_sorted[num_col] < 0).sum())
        zero_counts[col] = zeros
        neg_counts[col] = negs
        print(f"   {col}:")
        print(f"      Min: {df_sorted[num_col].min():.0f}, Max: {df_sorted[num_col].max():.0f}")
        print(f"      Zero-value days: {zeros}")
        print(f"      Negative values: {negs}")
    
    findings['zero_counts'] = zero_counts
    findings['negative_counts'] = neg_counts
    findings['any_negative'] = any(v > 0 for v in neg_counts.values())
    
    # ── 9. Flow identity check ──
    hhs_care = df_sorted['Children in HHS Care_num'].values
    transferred = df_sorted['Children transferred out of CBP custody_num'].values
    discharged = df_sorted['Children discharged from HHS Care_num'].values
    
    # Check: HHS_Care[t] ≈ HHS_Care[t-1] + Transferred[t] - Discharged[t]
    expected_hhs = hhs_care[:-1] + transferred[1:] - discharged[1:]
    actual_hhs = hhs_care[1:]
    residuals = actual_hhs - expected_hhs
    exact_matches = int(np.sum(residuals == 0))
    total_checks = len(residuals)
    exact_match_pct = exact_matches / total_checks * 100
    
    findings['flow_identity_exact_matches'] = exact_matches
    findings['flow_identity_total_checks'] = total_checks
    findings['flow_identity_exact_match_pct'] = float(exact_match_pct)
    findings['flow_identity_residual_median'] = float(np.median(residuals))
    findings['flow_identity_residual_min'] = float(np.min(residuals))
    findings['flow_identity_residual_max'] = float(np.max(residuals))
    findings['flow_identity_residual_mean'] = float(np.mean(residuals))
    
    print(f"\n{'─'*70}")
    print("8. FLOW IDENTITY CHECK")
    print(f"   HHS_Care[t] ≈ HHS_Care[t-1] + Transferred[t] - Discharged[t]")
    print(f"{'─'*70}")
    print(f"   Exact matches: {exact_matches}/{total_checks} ({exact_match_pct:.1f}%)")
    print(f"   Residual median: {np.median(residuals):.1f}")
    print(f"   Residual mean: {np.mean(residuals):.1f}")
    print(f"   Residual min: {np.min(residuals):.0f}")
    print(f"   Residual max: {np.max(residuals):.0f}")
    print(f"   Residual std: {np.std(residuals):.1f}")
    
    # ── 10. Gap analysis (relative to Sun-Thu schedule) ──
    dates_sorted = pd.DatetimeIndex(df_sorted['parsed_date'])
    
    # Build expected Sun-Thu schedule between min and max date
    full_range = pd.date_range(start=dates_sorted.min(), end=dates_sorted.max(), freq='D')
    # Sun=6, Mon=0, Tue=1, Wed=2, Thu=3
    expected_schedule = full_range[full_range.dayofweek.isin([6, 0, 1, 2, 3])]
    
    # Find gaps
    actual_set = set(dates_sorted)
    expected_set = set(expected_schedule)
    
    # Dates in actual but not in expected (off-schedule: Fridays)
    off_schedule = actual_set - expected_set
    # Dates in expected but not in actual (gaps)
    gaps = expected_set - actual_set
    
    findings['expected_schedule_count'] = len(expected_schedule)
    findings['gap_count'] = len(gaps)
    findings['off_schedule_count'] = len(off_schedule)
    findings['off_schedule_dates'] = sorted([str(d.date()) for d in off_schedule])
    
    print(f"\n{'─'*70}")
    print("9. SCHEDULE GAP ANALYSIS (Sun-Thu)")
    print(f"{'─'*70}")
    print(f"   Expected Sun-Thu slots in date range: {len(expected_schedule)}")
    print(f"   Actual observations: {len(dates_sorted)}")
    print(f"   Gaps (expected but missing): {len(gaps)}")
    print(f"   Off-schedule observations (Fri/Sat): {len(off_schedule)}")
    if off_schedule:
        print(f"   Off-schedule dates: {findings['off_schedule_dates']}")
    
    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("AUDIT SUMMARY")
    print(f"{'='*70}")
    
    checks = [
        ("Raw rows = 1170", findings['raw_rows'] == 1170),
        ("Usable rows = 720", findings['usable_rows'] == 720),
        ("Blank trailing rows = 450", findings['blank_rows'] == 450),
        ("Blank tail is contiguous", findings['blank_tail_contiguous']),
        ("Columns = 6", findings['raw_cols'] == 6),
        ("No nulls in usable rows", findings['nulls_in_usable'] == 0),
        ("No duplicate dates", findings['duplicate_dates'] == 0),
        ("No duplicate rows", findings['duplicate_rows'] == 0),
        ("HHS Care is string/object type", 'object' in findings['hhs_care_dtype']),
        ("HHS Care has commas", findings['hhs_care_has_commas']),
        ("No negative values", not findings['any_negative']),
        ("Raw file is descending", findings['raw_sort_descending']),
    ]
    
    all_pass = True
    for label, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"   [{status}] {label}")
    
    print(f"\n   Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*70}")
    
    return findings


if __name__ == "__main__":
    findings = run_audit()
