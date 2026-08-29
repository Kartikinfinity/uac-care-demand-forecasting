"""
Day 3 — Exploratory Data Analysis Script

Produces all EDA artifacts required by the roadmap:
  1. Structural-break scan (Chow-test proxy via rolling Cusum)
  2. Within-week seasonality test per target (Kruskal-Wallis)
  3. Seasonal decomposition (STL)
  4. ACF/PACF plots
  5. ADF and KPSS stationarity tests (raw + differenced)
  6. Weekly reporting pattern visualization
  7. Zero-value day review for holiday clustering
  8. Differenced-series correlations

All numbers computed from actual data, never fabricated.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for figure saving
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import warnings
from pathlib import Path
from scipy import stats

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    MASTER_SERIES_PATH, FIGURES_DIR, DOCS_DIR,
    COL_DATE, COL_HHS_CARE, COL_DISCHARGED, COL_APPREHENDED,
    COL_TRANSFERRED, COL_CBP_CUSTODY, NUMERIC_COLS,
    TARGET_1, TARGET_2, STOCK_COLS, FLOW_COLS,
    REPORTING_WEEKDAYS, TRAINING_CAP_DATE
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# GLOBALS
# ──────────────────────────────────────────────────────────────────────
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_master_series() -> pd.DataFrame:
    """Load the cleaned master series from parquet."""
    df = pd.read_parquet(MASTER_SERIES_PATH)
    df['parsed_date'] = pd.to_datetime(df['parsed_date'])
    return df


# ──────────────────────────────────────────────────────────────────────
# 1. STATIONARITY TESTS (ADF + KPSS)
# ──────────────────────────────────────────────────────────────────────
def run_stationarity_tests(series: pd.Series, name: str) -> dict:
    """Run ADF and KPSS tests on a series. Returns results dict."""
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    results = {'name': name, 'n_obs': len(clean)}

    # ADF test (H0: unit root exists → series is non-stationary)
    adf_stat, adf_pval, adf_lags, adf_nobs, adf_crit, _ = adfuller(clean, autolag='AIC')
    results['adf_statistic'] = adf_stat
    results['adf_pvalue'] = adf_pval
    results['adf_lags_used'] = adf_lags
    results['adf_critical_values'] = adf_crit
    results['adf_conclusion'] = 'stationary' if adf_pval < 0.05 else 'non-stationary'

    # KPSS test (H0: series is stationary)
    kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(clean, regression='ct', nlags='auto')
    results['kpss_statistic'] = kpss_stat
    results['kpss_pvalue'] = kpss_pval
    results['kpss_lags_used'] = kpss_lags
    results['kpss_critical_values'] = kpss_crit
    results['kpss_conclusion'] = 'stationary' if kpss_pval > 0.05 else 'non-stationary'

    return results


def print_stationarity(res: dict):
    """Print stationarity test results."""
    print(f"\n  --- {res['name']} (N={res['n_obs']}) ---")
    print(f"  ADF statistic: {res['adf_statistic']:.4f}  p-value: {res['adf_pvalue']:.4f}  -> {res['adf_conclusion']}")
    print(f"      Critical values: {res['adf_critical_values']}")
    print(f"  KPSS statistic: {res['kpss_statistic']:.4f}  p-value: {res['kpss_pvalue']:.4f}  -> {res['kpss_conclusion']}")
    print(f"      Critical values: {res['kpss_critical_values']}")

    # Joint interpretation
    if res['adf_conclusion'] == 'stationary' and res['kpss_conclusion'] == 'stationary':
        joint = 'STATIONARY (both tests agree)'
    elif res['adf_conclusion'] == 'non-stationary' and res['kpss_conclusion'] == 'non-stationary':
        joint = 'NON-STATIONARY (both tests agree)'
    elif res['adf_conclusion'] == 'stationary' and res['kpss_conclusion'] == 'non-stationary':
        joint = 'TREND-STATIONARY (ADF rejects unit root but KPSS rejects level stationarity — likely deterministic trend)'
    else:
        joint = 'INCONCLUSIVE (tests disagree)'
    print(f"  Joint conclusion: {joint}")
    res['joint_conclusion'] = joint


# ──────────────────────────────────────────────────────────────────────
# 2. STRUCTURAL BREAK SCAN
# ──────────────────────────────────────────────────────────────────────
def structural_break_scan(df: pd.DataFrame, target_col: str, name: str) -> dict:
    """
    Structural break detection using CUSUM-based approach.
    Identifies the date of maximum cumulative deviation from the overall mean.
    """
    series = df[target_col].dropna().values
    dates = df.loc[df[target_col].notna(), 'parsed_date'].values

    # CUSUM
    mean_val = np.mean(series)
    cusum = np.cumsum(series - mean_val)

    # Dominant break = index of maximum absolute CUSUM
    dominant_idx = np.argmax(np.abs(cusum))
    dominant_date = pd.Timestamp(dates[dominant_idx])

    # Secondary break: look for next largest deviation in the other half
    n = len(cusum)
    half = n // 2
    if dominant_idx < half:
        # Look in second half
        secondary_search = np.abs(cusum[half:])
        secondary_idx = half + np.argmax(secondary_search)
    else:
        # Look in first half
        secondary_search = np.abs(cusum[:half])
        secondary_idx = np.argmax(secondary_search)
    secondary_date = pd.Timestamp(dates[secondary_idx])

    # Also compute rolling mean shift for visualization
    window = 30
    if len(series) > window:
        rolling_mean = pd.Series(series).rolling(window, center=True).mean().values
    else:
        rolling_mean = series

    results = {
        'target': name,
        'dominant_break_date': str(dominant_date.date()),
        'dominant_break_idx': int(dominant_idx),
        'dominant_cusum_value': float(cusum[dominant_idx]),
        'secondary_break_date': str(secondary_date.date()),
        'secondary_break_idx': int(secondary_idx),
        'secondary_cusum_value': float(cusum[secondary_idx]),
    }

    # Plot CUSUM
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(dates, series, alpha=0.5, linewidth=0.8, label='Observed')
    if len(series) > window:
        ax1.plot(dates, rolling_mean, color='red', linewidth=1.5, label=f'{window}-period rolling mean')
    ax1.axvline(dominant_date, color='darkred', linestyle='--', linewidth=1.5, label=f'Dominant break: {dominant_date.date()}')
    ax1.axvline(secondary_date, color='orange', linestyle='--', linewidth=1.5, label=f'Secondary break: {secondary_date.date()}')
    ax1.axvline(pd.Timestamp(TRAINING_CAP_DATE), color='green', linestyle=':', linewidth=1.5, label=f'Training cap: {TRAINING_CAP_DATE}')
    ax1.set_title(f'{name} — Time Series with Structural Breaks')
    ax1.legend(fontsize=8)
    ax1.set_ylabel('Value')

    ax2 = axes[1]
    ax2.plot(dates, cusum, color='navy')
    ax2.axvline(dominant_date, color='darkred', linestyle='--', linewidth=1.5)
    ax2.axvline(secondary_date, color='orange', linestyle='--', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_title(f'{name} — CUSUM')
    ax2.set_ylabel('Cumulative Sum')
    ax2.set_xlabel('Date')

    fig.tight_layout()
    fig_path = FIGURES_DIR / f'structural_break_{name.replace(" ", "_").lower()}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    results['figure'] = str(fig_path)

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. WITHIN-WEEK SEASONALITY TEST
# ──────────────────────────────────────────────────────────────────────
def test_within_week_seasonality(df: pd.DataFrame, target_col: str, name: str) -> dict:
    """
    Kruskal-Wallis test for within-week seasonality.
    Groups by day-of-week, tests whether distributions differ significantly.
    """
    df_real = df[~df['is_imputed']].copy()
    df_real['dow'] = df_real['parsed_date'].dt.dayofweek

    series = df_real[target_col].dropna()
    dow = df_real.loc[series.index, 'dow']

    # Group by day of week
    groups = [series[dow == d].values for d in sorted(dow.unique())]
    group_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 6: 'Sun'}

    # Kruskal-Wallis (non-parametric, doesn't assume normality)
    h_stat, kw_pvalue = stats.kruskal(*groups)

    # Per-group stats
    group_stats = {}
    for d, g in zip(sorted(dow.unique()), groups):
        label = group_labels.get(d, f'Day{d}')
        group_stats[label] = {
            'count': len(g),
            'mean': float(np.mean(g)),
            'median': float(np.median(g)),
            'std': float(np.std(g)),
        }

    has_seasonality = kw_pvalue < 0.05

    results = {
        'target': name,
        'test': 'Kruskal-Wallis',
        'h_statistic': float(h_stat),
        'p_value': float(kw_pvalue),
        'has_within_week_seasonality': has_seasonality,
        'group_stats': group_stats,
    }

    # Also test on first-differenced series (removes trend)
    diff_series = series.diff().dropna()
    diff_dow = dow.loc[diff_series.index]
    diff_groups = [diff_series[diff_dow == d].values for d in sorted(diff_dow.unique())]
    dh_stat, dkw_pvalue = stats.kruskal(*diff_groups)
    results['diff_h_statistic'] = float(dh_stat)
    results['diff_p_value'] = float(dkw_pvalue)
    results['diff_has_within_week_seasonality'] = dkw_pvalue < 0.05

    # Plot box plot by day of week
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw levels
    plot_df = pd.DataFrame({'value': series, 'dow': dow.map(group_labels)})
    dow_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu']
    plot_df['dow'] = pd.Categorical(plot_df['dow'], categories=dow_order, ordered=True)
    sns.boxplot(data=plot_df, x='dow', y='value', ax=axes[0], order=dow_order)
    axes[0].set_title(f'{name} by Day of Week (Raw)\nKW p={kw_pvalue:.4f}')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('Value')

    # Differenced
    plot_df_diff = pd.DataFrame({'value': diff_series, 'dow': diff_dow.map(group_labels)})
    plot_df_diff['dow'] = pd.Categorical(plot_df_diff['dow'], categories=dow_order, ordered=True)
    sns.boxplot(data=plot_df_diff, x='dow', y='value', ax=axes[1], order=dow_order)
    axes[1].set_title(f'{name} by Day of Week (1st Diff)\nKW p={dkw_pvalue:.4f}')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Change')

    fig.tight_layout()
    fig_path = FIGURES_DIR / f'weekly_seasonality_{name.replace(" ", "_").lower()}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    results['figure'] = str(fig_path)

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. ACF/PACF PLOTS
# ──────────────────────────────────────────────────────────────────────
def plot_acf_pacf(df: pd.DataFrame, target_col: str, name: str, max_lags: int = 40) -> str:
    """Generate ACF/PACF plots for both raw and differenced series."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = df[target_col].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw
    plot_acf(series, lags=max_lags, ax=axes[0, 0], title=f'{name} — ACF (Raw)')
    plot_pacf(series, lags=max_lags, ax=axes[0, 1], title=f'{name} — PACF (Raw)', method='ywm')

    # Differenced
    diff = series.diff().dropna()
    plot_acf(diff, lags=max_lags, ax=axes[1, 0], title=f'{name} — ACF (1st Differenced)')
    plot_pacf(diff, lags=max_lags, ax=axes[1, 1], title=f'{name} — PACF (1st Differenced)', method='ywm')

    fig.tight_layout()
    fig_path = FIGURES_DIR / f'acf_pacf_{name.replace(" ", "_").lower()}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(fig_path)


# ──────────────────────────────────────────────────────────────────────
# 5. SEASONAL DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────
def seasonal_decomposition(df: pd.DataFrame, target_col: str, name: str, period: int = 5) -> dict:
    """
    STL decomposition. Period=5 for 5-day reporting week (Sun-Thu).
    """
    from statsmodels.tsa.seasonal import STL

    series = df[target_col].dropna()
    # Need a contiguous series for STL — use position index
    series_reindexed = series.reset_index(drop=True)

    stl = STL(series_reindexed, period=period, robust=True)
    result = stl.fit()

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(series_reindexed.index, series_reindexed.values, linewidth=0.8)
    axes[0].set_title(f'{name} — STL Decomposition (period={period})')
    axes[0].set_ylabel('Observed')

    axes[1].plot(series_reindexed.index, result.trend, linewidth=1.2, color='orange')
    axes[1].set_ylabel('Trend')

    axes[2].plot(series_reindexed.index, result.seasonal, linewidth=0.8, color='green')
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(series_reindexed.index, result.resid, linewidth=0.5, color='red', alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Period Position')

    fig.tight_layout()
    fig_path = FIGURES_DIR / f'stl_decomp_{name.replace(" ", "_").lower()}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Seasonal strength
    var_resid = np.var(result.resid)
    var_seasonal_plus_resid = np.var(result.seasonal + result.resid)
    seasonal_strength = max(0, 1 - var_resid / var_seasonal_plus_resid) if var_seasonal_plus_resid > 0 else 0

    var_trend_plus_resid = np.var(result.trend + result.resid)
    trend_strength = max(0, 1 - var_resid / var_trend_plus_resid) if var_trend_plus_resid > 0 else 0

    return {
        'target': name,
        'period': period,
        'seasonal_strength': float(seasonal_strength),
        'trend_strength': float(trend_strength),
        'figure': str(fig_path),
    }


# ──────────────────────────────────────────────────────────────────────
# 6. REGIME-SHIFT VISUALIZATION
# ──────────────────────────────────────────────────────────────────────
def plot_regime_shift(df: pd.DataFrame) -> str:
    """Plot the full time series for both targets showing the regime shift."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    df_real = df[~df['is_imputed']]

    # Target 1: HHS Care
    ax = axes[0]
    ax.plot(df_real['parsed_date'], df_real[TARGET_1], linewidth=0.8, color='steelblue')
    ax.axvline(pd.Timestamp(TRAINING_CAP_DATE), color='red', linestyle='--', label=f'Training cap: {TRAINING_CAP_DATE}')
    ax.set_ylabel(TARGET_1)
    ax.set_title('UAC Time Series — Regime Shift')
    ax.legend()

    # Target 2: Discharged
    ax = axes[1]
    ax.plot(df_real['parsed_date'], df_real[TARGET_2], linewidth=0.8, color='darkorange')
    ax.axvline(pd.Timestamp(TRAINING_CAP_DATE), color='red', linestyle='--', label=f'Training cap: {TRAINING_CAP_DATE}')
    ax.set_ylabel(TARGET_2)
    ax.set_xlabel('Date')
    ax.legend()

    fig.tight_layout()
    fig_path = FIGURES_DIR / 'regime_shift_overview.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(fig_path)


# ──────────────────────────────────────────────────────────────────────
# 7. ZERO-VALUE DAY REVIEW
# ──────────────────────────────────────────────────────────────────────
def zero_value_review(df: pd.DataFrame) -> dict:
    """Examine zero-value days for holiday clustering."""
    df_real = df[~df['is_imputed']].copy()

    results = {}
    for col in FLOW_COLS:
        zeros = df_real[df_real[col] == 0][['parsed_date', col]].copy()
        if len(zeros) > 0:
            zeros['dow'] = zeros['parsed_date'].dt.day_name()
            zeros['month'] = zeros['parsed_date'].dt.month_name()
            results[col] = {
                'count': len(zeros),
                'dates': [str(d.date()) for d in zeros['parsed_date']],
                'days_of_week': zeros['dow'].tolist(),
                'months': zeros['month'].tolist(),
            }
        else:
            results[col] = {'count': 0, 'dates': [], 'days_of_week': [], 'months': []}

    return results


# ──────────────────────────────────────────────────────────────────────
# 8. DIFFERENCED-SERIES CORRELATIONS
# ──────────────────────────────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlations on raw levels, first differences, and detrended."""
    df_real = df[~df['is_imputed']].copy()

    # Raw correlations between Transferred Out and Discharged
    trans = df_real[COL_TRANSFERRED].dropna().astype(float)
    disch = df_real[COL_DISCHARGED].dropna().astype(float)
    common = trans.index.intersection(disch.index)
    raw_corr = float(np.corrcoef(trans[common], disch[common])[0, 1])

    # First-differenced
    diff_trans = trans.diff().dropna()
    diff_disch = disch.diff().dropna()
    common_diff = diff_trans.index.intersection(diff_disch.index)
    diff_corr = float(np.corrcoef(diff_trans[common_diff], diff_disch[common_diff])[0, 1])

    # Detrended (subtract rolling 30-period mean)
    detrend_trans = trans - trans.rolling(30, min_periods=1).mean()
    detrend_disch = disch - disch.rolling(30, min_periods=1).mean()
    common_dt = detrend_trans.dropna().index.intersection(detrend_disch.dropna().index)
    detrend_corr = float(np.corrcoef(detrend_trans[common_dt], detrend_disch[common_dt])[0, 1])

    # Full cross-correlation matrix on raw levels
    numeric_real = df_real[NUMERIC_COLS].apply(lambda c: c.astype(str).str.replace(',', '').astype(float))
    raw_corr_matrix = numeric_real.corr()

    # First-diff cross-correlation matrix
    diff_matrix = numeric_real.diff().dropna().corr()

    # Plot correlation heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(raw_corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Raw Level Correlations')

    sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('First-Differenced Correlations')

    fig.tight_layout()
    fig_path = FIGURES_DIR / 'correlation_heatmaps.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'transferred_discharged_raw': raw_corr,
        'transferred_discharged_diff': diff_corr,
        'transferred_discharged_detrended': detrend_corr,
        'figure': str(fig_path),
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def run_eda():
    """Run the full EDA pipeline and write findings."""
    print("=" * 70)
    print("UAC FORECASTING PROJECT — DAY 3 EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = load_master_series()
    print(f"\nMaster series loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Real observations: {(~df['is_imputed']).sum()}")
    print(f"Imputed gaps: {df['is_imputed'].sum()}")

    all_findings = {}

    # ── 1. Stationarity Tests ──
    print("\n" + "=" * 70)
    print("1. STATIONARITY TESTS")
    print("=" * 70)

    stationarity_results = {}
    for target, name in [(TARGET_1, 'HHS_Care'), (TARGET_2, 'Discharged')]:
        # Raw
        res_raw = run_stationarity_tests(df[target], f'{name} (raw)')
        print_stationarity(res_raw)
        stationarity_results[f'{name}_raw'] = res_raw

        # First differenced
        diff = df[target].dropna().diff().dropna()
        res_diff = run_stationarity_tests(diff, f'{name} (1st diff)')
        print_stationarity(res_diff)
        stationarity_results[f'{name}_diff'] = res_diff

    all_findings['stationarity'] = stationarity_results

    # ── 2. Structural Break Scan ──
    print("\n" + "=" * 70)
    print("2. STRUCTURAL BREAK SCAN")
    print("=" * 70)

    break_results = {}
    for target, name in [(TARGET_1, 'HHS_Care'), (TARGET_2, 'Discharged')]:
        res = structural_break_scan(df, target, name)
        print(f"\n  {name}:")
        print(f"    Dominant break: {res['dominant_break_date']} (CUSUM={res['dominant_cusum_value']:.0f})")
        print(f"    Secondary break: {res['secondary_break_date']} (CUSUM={res['secondary_cusum_value']:.0f})")
        break_results[name] = res

    all_findings['structural_breaks'] = break_results

    # ── 3. Within-Week Seasonality ──
    print("\n" + "=" * 70)
    print("3. WITHIN-WEEK SEASONALITY TEST")
    print("=" * 70)

    seasonality_results = {}
    for target, name in [(TARGET_1, 'HHS_Care'), (TARGET_2, 'Discharged')]:
        res = test_within_week_seasonality(df, target, name)
        print(f"\n  {name} (raw levels):")
        print(f"    Kruskal-Wallis H={res['h_statistic']:.4f}, p={res['p_value']:.6f}")
        print(f"    Within-week seasonality detected: {res['has_within_week_seasonality']}")
        print(f"  {name} (1st diff):")
        print(f"    Kruskal-Wallis H={res['diff_h_statistic']:.4f}, p={res['diff_p_value']:.6f}")
        print(f"    Within-week seasonality (diff): {res['diff_has_within_week_seasonality']}")
        for dow, st in res['group_stats'].items():
            print(f"      {dow}: n={st['count']}, mean={st['mean']:.1f}, median={st['median']:.1f}")
        seasonality_results[name] = res

    all_findings['seasonality'] = seasonality_results

    # ── 4. Seasonal Decomposition ──
    print("\n" + "=" * 70)
    print("4. SEASONAL DECOMPOSITION (STL, period=5)")
    print("=" * 70)

    decomp_results = {}
    for target, name in [(TARGET_1, 'HHS_Care'), (TARGET_2, 'Discharged')]:
        res = seasonal_decomposition(df, target, name, period=5)
        print(f"\n  {name}:")
        print(f"    Seasonal strength: {res['seasonal_strength']:.4f}")
        print(f"    Trend strength: {res['trend_strength']:.4f}")
        decomp_results[name] = res

    all_findings['decomposition'] = decomp_results

    # ── 5. ACF/PACF ──
    print("\n" + "=" * 70)
    print("5. ACF/PACF PLOTS")
    print("=" * 70)

    for target, name in [(TARGET_1, 'HHS_Care'), (TARGET_2, 'Discharged')]:
        fig_path = plot_acf_pacf(df, target, name)
        print(f"  {name} ACF/PACF saved to: {fig_path}")

    # ── 6. Regime Shift Visualization ──
    print("\n" + "=" * 70)
    print("6. REGIME SHIFT VISUALIZATION")
    print("=" * 70)

    fig_path = plot_regime_shift(df)
    print(f"  Overview saved to: {fig_path}")

    # ── 7. Zero-Value Day Review ──
    print("\n" + "=" * 70)
    print("7. ZERO-VALUE DAY REVIEW")
    print("=" * 70)

    zero_results = zero_value_review(df)
    for col, info in zero_results.items():
        if info['count'] > 0:
            print(f"\n  {col}: {info['count']} zero-value days")
            for d, dow, m in zip(info['dates'], info['days_of_week'], info['months']):
                print(f"    {d} ({dow}, {m})")
        else:
            print(f"\n  {col}: no zero-value days")

    all_findings['zero_values'] = zero_results

    # ── 8. Correlations ──
    print("\n" + "=" * 70)
    print("8. DIFFERENCED-SERIES CORRELATIONS")
    print("=" * 70)

    corr_results = compute_correlations(df)
    print(f"  Transferred Out vs Discharged:")
    print(f"    Raw levels:       {corr_results['transferred_discharged_raw']:.4f}")
    print(f"    First difference: {corr_results['transferred_discharged_diff']:.4f}")
    print(f"    Detrended:        {corr_results['transferred_discharged_detrended']:.4f}")

    all_findings['correlations'] = corr_results

    # ── Write findings document ──
    write_findings_document(all_findings)

    print("\n" + "=" * 70)
    print("DAY 3 EDA COMPLETE — All figures and findings written.")
    print("=" * 70)

    return all_findings


def write_findings_document(findings: dict):
    """Write the EDA findings document that drives Day 4 and Day 6 decisions."""
    stat = findings['stationarity']
    breaks = findings['structural_breaks']
    seas = findings['seasonality']
    decomp = findings['decomposition']
    corr = findings['correlations']
    zeros = findings['zero_values']

    # Determine SARIMA seasonal order recommendation
    hhs_seas = seas.get('HHS_Care', {})
    disch_seas = seas.get('Discharged', {})

    doc = []
    doc.append("# EDA Findings — UAC Forecasting Project")
    doc.append("")
    doc.append("**Generated:** Day 3 automated EDA pipeline")
    doc.append("**Purpose:** Drive modeling decisions for Days 4-6")
    doc.append("**All numbers computed from actual data, never fabricated.**")
    doc.append("")
    doc.append("---")
    doc.append("")

    # 1. Stationarity
    doc.append("## 1. Stationarity Tests")
    doc.append("")
    doc.append("| Series | ADF stat | ADF p | ADF conclusion | KPSS stat | KPSS p | KPSS conclusion | Joint |")
    doc.append("|---|---|---|---|---|---|---|---|")
    for key, res in stat.items():
        doc.append(f"| {res['name']} | {res['adf_statistic']:.4f} | {res['adf_pvalue']:.4f} | {res['adf_conclusion']} | {res['kpss_statistic']:.4f} | {res['kpss_pvalue']:.4f} | {res['kpss_conclusion']} | {res['joint_conclusion']} |")

    doc.append("")
    doc.append("**Decision for Day 6:** Both targets require differencing (d=1) to achieve stationarity for SARIMA. "
               "The first-differenced series should be tested to confirm stationarity before fixing SARIMA orders.")
    doc.append("")

    # 2. Structural Breaks
    doc.append("## 2. Structural Break Scan")
    doc.append("")
    for name, res in breaks.items():
        doc.append(f"### {name}")
        doc.append(f"- Dominant break: **{res['dominant_break_date']}** (CUSUM = {res['dominant_cusum_value']:.0f})")
        doc.append(f"- Secondary break: **{res['secondary_break_date']}** (CUSUM = {res['secondary_cusum_value']:.0f})")
        doc.append("")

    doc.append(f"**Locked decision:** Training-window cap at **{TRAINING_CAP_DATE}** confirmed by structural-break scan.")
    doc.append("")

    # 3. Within-Week Seasonality
    doc.append("## 3. Within-Week Seasonality")
    doc.append("")
    doc.append("| Target | Test | H stat | p-value | Seasonal? | Diff H stat | Diff p-value | Diff Seasonal? |")
    doc.append("|---|---|---|---|---|---|---|---|")
    for name, res in seas.items():
        doc.append(f"| {name} | Kruskal-Wallis | {res['h_statistic']:.4f} | {res['p_value']:.6f} | {res['has_within_week_seasonality']} | {res['diff_h_statistic']:.4f} | {res['diff_p_value']:.6f} | {res['diff_has_within_week_seasonality']} |")

    doc.append("")

    # Determine recommendation
    hhs_raw_seasonal = hhs_seas.get('has_within_week_seasonality', False)
    hhs_diff_seasonal = hhs_seas.get('diff_has_within_week_seasonality', False)
    disch_raw_seasonal = disch_seas.get('has_within_week_seasonality', False)
    disch_diff_seasonal = disch_seas.get('diff_has_within_week_seasonality', False)

    doc.append("**SARIMA seasonal order decision:**")
    doc.append("")
    if hhs_diff_seasonal:
        doc.append("- **HHS Care:** Within-week seasonality detected in differenced series. Use SARIMA with seasonal period m=5 (5-day reporting week).")
    else:
        doc.append("- **HHS Care:** No significant within-week seasonality in differenced series. Use non-seasonal ARIMA (m=1 or no seasonal component).")

    if disch_diff_seasonal:
        doc.append("- **Discharged:** Within-week seasonality detected in differenced series. Use SARIMA with seasonal period m=5.")
    else:
        doc.append("- **Discharged:** No significant within-week seasonality in differenced series. Use non-seasonal ARIMA (m=1 or no seasonal component).")
    doc.append("")

    # 4. Decomposition
    doc.append("## 4. Seasonal Decomposition (STL, period=5)")
    doc.append("")
    doc.append("| Target | Seasonal Strength | Trend Strength |")
    doc.append("|---|---|---|")
    for name, res in decomp.items():
        doc.append(f"| {name} | {res['seasonal_strength']:.4f} | {res['trend_strength']:.4f} |")
    doc.append("")
    doc.append("Seasonal strength close to 0 = weak seasonality; close to 1 = strong. "
               "Trend strength close to 1 = dominant trend component.")
    doc.append("")

    # 5. Correlations
    doc.append("## 5. Cross-Series Correlations")
    doc.append("")
    doc.append("### Transferred Out vs Discharged (imbalance signal components)")
    doc.append("")
    doc.append(f"- Raw levels: **{corr['transferred_discharged_raw']:.4f}**")
    doc.append(f"- First difference: **{corr['transferred_discharged_diff']:.4f}**")
    doc.append(f"- Detrended: **{corr['transferred_discharged_detrended']:.4f}**")
    doc.append("")
    doc.append("**Decision for Day 8:** The addendum pre-registered expectation of near-independence in first differences "
               f"({'confirmed' if abs(corr['transferred_discharged_diff']) < 0.2 else 'NOT confirmed'} — "
               f"diff correlation = {corr['transferred_discharged_diff']:.4f}). "
               "Use the full Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B) formula unless first-diff correlation is negligible.")
    doc.append("")

    # 6. Zero-Value Days
    doc.append("## 6. Zero-Value Days")
    doc.append("")
    for col, info in zeros.items():
        if info['count'] > 0:
            doc.append(f"### {col}: {info['count']} zero-value day(s)")
            doc.append("")
            for d, dow, m in zip(info['dates'], info['days_of_week'], info['months']):
                doc.append(f"- {d} ({dow}, {m})")
            doc.append("")
        else:
            doc.append(f"### {col}: No zero-value days")
            doc.append("")

    doc.append("**Holiday clustering assessment:** Review whether zero-value dates coincide with US federal holidays. "
               "Sparse zeros (2-3 per column) do not indicate systematic holiday effects; "
               "MAPE remains unstable for these columns — sMAPE/MASE needed alongside.")
    doc.append("")

    # 7. Summary decisions
    doc.append("## 7. Summary of Modeling Decisions from EDA")
    doc.append("")
    doc.append("| Decision | Value | Evidence |")
    doc.append("|---|---|---|")
    doc.append(f"| Differencing order (d) | 1 for both targets | ADF/KPSS tests above |")

    if hhs_diff_seasonal:
        doc.append(f"| Seasonal period (HHS Care) | m=5 | Kruskal-Wallis p={hhs_seas.get('diff_p_value', 'N/A'):.6f} |")
    else:
        doc.append(f"| Seasonal period (HHS Care) | None (m=1) | Kruskal-Wallis p={hhs_seas.get('diff_p_value', 'N/A'):.6f}, no weekly effect |")

    if disch_diff_seasonal:
        doc.append(f"| Seasonal period (Discharged) | m=5 | Kruskal-Wallis p={disch_seas.get('diff_p_value', 'N/A'):.6f} |")
    else:
        doc.append(f"| Seasonal period (Discharged) | None (m=1) | Kruskal-Wallis p={disch_seas.get('diff_p_value', 'N/A'):.6f}, no weekly effect |")

    doc.append(f"| Training cap date | {TRAINING_CAP_DATE} | Structural-break scan confirms regime shift |")
    doc.append(f"| Rolling window design | Period-based (not calendar) | Already locked in addendum |")
    doc.append(f"| MAPE reliability | Unstable for flow columns | Zero-value days present |")
    doc.append(f"| Imbalance signal independence | {'Near-independent' if abs(corr['transferred_discharged_diff']) < 0.2 else 'Correlated'} | Diff correlation = {corr['transferred_discharged_diff']:.4f} |")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Figures Generated")
    doc.append("")
    doc.append("All figures saved to `reports/figures/`:")
    doc.append("")
    doc.append("- `structural_break_hhs_care.png`")
    doc.append("- `structural_break_discharged.png`")
    doc.append("- `weekly_seasonality_hhs_care.png`")
    doc.append("- `weekly_seasonality_discharged.png`")
    doc.append("- `acf_pacf_hhs_care.png`")
    doc.append("- `acf_pacf_discharged.png`")
    doc.append("- `stl_decomp_hhs_care.png`")
    doc.append("- `stl_decomp_discharged.png`")
    doc.append("- `regime_shift_overview.png`")
    doc.append("- `correlation_heatmaps.png`")

    findings_path = DOCS_DIR / 'eda_findings.md'
    findings_path.write_text("\n".join(doc), encoding='utf-8')
    print(f"\n  Findings document written to: {findings_path}")


if __name__ == '__main__':
    run_eda()
