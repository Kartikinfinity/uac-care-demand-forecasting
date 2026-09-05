"""
artifacts.py -- The dashboard's ONLY data access layer.

Roadmap Part 9: the app "reads only from `forecasts/` and `models/` metadata"
and "never trains or refits a model as part of serving a page". This module is
where that boundary is enforced: it loads flat files and nothing else. There is
no model import here, no fitting, no `src.models` dependency -- if a future page
needs a number, it comes from an artifact `generate.py` already wrote, or it
does not appear.

Every loader is `@st.cache_data`, so a page re-render re-reads nothing. Missing
artifacts raise a single, actionable error naming the command that produces
them, rather than a bare `FileNotFoundError` from somewhere deep in a page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    CHAMPION_METRICS_PATH,
    EARLY_WARNING_BACKTEST_PATH,
    FORECAST_HORIZONS,
    FORECAST_PROVENANCE_PATH,
    FORWARD_FORECASTS_PATH,
    FULL_COMPARISON_PATH,
    HOLDOUT_EVALUATION_PATH,
    IMBALANCE_FORECAST_PATH,
    INTERVAL_COVERAGE_PATH,
    KPI_SUMMARY_PATH,
    MASTER_SERIES_PATH,
    SELECTION_SCOPE,
    SELECTION_WINDOW_RULE,
    TARGET_1,
    TARGET_2,
)

TARGET_LABELS = {
    TARGET_1: "Children in HHS Care",
    TARGET_2: "Children discharged from HHS Care",
}

MODEL_LABELS = {
    "naive": "Naive (persistence)",
    "seasonal_naive": "Seasonal naive (m=5)",
    "moving_average": "Moving average (w=7)",
    "sarima": "SARIMA",
    "exponential_smoothing": "Exponential smoothing (ETS)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ensemble": "Ensemble (statistical + ML average)",
}

# Measured from the actual reporting calendar, not assumed. The series runs
# Sunday-Thursday, so a period-position offset is NOT a calendar-day offset --
# the addendum requires horizon labels to show both so a reader planning in days
# is never misled by a number counted in reporting periods.
HORIZON_CALENDAR_DAYS = {1: "~1 day", 7: "~9 days", 14: "~20 days"}


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            "Missing artifact: %s\n\nThe dashboard reads only pre-generated files "
            "and never trains a model. Produce it with:\n    %s" % (path.name, command)
        )
    return path


GENERATE = "python -m src.forecast.generate"


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    """The cleaned master series -- the historical context every chart sits on."""
    df = pd.read_parquet(_require(MASTER_SERIES_PATH, "python -m src.data.clean"))
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    return df


@st.cache_data(show_spinner=False)
def load_forward() -> pd.DataFrame:
    df = pd.read_csv(_require(FORWARD_FORECASTS_PATH, GENERATE))
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame:
    return pd.read_csv(_require(KPI_SUMMARY_PATH, GENERATE))


@st.cache_data(show_spinner=False)
def load_coverage() -> pd.DataFrame:
    return pd.read_csv(_require(INTERVAL_COVERAGE_PATH, GENERATE))


@st.cache_data(show_spinner=False)
def load_holdout() -> pd.DataFrame:
    return pd.read_csv(_require(HOLDOUT_EVALUATION_PATH, GENERATE))


@st.cache_data(show_spinner=False)
def load_imbalance() -> pd.DataFrame:
    return pd.read_csv(_require(IMBALANCE_FORECAST_PATH, GENERATE))


@st.cache_data(show_spinner=False)
def load_early_warning() -> pd.DataFrame:
    return pd.read_csv(_require(EARLY_WARNING_BACKTEST_PATH, GENERATE))


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(_require(FULL_COMPARISON_PATH, "python -m src.evaluation.run_ml"))


@st.cache_data(show_spinner=False)
def load_champions() -> pd.DataFrame:
    return pd.read_csv(_require(CHAMPION_METRICS_PATH, "python -m src.evaluation.run_selection"))


@st.cache_data(show_spinner=False)
def load_provenance() -> dict:
    return json.loads(_require(FORECAST_PROVENANCE_PATH, GENERATE).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Derived views -- kept here so the two forecast pages cannot drift apart
# ----------------------------------------------------------------------
def forecast_row(forward: pd.DataFrame, target: str, model: str, horizon: int):
    """The one forward-forecast record a page is currently showing, or None."""
    sub = forward[
        (forward["target"] == target)
        & (forward["model"] == model)
        & (forward["horizon"] == horizon)
    ]
    return None if sub.empty else sub.iloc[0]


def champion_for(forward: pd.DataFrame, target: str, horizon: int) -> str:
    sub = forward[(forward["target"] == target) & (forward["horizon"] == horizon)]
    champs = sub[sub["is_champion"]]
    return None if champs.empty else champs.iloc[0]["model"]


def available_models(forward: pd.DataFrame, target: str, horizon: int) -> list:
    """Models offered by the toggle, champion first, then by label."""
    sub = forward[(forward["target"] == target) & (forward["horizon"] == horizon)]
    champion = champion_for(forward, target, horizon)
    others = sorted(m for m in sub["model"].unique() if m != champion)
    return ([champion] + others) if champion else others


def horizon_label(horizon: int) -> str:
    """
    "h=7 periods (~9 days)". Both units, always.

    The reporting calendar is Sunday-Thursday, so 7 reporting periods is about 9
    calendar days, not 7. A stakeholder planning staffing in days would be
    misled by the period count alone, which is why the addendum requires the
    calendar equivalent alongside it.
    """
    return "%d period%s (%s)" % (horizon, "" if horizon == 1 else "s",
                                 HORIZON_CALENDAR_DAYS.get(horizon, "~"))


def coverage_note(coverage: pd.DataFrame, target: str, model: str, horizon: int):
    """
    The calibration caveat that must ride along with every interval.

    Returns (is_calibrated, message). A cell whose empirical coverage band
    excludes the nominal level is a measured calibration gap and the reader is
    told so on the same screen as the band -- not left to find it in an appendix.
    """
    sub = coverage[
        (coverage["target"] == target)
        & (coverage["champion"] == model)
        & (coverage["horizon"] == horizon)
    ]
    if sub.empty:
        return None, (
            "No calibration evidence exists for this model at this horizon. "
            "Coverage was measured for the champion of each cell; other models "
            "show the interval implied by their own out-of-sample residuals, "
            "without a calibration check behind it."
        )
    row = sub.iloc[0]
    if bool(row["covers_nominal"]):
        return True, (
            "Measured coverage %.0f%% against a %.0f%% nominal level (n=%d). The "
            "binomial band [%.0f%%, %.0f%%] contains the nominal level, so the gap "
            "is not distinguishable from sampling noise at this sample size."
            % (row["empirical_coverage"] * 100, row["nominal_coverage"] * 100,
               row["n"], row["band_low"] * 100, row["band_high"] * 100)
        )
    return False, (
        "Measured coverage %.0f%% against a %.0f%% nominal level (n=%d). The "
        "binomial band [%.0f%%, %.0f%%] does NOT contain the nominal level: this "
        "interval is too narrow, by a margin larger than sampling noise explains. "
        "Treat the band as optimistic."
        % (row["empirical_coverage"] * 100, row["nominal_coverage"] * 100,
           row["n"], row["band_low"] * 100, row["band_high"] * 100)
    )


def provenance_caption(provenance: dict) -> str:
    """The one-line readout the addendum requires wherever forecasts appear."""
    return ("Data as of **%s** · forecasts generated **%s** · champions selected on "
            "`%s` (`%s` window) · manual refresh only"
            % (provenance["data_as_of"], provenance["generated_at_utc"],
               SELECTION_SCOPE, SELECTION_WINDOW_RULE))
