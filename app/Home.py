"""
Home.py -- Page 1 of Part 8, "Executive Overview" [REC], and the app entry point.

Single-screen orientation for a non-technical visitor: where things stand, and
how much the forecast can be trusted. The four KPI cards are wired to the real
computed values in `forecasts/kpi_summary.csv` -- none is hard-coded, and each
carries the formula that produced it, because two of the four are proxies or
substitutions and a bare number would overstate them.

This app NEVER trains a model. Every figure was produced offline by
`src/forecast/generate.py` and read from a flat file.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="UAC Care Load Forecasting", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from src.config import TARGET_1, TARGET_2  # noqa: E402
from app.lib.artifacts import (  # noqa: E402
    MODEL_LABELS,
    horizon_label,
    load_forward,
    load_history,
    load_kpis,
    load_provenance,
    provenance_caption,
)

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("Unaccompanied Alien Children (UAC) Program · U.S. Department of Health "
           "and Human Services · independent analytical exercise")

try:
    provenance = load_provenance()
    kpis = load_kpis()
    history = load_history()
    forward = load_forward()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(provenance_caption(provenance))
st.divider()

# ----------------------------------------------------------------------
# KPI cards -- the four the official documentation names
# ----------------------------------------------------------------------
st.subheader("Key performance indicators")
focus = st.radio("Target", [TARGET_1, TARGET_2], horizontal=True, label_visibility="collapsed")
row = kpis[kpis["target"] == focus].iloc[0]

cards = st.columns(4)
cards[0].metric(
    "Forecast accuracy", "%.1f%%" % row["forecast_accuracy_pct"],
    help=str(row["forecast_accuracy_formula"]),
)
cards[1].metric(
    "Surge lead time",
    ("%.1f periods" % row["median_surge_lead_time_periods"]
     if pd.notna(row["median_surge_lead_time_periods"]) else "—"),
    help=str(row["surge_lead_time_formula"]),
)
cards[2].metric(
    "Capacity state (proxy)", str(row["capacity_tier"]),
    help=str(row["capacity_tier_formula"]),
)
cards[3].metric(
    "Forecast stability index",
    ("%.2f" % row["forecast_stability_index"]
     if pd.notna(row["forecast_stability_index"]) else "—"),
    help=str(row["stability_formula"]),
)

warn = st.columns(2)
with warn[0]:
    st.warning(
        "**Capacity state is a proxy, not an official threshold.** No capacity figure "
        "exists in the programme documentation or the dataset. This is a relative "
        "measure against the programme's own recent levels — it does not indicate that "
        "capacity has been or will be breached."
    )
with warn[1]:
    fn = row["false_negative_rate"]
    st.warning(
        "**The early-warning signal is conservative.** False-positive rate %.0f%%, but "
        "it misses %.0f%% of threshold crossings, and a %.1f-period median lead time is "
        "short relative to most staffing decisions. Treat a firing as informative and "
        "silence as uninformative."
        % ((row["false_positive_rate"] or 0) * 100, (fn or 0) * 100,
           row["median_surge_lead_time_periods"] or 0)
    )

# ----------------------------------------------------------------------
# Small multiples -- recent trend for both targets
# ----------------------------------------------------------------------
st.divider()
st.subheader("Where things stand")
tail = st.select_slider("Recent history shown", options=[30, 60, 90, 180],
                        value=90, format_func=lambda n: "%d periods" % n)

dates = pd.to_datetime(history["parsed_date"])
panels = st.columns(2)
for panel, target in zip(panels, (TARGET_1, TARGET_2)):
    window = history.iloc[-tail:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(window["parsed_date"]), y=window[target].astype(float),
        mode="lines", line=dict(color="#1f4e79", width=2), name="Observed",
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(height=240, margin=dict(l=55, r=15, t=35, b=10),
                      showlegend=False, title=dict(text=target, font=dict(size=14)),
                      yaxis_title=None, hovermode="x unified")
    panel.plotly_chart(fig, use_container_width=True)

    latest = history[target].astype(float).iloc[-1]
    champ = forward[(forward["target"] == target) & forward["is_champion"]
                    & (forward["horizon"] == 1)]
    if len(champ):
        c = champ.iloc[0]
        panel.markdown(
            "Latest reported **%s** · next-period forecast **%s** "
            "(%s, %s)"
            % (f"{latest:,.0f}", f"{c['point_forecast']:,.0f}",
               MODEL_LABELS.get(c["model"], c["model"]), horizon_label(1))
        )

# ----------------------------------------------------------------------
# Orientation
# ----------------------------------------------------------------------
st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("What this dashboard is")
    st.markdown(
        "Short-term forecasts of **children in HHS care** and **discharge demand**, "
        "with an intake-versus-exit pressure signal and an early-warning layer.\n\n"
        "Every model was validated by expanding-window walk-forward cross-validation on "
        "65 chronological folds, and no model was accepted unless it beat a naive and a "
        "seasonal-naive baseline on held-out folds."
    )
    st.page_link("pages/1_Historical_Trends.py", label="Historical Trends", icon="📉")
    st.page_link("pages/2_Care_Load_Forecast.py", label="Care Load Forecast", icon="📈")
    st.page_link("pages/3_Discharge_Demand_Forecast.py", label="Discharge Demand Forecast", icon="📤")
    st.page_link("pages/4_Intake_vs_Exit_Pressure.py", label="Intake vs. Exit Pressure", icon="⚠️")
    st.page_link("pages/5_Model_Comparison.py", label="Model Comparison & Accuracy", icon="📋")
    st.page_link("pages/6_Scenario_Comparison.py", label="Scenario Comparison", icon="🔀")
    st.page_link("pages/7_Methodology.py", label="Methodology & Data", icon="📖")

with right:
    st.subheader("Read these first")
    st.info(
        "**Horizons are reporting periods, not days.** The programme reports "
        "Sunday–Thursday, so a 7-period horizon is roughly 9 calendar days. Both units "
        "are shown wherever a horizon appears."
    )
    st.info(
        "**Not continuously live.** Data is refreshed manually: replace the source CSV "
        "and re-run the offline generation script. The dashboard never fits a model."
    )
    st.info(
        "**Baselines won.** Champion selection preferred the simpler model wherever the "
        "evidence could not separate candidates. On this data that means simple "
        "baselines beat SARIMA, exponential smoothing and both ML families in every "
        "target/horizon cell. See Model Comparison for the full evidence."
    )

st.divider()
st.subheader("Current champions")
st.caption("Selected on recent-regime walk-forward evidence under a rule that prefers "
           "the simpler model when candidates cannot be told apart.")
cols = st.columns(3)
for i, entry in enumerate(provenance["winning_configuration"]):
    with cols[i % 3]:
        st.metric(
            label="%s · h=%d" % (entry["target"].replace("Children ", ""), entry["horizon"]),
            value=MODEL_LABELS.get(entry["champion"], entry["champion"]).split(" (")[0],
        )
