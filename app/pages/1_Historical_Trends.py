"""
Page 2 of Part 8 -- "Historical Trends" [REC].

Exists so the forecast pages make sense: without seeing the 5.8x regime shift
and the Sunday-Thursday reporting cadence, a reader cannot judge why the models
behave as they do, or why a baseline can beat a tuned SARIMA on this series.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Historical Trends", page_icon="📉", layout="wide")

from app.lib.artifacts import (  # noqa: E402
    VARIABLE_LABELS, load_history, load_provenance, provenance_caption,
)

st.title("Historical Trends")
st.caption("Full reporting history — the context every forecast on this site sits on")

history = load_history()
st.markdown(provenance_caption(load_provenance()))
st.divider()

dates = pd.to_datetime(history["parsed_date"])
observed = ~history["is_imputed"].to_numpy(dtype=bool)

controls = st.columns([3, 2])
with controls[0]:
    chosen = st.multiselect(
        "Variables", list(VARIABLE_LABELS), default=["Children in HHS Care"],
        format_func=lambda c: VARIABLE_LABELS[c],
    )
with controls[1]:
    span = st.select_slider(
        "Date range", options=["Last 90 periods", "Last 180 periods",
                               "Last 365 periods", "Full history"],
        value="Full history",
    )

periods = {"Last 90 periods": 90, "Last 180 periods": 180,
           "Last 365 periods": 365, "Full history": len(history)}[span]
window = history.iloc[-periods:]
window_dates = pd.to_datetime(window["parsed_date"])

if not chosen:
    st.info("Select at least one variable.")
    st.stop()

# Stock and flow columns differ by orders of magnitude (thousands against tens),
# so plotting them on one axis would flatten the flows into a line at zero.
# Each variable gets its own panel instead of a shared, misleading scale.
fig = go.Figure()
for col in chosen:
    fig.add_trace(go.Scatter(
        x=window_dates, y=window[col].astype(float), mode="lines",
        name=VARIABLE_LABELS[col],
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>" + VARIABLE_LABELS[col] + "</extra>",
    ))
fig.update_layout(height=420, margin=dict(l=60, r=20, t=30, b=10),
                  hovermode="x unified", yaxis_title="Children",
                  legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
if len(chosen) > 1:
    st.caption("⚠️ Stock series (thousands) and flow series (tens) share one axis "
               "here — select them separately to read the flows properly.")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Why the series looks the way it does")
care = history["Children in HHS Care"].astype(float)
peak_i, trough_i = int(care.idxmax()), int(care.idxmin())
cols = st.columns(3)
cols[0].metric("Peak", f"{care.max():,.0f}", help=str(dates.iloc[peak_i].date()))
cols[1].metric("Trough", f"{care.min():,.0f}", help=str(dates.iloc[trough_i].date()))
cols[2].metric("Peak ÷ trough", f"{care.max() / care.min():.1f}×")

st.markdown(
    "**A structural decline, not noise.** `Children in HHS Care` falls **%.1f×** from "
    "its peak of %s (%s) to its trough of %s (%s), then edges back up. This is why "
    "models are trained on a window capped at the 2025-02-05 regime boundary rather "
    "than on the full history — and why tree-based models, which cannot extrapolate "
    "beyond their training range, perform poorly on this target."
    % (care.max() / care.min(), f"{care.max():,.0f}", dates.iloc[peak_i].date(),
       f"{care.min():,.0f}", dates.iloc[trough_i].date())
)

weekday = dates.dt.day_name().value_counts()
st.markdown(
    "**Reporting is Sunday–Thursday, not daily.** Across %d reporting slots: %s. "
    "Friday appears only twice and Saturday never. Every lag, rolling window and "
    "forecast horizon in this project is counted in *reporting periods*, never "
    "calendar days — reindexing to a 7-day calendar would have made roughly a third "
    "of the series synthetic."
    % (len(history), ", ".join("%s %d" % (d, n) for d, n in weekday.items()))
)

n_interp = int((~observed).sum())
st.markdown(
    "**%d of %d slots are interpolated**, not reported. Stock columns are filled by "
    "linear interpolation so the index stays regular; flow columns are left genuinely "
    "missing and are never zero-filled. Interpolated points are marked distinctly on "
    "the forecast pages, and are excluded from every accuracy figure in this app."
    % (n_interp, len(history))
)
