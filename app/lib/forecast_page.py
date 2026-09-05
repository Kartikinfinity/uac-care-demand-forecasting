"""
forecast_page.py -- The shared renderer behind both mandated forecast pages.

Part 8 specifies pages 3 and 4 ("Future Care Load Forecast Chart" and "Discharge
Demand Forecast Panel") as the same pattern applied to different targets: a
forecast line over a historical tail with a confidence band, KPI cards for the
point forecast / interval width / model used, and a horizon selector plus model
toggle.

Written once here rather than twice, for the same reason the agent instructions
live in one module elsewhere in this project: two copies of a chart drift, and a
reader comparing care load against discharge demand needs to know the difference
they see is in the data, not in two slightly different renderers.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib.artifacts import (
    HORIZON_CALENDAR_DAYS,
    MODEL_LABELS,
    available_models,
    champion_for,
    coverage_note,
    forecast_row,
    horizon_label,
    load_coverage,
    load_forward,
    load_history,
    load_provenance,
    provenance_caption,
)

HISTORY_TAIL_DEFAULT = 90


def _forecast_date(history: pd.DataFrame, origin_pos: int, horizon: int):
    """
    Calendar date of the forecast target position.

    The forecast lands `horizon` PERIOD-POSITIONS past the origin, which is a
    different number of calendar days depending on where the weekend falls. Past
    the end of the series there is no recorded date, so the next positions are
    projected forward on the Sunday-Thursday reporting cadence rather than by
    adding `horizon` days -- adding days would silently place a Saturday
    forecast on a calendar the programme never reports on.
    """
    dates = pd.to_datetime(history["parsed_date"])
    last_pos = len(dates) - 1
    if origin_pos + horizon <= last_pos:
        return dates.iloc[origin_pos + horizon]

    # `datetime.timedelta`, not `pd.Timedelta(days=1)`: the latter raises a
    # NumPy deprecation about bare-integer timedelta units on every call.
    current = dates.iloc[last_pos]
    for _ in range(origin_pos + horizon - last_pos):
        current = current + timedelta(days=1)
        while current.dayofweek in (4, 5):      # skip Friday and Saturday
            current = current + timedelta(days=1)
    return current


def _chart(history, target, tail, origin_pos, forecast_date, row, model_label):
    """Historical tail, the point forecast, and its interval band."""
    dates = pd.to_datetime(history["parsed_date"])
    values = history[target].astype(float)
    observed = ~history["is_imputed"].to_numpy(dtype=bool)

    start = max(0, origin_pos - tail + 1)
    idx = slice(start, origin_pos + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.iloc[idx], y=values.iloc[idx], mode="lines",
        name="Observed", line=dict(color="#1f4e79", width=2),
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Observed</extra>",
    ))

    # Interpolated slots are drawn distinctly: they are values the cleaning step
    # invented to keep the index regular, not figures the programme reported.
    gap_mask = np.zeros(len(history), dtype=bool)
    gap_mask[idx] = True
    gap_mask &= ~observed
    if gap_mask.any():
        fig.add_trace(go.Scatter(
            x=dates[gap_mask], y=values[gap_mask], mode="markers",
            name="Interpolated (not reported)",
            marker=dict(color="#c00000", size=5, symbol="x"),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Interpolated</extra>",
        ))

    anchor_date, anchor_value = dates.iloc[origin_pos], values.iloc[origin_pos]

    if row is not None and bool(row["interval_sufficient"]):
        fig.add_trace(go.Scatter(
            x=[anchor_date, forecast_date, forecast_date, anchor_date],
            y=[anchor_value, row["upper"], row["lower"], anchor_value],
            fill="toself", fillcolor="rgba(31,78,121,0.15)",
            line=dict(width=0), hoverinfo="skip",
            name="95% interval",
        ))

    if row is not None and np.isfinite(row["point_forecast"]):
        fig.add_trace(go.Scatter(
            x=[anchor_date, forecast_date], y=[anchor_value, row["point_forecast"]],
            mode="lines+markers", name="Forecast (%s)" % model_label,
            line=dict(color="#c55a11", width=2, dash="dash"),
            marker=dict(size=9),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Forecast</extra>",
        ))

    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis_title=None, yaxis_title="Children",
    )
    return fig


def render(target: str, page_title: str, subtitle: str, question: str) -> None:
    """Render one complete forecast page for `target`."""
    st.title(page_title)
    st.caption(subtitle)

    history = load_history()
    forward = load_forward()
    coverage = load_coverage()
    provenance = load_provenance()

    st.markdown(provenance_caption(provenance))
    st.divider()

    # -- controls ------------------------------------------------------
    controls = st.columns([2, 3, 2])
    with controls[0]:
        horizons = sorted(forward[forward["target"] == target]["horizon"].unique())
        horizon = st.selectbox(
            "Forecast horizon", horizons, index=0,
            format_func=horizon_label,
            help="Horizons are counted in REPORTING PERIODS (Sun-Thu), not calendar "
                 "days. The calendar equivalent is shown in brackets.",
        )
    champion = champion_for(forward, target, horizon)
    with controls[1]:
        models = available_models(forward, target, horizon)
        model = st.selectbox(
            "Model", models, index=0,
            format_func=lambda m: MODEL_LABELS.get(m, m) + (" — champion" if m == champion else ""),
            help="Every model was evaluated on identical walk-forward folds. The "
                 "champion is the one the Day-8 selection rule chose for this cell.",
        )
    with controls[2]:
        tail = st.slider("History shown (periods)", 30, 300, HISTORY_TAIL_DEFAULT, step=10)

    row = forecast_row(forward, target, model, horizon)
    if row is None:
        st.error("No forecast artifact exists for this combination.")
        return

    origin_pos = int(row["origin_pos"])
    forecast_date = _forecast_date(history, origin_pos, horizon)

    # -- KPI cards -----------------------------------------------------
    # KPI cards carry only SHORT values. `st.metric` truncates on font size
    # regardless of column width, and its `delta` slot renders a direction arrow
    # even with delta_color="off" -- an up-arrow beside an interval range reads
    # as "the interval went up", which is meaningless. The range therefore gets
    # its own full-width line below, where it cannot be cut or misread.
    sufficient = bool(row["interval_sufficient"])
    cards = st.columns(4)
    cards[0].metric("Point forecast", f"{row['point_forecast']:,.0f}",
                    help="Forecast for %s" % forecast_date.date())
    cards[1].metric("Interval width",
                    f"{row['interval_width']:,.0f}" if sufficient else "—",
                    help="Wider means less certainty.")
    cards[2].metric("Horizon", "h=%d" % horizon,
                    help="Reporting periods (Sun-Thu), not calendar days.")
    cards[3].metric("Model", MODEL_LABELS.get(model, model).split(" (")[0],
                    delta="champion" if model == champion else "not champion",
                    delta_color="normal" if model == champion else "off")

    if sufficient:
        st.markdown(
            "**95%% prediction interval: %s – %s**  ·  built from %d out-of-sample "
            "residuals  ·  horizon %s"
            % (f"{row['lower']:,.0f}", f"{row['upper']:,.0f}",
               int(row["n_residuals"]), horizon_label(horizon))
        )
    else:
        st.markdown("**95%% prediction interval: not available**  ·  horizon %s"
                    % horizon_label(horizon))

    st.plotly_chart(_chart(history, target, tail, origin_pos, forecast_date, row,
                           MODEL_LABELS.get(model, model)),
                    use_container_width=True)

    st.caption(
        "Forecast issued from **%s** (the last reported observation) for **%s**, "
        "%s ahead."
        % (row["origin_date"].date(), forecast_date.date(), horizon_label(horizon))
    )

    # -- the honest bits, on the same screen as the number -------------
    if not bool(row["interval_sufficient"]):
        st.warning(
            "**No interval is shown for this model.** %s Only the point forecast "
            "is available." % row["interval_note"]
        )
    else:
        calibrated, message = coverage_note(coverage, target, model, horizon)
        if calibrated is True:
            st.success("**Interval calibration.** " + message)
        elif calibrated is False:
            st.error("**Interval calibration.** " + message)
        else:
            st.info("**Interval calibration.** " + message)

        if bool(row.get("clipped_at_zero", False)):
            st.warning(
                "The lower bound was clipped at zero. The raw empirical quantile "
                "was %.1f, which is impossible for a count of children — the "
                "unclipped value is kept in the artifact for transparency."
                % row["lower_unclipped"]
            )

    if model != champion:
        st.info(
            "You are viewing **%s**, which is not the champion for this cell. The "
            "Day-8 selection rule chose **%s**. Non-champion forecasts are shown so "
            "model choice can be inspected, not because they are recommended."
            % (MODEL_LABELS.get(model, model), MODEL_LABELS.get(champion, champion))
        )

    with st.expander("What this page answers, and what it cannot"):
        st.markdown(question)
        st.markdown(
            "- Horizons are **reporting periods**, not calendar days. The programme "
            "reports Sunday–Thursday, so 7 periods is about 9 calendar days.\n"
            "- The interval comes from **out-of-sample** walk-forward residuals "
            "restricted to the current regime (origins on or after 2025-02-05). It "
            "is not a model-internal confidence interval.\n"
            "- Residual pools are **12–15 observations**, so interval endpoints are "
            "estimated from very few points. The calibration note above reports the "
            "measured coverage and its uncertainty for this exact cell.\n"
            "- Nothing on this page is computed live. Every number was produced "
            "offline by `src/forecast/generate.py` and read from disk."
        )
