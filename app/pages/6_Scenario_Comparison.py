"""
Page 7 of Part 8 -- "Scenario Comparison" [DOC].

Covers the documented User Capability "scenario comparison view": overlay two or
more models and/or horizons on one chart, so a reader can see how much model
choice and horizon choice actually matter on this data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Scenario Comparison", page_icon="🔀", layout="wide")

from app.lib.artifacts import (  # noqa: E402
    MODEL_LABELS, champion_for, horizon_label, load_forward, load_history,
    load_provenance, provenance_caption,
)
from app.lib.forecast_page import _forecast_date  # noqa: E402

st.title("Scenario Comparison")
st.caption("How much do model choice and horizon choice actually matter?")

history = load_history()
forward = load_forward()
st.markdown(provenance_caption(load_provenance()))
st.divider()

controls = st.columns([2, 3, 2])
with controls[0]:
    target = st.selectbox("Target", sorted(forward["target"].unique()))
available = forward[forward["target"] == target]
with controls[1]:
    champion = champion_for(forward, target, 1)
    models = st.multiselect(
        "Models to overlay", sorted(available["model"].unique()),
        default=[m for m in (champion, "sarima", "gradient_boosting")
                 if m in set(available["model"])],
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )
with controls[2]:
    horizons = st.multiselect("Horizons", sorted(available["horizon"].unique()),
                              default=sorted(available["horizon"].unique()),
                              format_func=horizon_label)

if not models or not horizons:
    st.info("Select at least one model and one horizon.")
    st.stop()

dates = pd.to_datetime(history["parsed_date"])
values = history[target].astype(float)
origin_pos = int(available["origin_pos"].iloc[0])
tail = 60
start = max(0, origin_pos - tail + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates.iloc[start:origin_pos + 1], y=values.iloc[start:origin_pos + 1],
    mode="lines", name="Observed", line=dict(color="#333333", width=2),
    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Observed</extra>",
))

palette = ["#c55a11", "#1f4e79", "#548235", "#7030a0", "#bf9000",
           "#c00000", "#00728e", "#7f7f7f"]
rows = []
for i, model in enumerate(models):
    colour = palette[i % len(palette)]
    xs, ys = [dates.iloc[origin_pos]], [values.iloc[origin_pos]]
    for horizon in sorted(horizons):
        row = available[(available["model"] == model) & (available["horizon"] == horizon)]
        if row.empty or not np.isfinite(row["point_forecast"].iloc[0]):
            continue
        r = row.iloc[0]
        xs.append(_forecast_date(history, origin_pos, horizon))
        ys.append(float(r["point_forecast"]))
        rows.append({
            "model": MODEL_LABELS.get(model, model),
            "horizon": horizon_label(horizon),
            "point_forecast": float(r["point_forecast"]),
            "lower": r["lower"], "upper": r["upper"],
            "interval_width": r["interval_width"],
            "is_champion": bool(r["is_champion"]),
        })
    if len(xs) > 1:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=MODEL_LABELS.get(model, model),
            line=dict(color=colour, width=2, dash="dash"), marker=dict(size=8),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>"
                          + MODEL_LABELS.get(model, model) + "</extra>",
        ))

fig.update_layout(height=460, margin=dict(l=60, r=20, t=30, b=10),
                  hovermode="x unified", yaxis_title="Children",
                  legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
st.plotly_chart(fig, use_container_width=True)
st.caption("Point forecasts only — overlaying several interval bands would be "
           "unreadable. Per-scenario intervals are in the table below.")

table = pd.DataFrame(rows)
if len(table):
    st.subheader("Scenario table")
    st.dataframe(table.round(1), use_container_width=True, hide_index=True)

    spread = table.groupby("horizon")["point_forecast"].agg(["min", "max", "mean"])
    spread["model_spread"] = spread["max"] - spread["min"]
    widest = table.loc[table["interval_width"].idxmax()]

    st.subheader("How much does the choice matter?")
    st.markdown(
        "Across the models selected, the **disagreement between models** at each "
        "horizon is:\n\n"
        + "\n".join("- %s: spread of **%s**" % (h, f"{r['model_spread']:,.0f}")
                    for h, r in spread.iterrows())
    )
    st.info(
        "The widest single interval here is **%s** (%s at %s), which is %s the largest "
        "spread between model point forecasts. Where that holds, the honest reading is "
        "that **uncertainty within a model exceeds disagreement between models** — so "
        "picking a different model buys less than the interval width suggests."
        % (f"{widest['interval_width']:,.0f}", widest["model"], widest["horizon"],
           "larger than" if widest["interval_width"] > spread["model_spread"].max()
           else "comparable to")
    )
