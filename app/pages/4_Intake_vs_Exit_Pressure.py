"""
Page 5 of Part 8 -- "Intake vs. Exit Pressure (Early Warning)" [DOC].

Covers the documented objective "estimate future imbalance between intake and
exits" and the secondary objective "provide early warnings for healthcare
planners".

Every capacity figure on this page is a RELATIVE, DATA-DERIVED PROXY. No
official capacity threshold exists in either source document, and the caveat is
repeated wherever a number appears rather than parked in a footnote.
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

st.set_page_config(page_title="Intake vs Exit Pressure", page_icon="⚠️", layout="wide")

from src.config import (  # noqa: E402
    COL_DISCHARGED, COL_TRANSFERRED, EARLY_WARNING_PERCENTILE, TARGET_1,
)
from app.lib.artifacts import (  # noqa: E402
    horizon_label, load_early_warning, load_history, load_imbalance,
    load_kpis, load_provenance, load_sensitivity, provenance_caption,
)
st.title("Intake vs. Exit Pressure")
st.caption("Is pressure building, and how much lead time would we have?")

history = load_history()
imbalance = load_imbalance()
backtest = load_early_warning()
sensitivity = load_sensitivity()
kpis = load_kpis()

# The disclaimer is read from the provenance artifact rather than imported from
# src.signals: the app must depend on generated files only, so the caveat the
# user sees is the exact one the pipeline recorded, not a second copy that could
# drift away from it.
provenance = load_provenance()
st.markdown(provenance_caption(provenance))
st.error("**This is a proxy, not an official threshold.** "
         + provenance["early_warning"]["disclaimer"])
st.divider()

# ----------------------------------------------------------------------
# Net pressure: history and forward projection
# ----------------------------------------------------------------------
st.subheader("Net flow (Transferred out of CBP − Discharged from HHS)")

dates = pd.to_datetime(history["parsed_date"])
net = history[COL_TRANSFERRED].astype(float) - history[COL_DISCHARGED].astype(float)
tail = st.slider("History shown (periods)", 30, 300, 120, step=10)
idx = slice(max(0, len(history) - tail), len(history))

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates.iloc[idx], y=net.iloc[idx], mode="lines",
                         name="Net flow (observed)", line=dict(color="#1f4e79", width=2),
                         hovertemplate="%{x|%d %b %Y}<br>%{y:+,.0f}<extra>Net flow</extra>"))
fig.add_hline(y=0, line_dash="dot", line_color="#888",
              annotation_text="balance", annotation_position="right")
fig.update_layout(height=380, margin=dict(l=60, r=20, t=30, b=10),
                  hovermode="x unified", yaxis_title="Children (intake − exits)")
st.plotly_chart(fig, use_container_width=True)
st.caption("Positive means more children entering HHS care than leaving it. "
           "Gaps are periods with no published flow figure — never zero-filled.")

st.subheader("Forward pressure")
show = imbalance[["horizon", "transferred_forecast", "discharged_forecast",
                  "net_pressure", "std", "uncertainty_form", "measured_correlation",
                  "n_paired_residuals", "interpretation"]].copy()
show.insert(1, "horizon_label", show["horizon"].map(horizon_label))
st.dataframe(show.round(2), use_container_width=True, hide_index=True)

worst = imbalance.iloc[imbalance["net_pressure"].abs().idxmax()]
st.info(
    "Combined uncertainty uses **%s** form. The full "
    "`Var(A−B) = Var(A) + Var(B) − 2·Cov(A,B)` is the default; the independence "
    "simplification applies only where the *measured* paired residual correlation "
    "earns it. Across horizons the standard deviation (%.1f–%.1f) is comparable to "
    "or larger than the net pressure itself, so the sign of the signal is not "
    "reliable at these sample sizes — read it as pressure direction, not magnitude."
    % (imbalance["uncertainty_form"].iloc[0], imbalance["std"].min(), imbalance["std"].max())
)

# ----------------------------------------------------------------------
# Early warning
# ----------------------------------------------------------------------
st.divider()
st.subheader("Early-warning backtest")

target = st.selectbox("Target", sorted(backtest["target"].unique()), index=0)
frozen = sensitivity[(sensitivity["target"] == target)
                     & sensitivity["is_frozen_operating_point"]].iloc[0]

cards = st.columns(4)
cards[0].metric("Median Surge Lead Time", "%.1f periods" % frozen["median_surge_lead_time"]
                if pd.notna(frozen["median_surge_lead_time"]) else "—")
cards[1].metric("False-positive rate", "%.0f%%" % (frozen["false_positive_rate"] * 100)
                if pd.notna(frozen["false_positive_rate"]) else "—")
cards[2].metric("False-negative rate", "%.0f%%" % (frozen["false_negative_rate"] * 100)
                if pd.notna(frozen["false_negative_rate"]) else "—")
cards[3].metric("Capacity state",
                kpis[kpis["target"] == target]["capacity_tier"].iloc[0])

lead = frozen["median_surge_lead_time"]
if pd.notna(lead) and lead <= 3:
    st.warning(
        "**A %.1f-period median lead time is operationally thin.** At this cadence "
        "that is roughly %.0f–%.0f calendar days of notice — less than most staffing "
        "or placement decisions need. This is the horizon-versus-decision-timescale "
        "gap, stated plainly rather than presented as an operational capability."
        % (lead, lead, lead * 1.4)
    )
if pd.notna(frozen["false_negative_rate"]) and frozen["false_negative_rate"] > 0.3:
    st.warning(
        "**The signal misses %.0f%% of threshold crossings** for this target. It is "
        "conservative by construction: it rarely fires falsely (%.0f%% false-positive "
        "rate), but it stays silent through most genuine crossings. Treat a firing as "
        "informative and silence as uninformative."
        % (frozen["false_negative_rate"] * 100, frozen["false_positive_rate"] * 100)
    )

st.markdown("##### Threshold sensitivity")
st.caption(
    "The %dth percentile is the FROZEN operating point and the only one the figures "
    "above are quoted at. This grid is precomputed disclosure — it shows how much the "
    "signal depends on that choice. It is not a tuning control: the percentile was "
    "never searched against the window used to report these numbers."
    % EARLY_WARNING_PERCENTILE
)
grid = sensitivity[sensitivity["target"] == target][
    ["percentile", "is_frozen_operating_point", "n_fired", "true_positive",
     "false_positive", "false_negative", "median_surge_lead_time",
     "false_positive_rate", "false_negative_rate"]
].copy()
st.dataframe(grid.round(3), use_container_width=True, hide_index=True)

fired = backtest[(backtest["target"] == target) & backtest["highest_tier"].notna()]
st.markdown("##### Origins where a tier fired (development portion only)")
st.caption("The held-out window is never used to report or tune these numbers.")
if len(fired):
    st.dataframe(
        fired[["origin_date", "threshold", "highest_tier", "earliest_tier",
               "fired_horizons", "outcome", "lead_time"]].round(1),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No tier fired at any development-portion origin for this target.")

st.caption(
    "Tiers: **Watch** fires from the 14-period forecast (earliest, least confident — "
    "thinnest residual pool), **Warning** from 7 periods, **Alert** from 1 period "
    "(latest, most confident)."
)
