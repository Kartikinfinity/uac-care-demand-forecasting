"""
Home.py -- Entry point for the UAC forecasting dashboard.

Day-10 scope is the skeleton plus the two mandated forecast pages; the remaining
pages from Part 8 arrive at Day 11. What is already load-bearing here is the
provenance readout and the standing caveats: a reader should meet the honest
framing before they meet a number, not after.

This app NEVER trains a model. Every figure it shows was produced offline by
`src/forecast/generate.py` and read from a flat file.
"""
import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="UAC Care Load Forecasting", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from app.lib.artifacts import load_provenance, provenance_caption  # noqa: E402

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("Unaccompanied Alien Children (UAC) Program · U.S. Department of Health "
           "and Human Services · independent analytical exercise")

try:
    provenance = load_provenance()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(provenance_caption(provenance))
st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("What this dashboard is")
    st.markdown(
        "Short-term forecasts of **children in HHS care** and **discharge demand**, "
        "with an intake-versus-exit pressure signal and an early-warning layer.\n\n"
        "Every model was validated by expanding-window walk-forward cross-validation "
        "on 65 chronological folds, and no model was accepted unless it beat a naive "
        "and a seasonal-naive baseline on held-out folds."
    )
    st.subheader("Available now")
    st.page_link("pages/2_Care_Load_Forecast.py", label="Care Load Forecast", icon="📈")
    st.page_link("pages/3_Discharge_Demand_Forecast.py", label="Discharge Demand Forecast", icon="📤")
    st.caption("Historical trends, early warning, model comparison, scenario "
               "comparison and methodology pages follow at Day 11.")

with right:
    st.subheader("Read these first")
    st.warning(
        "**No official capacity threshold exists.** Neither the programme "
        "documentation nor the dataset states a capacity figure. Every "
        "capacity-stress signal here is a *relative, data-derived proxy* against "
        "the programme's own recent levels — not an official threshold, and not a "
        "statement that capacity has been or will be breached."
    )
    st.info(
        "**Horizons are reporting periods, not days.** The programme reports "
        "Sunday–Thursday, so a 7-period horizon is roughly 9 calendar days. Both "
        "units are shown wherever a horizon appears."
    )
    st.info(
        "**The forecasts are not continuously live.** Data is refreshed manually: "
        "replace the source CSV and re-run the offline generation script. The "
        "dashboard never fits a model itself."
    )

st.divider()
st.subheader("Current champions")
st.caption(
    "Selected at Day 8 on recent-regime walk-forward evidence, under a rule that "
    "prefers the simpler model when candidates cannot be told apart. Baselines are "
    "eligible champions — and on this data they won."
)
cols = st.columns(3)
for i, entry in enumerate(provenance["winning_configuration"]):
    with cols[i % 3]:
        st.metric(
            label="%s · h=%d" % (entry["target"].replace("Children ", ""), entry["horizon"]),
            value=entry["champion"].replace("_", " "),
        )
