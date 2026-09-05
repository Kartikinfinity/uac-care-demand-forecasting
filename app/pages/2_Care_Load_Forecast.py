"""
Page 3 of Part 8 -- "Future Care Load Forecast Chart" (documented Core Module).

Filename prefix 2_ places it directly after Home in Streamlit's sidebar ordering;
the Part-8 numbering counts Home as page 1.
"""
import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Care Load Forecast", page_icon="📈", layout="wide")

from app.lib.artifacts import TARGET_1  # noqa: E402
from app.lib.forecast_page import render  # noqa: E402

render(
    target=TARGET_1,
    page_title="Care Load Forecast",
    subtitle="Children in HHS Care — expected load at each forecast horizon",
    question=(
        "**What's expected, and how confident are we?**\n\n"
        "This is the primary forecasting deliverable for care load: how many "
        "children are expected to be in HHS care at the selected horizon, and how "
        "wide the plausible range around that is."
    ),
)
