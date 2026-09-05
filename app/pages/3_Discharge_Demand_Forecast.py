"""
Page 4 of Part 8 -- "Discharge Demand Forecast Panel" (documented Core Module).
"""
import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Discharge Demand Forecast", page_icon="📤", layout="wide")

from app.lib.artifacts import TARGET_2  # noqa: E402
from app.lib.forecast_page import render  # noqa: E402

render(
    target=TARGET_2,
    page_title="Discharge Demand Forecast",
    subtitle="Children discharged from HHS Care — expected placement throughput",
    question=(
        "**Will discharge throughput keep pace?**\n\n"
        "Discharge demand is the exit side of the system. Read alongside the care "
        "load forecast, it indicates whether placements are keeping up with intake "
        "or whether pressure is accumulating."
    ),
)
