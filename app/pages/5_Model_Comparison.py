"""
Page 6 of Part 8 -- "Model Selection & Comparison" [DOC].

Covers the documented Core Module and the secondary objective "compare
statistical vs machine-learning forecasting approaches". Shows the FULL
comparison matrix, including the results that do not flatter the complex
models, because the roadmap requires the evidence trail rather than the
conclusion alone.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Model Comparison", page_icon="📋", layout="wide")

from src.config import SELECTION_SCOPE, SELECTION_WINDOW_RULE  # noqa: E402
from app.lib.artifacts import (  # noqa: E402
    MODEL_LABELS, horizon_label, load_champions, load_comparison, load_holdout,
    load_provenance, provenance_caption,
)

st.title("Model Comparison & Accuracy")
st.caption("Which model wins, and the evidence that decided it")

comparison = load_comparison()
champions = load_champions()
holdout = load_holdout()

st.markdown(provenance_caption(load_provenance()))
st.divider()

BASELINES = {"naive", "seasonal_naive", "moving_average"}

controls = st.columns([3, 2, 2])
with controls[0]:
    target = st.selectbox("Target", sorted(comparison["target"].unique()))
with controls[1]:
    horizon = st.selectbox("Horizon", sorted(comparison["horizon"].unique()),
                           format_func=horizon_label)
with controls[2]:
    scope = st.selectbox(
        "Evidence scope",
        [SELECTION_SCOPE, "common_support"],
        format_func=lambda s: ("Recent regime (governs selection)"
                               if s == SELECTION_SCOPE else "All development folds"),
        help="The addendum makes the recent-regime ranking governing where the two "
             "disagree. All-development folds carry more statistical power but "
             "include the pre-2025 regime.",
    )

cell = comparison[
    (comparison["fold_scope"] == scope)
    & (comparison["window_rule"] == SELECTION_WINDOW_RULE)
    & (comparison["target"] == target)
    & (comparison["horizon"] == horizon)
].copy()

if cell.empty:
    st.error("No metrics for this combination.")
    st.stop()

ref = cell[cell["model"].isin(["naive", "seasonal_naive"])]["MAE"].min()
cell["beats_both_baselines"] = cell["MAE"] < ref
cell.loc[cell["model"].isin(BASELINES), "beats_both_baselines"] = pd.NA
cell["family"] = cell["model"].map(
    lambda m: "baseline" if m in BASELINES
    else ("statistical" if m in {"sarima", "exponential_smoothing"}
          else ("ensemble" if m == "ensemble" else "machine learning"))
)
cell["model"] = cell["model"].map(lambda m: MODEL_LABELS.get(m, m))

st.subheader("Full comparison matrix")
st.dataframe(
    cell[["model", "family", "n_scored", "MAE", "RMSE", "MAPE", "sMAPE", "MASE",
          "ME_bias", "beats_both_baselines"]].sort_values("MAE").round(3),
    use_container_width=True, hide_index=True,
    column_config={"beats_both_baselines": st.column_config.CheckboxColumn(
        "Beats both baselines", help="Must beat naive AND seasonal-naive, strictly. "
                                     "Blank for the baselines themselves.")},
)
st.caption(
    "All models scored on identical walk-forward folds with common support, so the "
    "numbers are directly comparable. MAPE is unstable for the discharge target "
    "(values near zero) — read sMAPE and MASE there instead."
)

entry = champions[(champions["target"] == target) & (champions["horizon"] == horizon)]
if len(entry):
    row = entry.iloc[0]
    st.subheader("Selection decision")
    cards = st.columns(4)
    cards[0].metric("Champion", MODEL_LABELS.get(row["champion"], row["champion"]))
    cards[1].metric("Numerical leader",
                    MODEL_LABELS.get(row["numerical_leader"], "none cleared gate")
                    if pd.notna(row["numerical_leader"]) else "none cleared gate")
    cards[2].metric("Full-dev champion",
                    MODEL_LABELS.get(row["champion_full_dev"], row["champion_full_dev"]))
    cards[3].metric("Rankings", "agree" if row["rankings_agree"] else "disagree")

    st.info("**Why this champion.** " + str(row["reason"]))

    if not row["rankings_agree"]:
        st.warning(
            "**The two evidence scopes disagree here.** On all development folds "
            "(n≈46–63) the same rule selects **%s**; on the recent regime (n=12–15) it "
            "selects **%s**. The addendum makes the recent-regime ranking governing. "
            "This is a finding about statistical power, not model quality: at 12–15 "
            "observations a paired bootstrap cannot separate most candidates, so the "
            "rule falls back to preferring the simpler model."
            % (MODEL_LABELS.get(row["champion_full_dev"], row["champion_full_dev"]),
               MODEL_LABELS.get(row["champion"], row["champion"]))
        )

    if row["champion"] in BASELINES:
        st.warning(
            "**A baseline is the champion here, and that is the result — not a "
            "placeholder.** Complexity is kept only where the validation evidence "
            "earns it. Forcing a statistical or ML model into a cell whose bootstrap "
            "interval spans zero would be selecting on noise."
        )

st.divider()
st.subheader("Held-out confirmation")
st.caption(
    "The final 60 observations were reserved and touched exactly once, after "
    "champions were frozen. Nothing here fed back into selection."
)
hold = holdout[(holdout["target"] == target) & (holdout["horizon"] == horizon)]
if len(hold):
    cols = st.columns(3)
    cols[0].metric("Holdout points", len(hold))
    cols[1].metric("Holdout MAE", f"{hold['abs_error'].mean():,.1f}")
    covered = pd.to_numeric(hold["covered"], errors="coerce").fillna(0).sum()
    cols[2].metric("Interval coverage", "%.0f%%" % (100 * covered / len(hold)))
    st.caption("n=%d per cell — far too few for a precise coverage estimate. Reported "
               "as confirmatory, not as a headline accuracy claim." % len(hold))
else:
    st.info("No scorable holdout points for this combination.")
