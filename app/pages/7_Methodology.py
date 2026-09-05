"""
Page 8 of Part 8 -- "Methodology / Data Information" [REC].

Transparency page. Its purpose is to put every documented limitation in the
product itself rather than only in the research paper: a reader who never opens
the paper should still meet the 450 blank rows, the Sunday-Thursday cadence, the
absent capacity threshold, the malformed KPI table and the unexplained asterisk.

Also carries the provenance readout the addendum requires here specifically.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Methodology & Data", page_icon="📖", layout="wide")

from src.config import (  # noqa: E402
    COL_APPREHENDED,
    COL_CBP_CUSTODY,
    COL_DISCHARGED,
    COL_HHS_CARE,
    COL_TRANSFERRED,
    FINAL_TEST_WINDOW,
    FORECAST_HORIZONS,
    MIN_INITIAL_TRAINING,
    RANDOM_SEED,
    TRAINING_CAP_DATE,
    WALK_FORWARD_STEP,
)
from app.lib.artifacts import (  # noqa: E402
    load_coverage,
    load_history,
    load_provenance,
    provenance_caption,
)

st.title("Methodology & Data Information")
st.caption("What was done, what the data cannot support, and where each number comes from")

provenance = load_provenance()
history = load_history()

st.markdown(provenance_caption(provenance))
st.code(
    "raw CSV SHA-256        %s\n"
    "master series SHA-256  %s\n"
    "forecasts generated    %s"
    % (provenance["raw_csv_sha256"], provenance["master_series_sha256"],
       provenance["generated_at_utc"]),
    language=None,
)
st.caption("Every artifact this dashboard reads carries these hashes, so a stale copy "
           "is detectable rather than silently displayed as current.")

st.divider()
st.header("Known limitations")
st.caption("From the source-verification audit. Each was found by inspecting the actual "
           "files, not assumed.")

LIMITATIONS = [
    ("The CSV has 1,170 rows but only 720 contain data",
     "Rows 721-1,170 are entirely blank across all six columns. Any statistic computed "
     "on the raw row count is corrupted. The pipeline truncates to the 720 real rows as "
     "its very first data-quality gate."),
    ("Reporting is Sunday-Thursday, not daily",
     "Across the 720 real rows: Mon 145, Tue 149, Wed 147, Thu 147, Fri 2, Sat 0, "
     "Sun 130. Reindexing to a 7-day calendar would have made roughly a third of the "
     "series synthetic. Every lag, rolling window and forecast horizon in this project "
     "is therefore counted in reporting periods, never calendar days."),
    ("`Children in HHS Care` is string-typed in the source",
     "Every value carries a thousands separator, because every value is at least 1,000. "
     "This is the primary forecasting target, so it is cast explicitly during cleaning, "
     "with a test asserting zero parse failures."),
    ("The official KPI table is malformed",
     "The source documentation contains two consecutive rows where the second row's KPI "
     "*name* is literally the first row's *description*. Treated as 4 well-defined KPIs "
     "plus 1 flagged ambiguous item, rather than inventing a plausible-sounding name "
     "for the fifth."),
    ("No official capacity threshold exists anywhere in either source",
     "Confirmed by full-text extraction of the documentation and full-column review of "
     "the dataset. Every capacity-stress signal in this application is a relative, "
     "data-derived proxy against the programme's own recent levels, and says so "
     "wherever it appears."),
    ("The `apprehended` column carries a footnote asterisk with no footnote text",
     "No definition exists anywhere in the extracted source. An unexplained asterisk on "
     "a government figure often flags a scope caveat, so the column is used strictly per "
     "its plain documented meaning and the gap is recorded rather than explained away."),
    ("`Children in HHS Care` swings 5.8x across the observed window",
     "From 11,516 (December 2023) to 1,972 (August 2025), then a mild rebound. This is a "
     "structural regime change, not noise around a stable mean. Training is capped at the "
     "%s regime boundary, and stationarity was tested rather than assumed."
     % TRAINING_CAP_DATE),
    ("Flow columns do not reconcile exactly against the stock column",
     "Only about 2.5% of periods reconcile exactly, so the series cannot be treated as a "
     "closed accounting identity. Both targets are modelled directly and never derived "
     "from the others."),
]
for title, detail in LIMITATIONS:
    with st.expander(title):
        st.write(detail)

st.divider()
st.header("How the forecasts were validated")

left, right = st.columns(2)
with left:
    st.markdown(
        "**Chronological only.** No random splitting at any stage. A random split would "
        "let a model train on dates that fall after its test dates, and — because of the "
        "5.8x regime shift — would scatter high-load and low-load periods across both "
        "sides, testing models on regimes they had already seen.\n\n"
        "**Expanding-window walk-forward.** 65 folds, origins spaced %d reporting "
        "periods apart, a minimum of %d periods of initial training, horizons %s.\n\n"
        "**Held-out window.** The most recent %d real observations were reserved and "
        "touched exactly once, after champions were frozen."
        % (WALK_FORWARD_STEP, MIN_INITIAL_TRAINING,
           ", ".join("h=%d" % h for h in FORECAST_HORIZONS), FINAL_TEST_WINDOW)
    )
with right:
    st.markdown(
        "**Baseline gate.** No statistical or machine-learning model was accepted unless "
        "it beat *both* naive and seasonal-naive on held-out folds.\n\n"
        "**Practical equivalence.** Where a paired bootstrap could not separate two "
        "candidates, the simpler one wins. Baselines are eligible champions — and on "
        "this data they won every cell.\n\n"
        "**Intervals.** Empirical quantiles of out-of-sample walk-forward residuals, "
        "restricted to the current regime. Never in-sample residuals, and never a "
        "model-internal confidence interval.\n\n"
        "**Seed %d** on every stochastic fit." % RANDOM_SEED
    )

st.subheader("Interval calibration, measured")
coverage = load_coverage()
st.dataframe(
    coverage[["target", "horizon", "champion", "n", "empirical_coverage",
              "nominal_coverage", "band_low", "band_high", "covers_nominal"]].round(3),
    use_container_width=True, hide_index=True,
)
missed = coverage[~coverage["covers_nominal"]]
st.warning(
    "**%d of %d cells fall outside their binomial confidence band** — those intervals "
    "are measurably too narrow. Residual pools hold only 12-15 observations, so interval "
    "endpoints rest on very few points. This gap is disclosed rather than closed by "
    "widening the bands after the fact." % (len(missed), len(coverage))
)

st.divider()
st.header("Data dictionary")

MEANINGS = {
    COL_APPREHENDED: "Daily intake volume",
    COL_CBP_CUSTODY: "Active CBP care load",
    COL_TRANSFERRED: "Flow into HHS system",
    COL_HHS_CARE: "Active HHS care load",
    COL_DISCHARGED: "Successful sponsor placements",
}
STOCKS = {COL_CBP_CUSTODY, COL_HHS_CARE}
ROLES = {COL_HHS_CARE: "TARGET 1", COL_DISCHARGED: "TARGET 2"}

rows = []
for col, meaning in MEANINGS.items():
    series = history[col].astype(float)
    rows.append({
        "Column": col,
        "Documented meaning": meaning,
        "Type": "stock" if col in STOCKS else "flow",
        "Role": ROLES.get(col, "predictor / signal component"),
        "Missing handling": ("interpolated, flagged per column" if col in STOCKS
                             else "left genuinely missing, never zero-filled"),
        "Min": series.min(),
        "Max": series.max(),
        "Reported values": int(series.notna().sum()),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.header("Pipeline")
st.code(
    "raw CSV\n"
    "  -> clean.py         truncate blanks, parse dates, reindex to 769 period-\n"
    "                      positions, interpolate stocks, leave flows missing, hash\n"
    "  -> build_features   lags 1/7/14, rolling 7/14, calendar - strictly backward\n"
    "  -> walk_forward     65 expanding-window folds, identical for every model\n"
    "  -> models           3 baselines + SARIMA + ETS + RF + HistGB + ensemble\n"
    "  -> selection        baseline gate, practical equivalence, simplicity tie-break\n"
    "  -> generate.py      forward forecasts, intervals, KPIs, provenance sidecar\n"
    "  -> this dashboard   reads flat files ONLY, never fits a model",
    language=None,
)
st.info(
    "**Refresh policy: manual only.** This dashboard is not a live system. To update it, "
    "replace the source CSV and re-run the offline generation pipeline. Nothing retrains "
    "on a schedule, and no page trains anything at any time.\n\n"
    "Intended for Streamlit Community Cloud, whose free tier may take around 30 seconds "
    "to wake on first load after a period of inactivity."
)
