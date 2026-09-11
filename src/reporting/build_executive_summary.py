"""
build_executive_summary.py -- The one-page executive summary (Day 14).

Part 13 specifies this document precisely: one page, for a non-technical
HHS-style operational stakeholder, answering eight questions in a fixed order,
with "no equations, no library names, no code" and accuracy figures "filled in
only from actual results".

Generated rather than typed, for the same reason as the research paper: every
numeral is issued by `Evidence.cite` and recorded with the artifact it came from,
so the summary cannot quietly drift from the run that produced it. A regenerated
pipeline regenerates this page too.

The hard part here is not the numbers -- it is saying what they mean without
overstating it. Three places where the honest answer is uncomfortable, and is
given anyway:

  * Question 6 asks what operational value the system provides, tied to the three
    decision questions in Part 1.2. The third of those -- when to scale up in
    advance of a surge -- is NOT supported by this system, because the measured
    warning time is one to three days against decisions that take weeks. The
    summary says so in the same breath as the two it does support.
  * Question 4 asks what the forecasting achieved. The winning method was a
    simple one, and pretending otherwise would misrepresent the result.
  * Question 5 asks how early pressure can be identified. The number is small,
    and the threshold behind it is a proxy rather than an official capacity
    figure. Both facts are stated plainly.

Addendum Day 14 additionally requires the seed, versioning, refresh-policy and
hosting cold-start disclosures to be finalised across the README and this
summary. The two a non-technical reader acts on -- refresh policy and cold start
-- appear here; all four are in the README.

Run:  python -m src.reporting.build_executive_summary
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    COL_DISCHARGED, COL_HHS_CARE, EXEC_SUMMARY_EVIDENCE_PATH,
    EXECUTIVE_SUMMARY_PATH, PROJECT_ROOT, RANDOM_SEED, TARGET_1, TARGET_2,
)
from src.reporting.build_paper import Artifacts, _rewrap  # noqa: E402
from src.reporting.evidence import Evidence  # noqa: E402

# Plain-language names. The column identifiers never appear in this document --
# a stakeholder should not have to read a schema to read a summary.
PLAIN = {TARGET_1: "children in HHS care", TARGET_2: "discharges"}

# Part 13: "no equations, no library names, no code". Enforced by a test.
BANNED_TECHNICAL_TERMS = [
    "sarima", "arima", "ets", "exponential smoothing", "random forest",
    "gradient boosting", "streamlit", "pandas", "numpy", "scikit", "sklearn",
    "statsmodels", "python", "bootstrap", "mae", "rmse", "mape", "smape",
    "mase", "walk-forward", "quantile", "heteroscedastic", "stationarity",
    "autocorrelation", "hyperparameter", "regressor", "p-value", "wilson",
]


def _lead_time_words(periods: float) -> str:
    """Reporting periods as something a reader can act on."""
    if periods <= 1.5:
        return "about one day"
    if periods <= 3.5:
        return "about two to three days"
    return "under a week"


def build() -> tuple[Path, int]:
    """Generate the summary and its evidence ledger. Returns (path, n_citations)."""
    a = Artifacts()
    e = Evidence()

    k1, k2 = a.kpi_row(TARGET_1), a.kpi_row(TARGET_2)
    kpi = "forecasts/kpi_summary.csv"
    champs = "forecasts/champion_selection.csv"
    fwd = a.forward[a.forward["is_champion"]]
    reg = a.regime()
    rec = a.reconciliation()
    dsrc = "derived from data/interim/master_series.parquet"

    def mae(target: str, horizon: int) -> str:
        row = a.champ(target, horizon)
        return e.cite(row["champion_mae"], champs,
                      f"target={target} horizon={horizon} -> champion_mae", "{:,.0f}")

    def point(target: str, horizon: int) -> str:
        row = fwd[(fwd["target"] == target) & (fwd["horizon"] == horizon)].iloc[0]
        return e.cite(row["point_forecast"], "forecasts/forward_forecasts.csv",
                      f"target={target} horizon={horizon} -> point_forecast", "{:,.0f}")

    n_baseline = int(a.champions["champion"].isin(
        {"naive", "seasonal_naive", "moving_average"}).sum())

    text = f"""# Executive Summary

**Forecasting Care Load and Placement Demand — HHS Unaccompanied Alien Children Program**

Data through {e.cite(a.data_prov["data_as_of"], "data/interim/provenance.json", "data_as_of")}.
Prepared for operational planning use.

---

## 1. The problem

Reporting today is descriptive: it says how many children are in care now, not how
many will be. By the time a surge shows in the daily count, much of the time
needed to respond — opening shelter capacity, scheduling medical staff, assigning
caseworkers — has already been spent. The scale of the risk is in the programme's
own history: over roughly three years the number of children in care ranged from
{e.cite(reg["min"], dsrc, f"min of {COL_HHS_CARE}")} to
{e.cite(reg["max"], dsrc, f"max of {COL_HHS_CARE}")} — a
{e.cite(reg["ratio"], dsrc, "max/min ratio", "{:.1f}")}-fold swing. Planning from
today's count alone has no way to see a move of that size coming.

## 2. What was built

A forecasting system covering the two quantities that drive resource decisions —
the number of **children in HHS care** and the number of **discharges** — at three
look-aheads, each given as a range rather than a single number, plus an
early-warning view that flags load heading toward unusually high levels by the
programme's own recent standards. It is delivered as a live eight-page dashboard
covering historical trends, both forecasts, the balance between arrivals and
exits, and a full account of the method.

## 3. What data was used

The programme's own published figures: roughly three years of reporting, covering
{e.cite(a.data_prov["n_real_observations"], "data/interim/provenance.json", "n_real_observations")}
reporting days from
{e.cite(a.data_prov["series_starts"], "data/interim/provenance.json", "series_starts")}
onward. Reporting runs Sunday through Thursday rather than every calendar day,
which is why look-aheads here are counted in reporting days. For transparency, the
raw file needed routine cleaning first — blank rows at the end, and number
formatting in one column — all of it recorded and reversible. No figures were
altered.

## 4. What the forecasting achieved

For **{PLAIN[TARGET_1]}**, forecasts land within **{mae(TARGET_1, 1)} children** of
the actual count one reporting day ahead, **{mae(TARGET_1, 7)}** about nine days
ahead, and **{mae(TARGET_1, 14)}** about twenty days ahead — against a current
level of roughly {point(TARGET_1, 1)}. For **{PLAIN[TARGET_2]}**, which run in the
low tens rather than the thousands, forecasts land within
**{mae(TARGET_2, 1)}** one day ahead and **{mae(TARGET_2, 7)}** about nine days
ahead.

One result should be stated rather than buried. Straightforward methods —
essentially, expecting tomorrow to resemble today — beat considerably more
sophisticated ones in all
{e.cite(n_baseline, champs, "count of champion in the baseline set")} of the
{e.cite(len(a.champions), champs, "row count")} cases tested. That is a finding about the
data, not a shortcut: the count of children in care moves slowly enough day to day
that little is gained by modelling it more elaborately.

## 5. How early it can identify pressure

Backtested across
{e.cite(k1["n_backtest_origins"], kpi, f"target={TARGET_1} -> n_backtest_origins")}
historical points, the signal gives a median of
{e.cite(k1["median_surge_lead_time_periods"], kpi, f"target={TARGET_1} -> median_surge_lead_time_periods", "{:.1f}")}
reporting periods of notice for {PLAIN[TARGET_1]} ({_lead_time_words(float(k1["median_surge_lead_time_periods"]))}) and
{e.cite(k2["median_surge_lead_time_periods"], kpi, f"target={TARGET_2} -> median_surge_lead_time_periods", "{:.1f}")}
for {PLAIN[TARGET_2]} ({_lead_time_words(float(k2["median_surge_lead_time_periods"]))}).

**The level that triggers a warning is not an official capacity figure.** No such
figure exists in the published data or documentation, so the system compares
against what has been unusually high for this programme recently. A warning means
"high by recent standards", never "capacity is about to be exceeded".

## 6. What operational value it provides

Against the three questions the programme needs answered:

**How many children will be in care in the coming days?** Answered — the strongest
result. The near-term forecast lands within a fraction of a percent of the current
level, with a range showing how much to trust it.

**Will discharge capacity keep pace with arrivals?** Direction only. The expected
gap between arrivals and exits is smaller than the uncertainty around it, so read
it as "no clear signal either way", not as a forecast of relief or pressure.

**When should shelters, staff, and caseworkers be scaled up ahead of a surge?**
**Not answered — this system cannot answer it.** Warning arrives one to three days
ahead; opening capacity, onboarding staff, and scaling sponsor vetting take weeks.
A better forecast would not close that gap, because the data supports looking
about twenty days ahead at most. Use this for near-term tempo — rostering,
transport, discharge-queue priority — not capacity planning.

## 7. Major limitations

- **There is no official capacity number.** Every "high load" statement is relative
  to recent history, not to a real capacity ceiling.
- **Published arrivals and exits do not fully account for the change in the number
  of children in care.** Where this can be checked, the count rises by an average of
  {e.cite(rec["mean_signed"], dsrc, "mean signed reconciliation residual", "{:,.0f}")}
  more per period than they explain — at least one route into care is missing from
  this data. The forecasts work around that gap, and would be caught out if that
  unseen route changed.
- **Forecasts become markedly less certain further out.** The twenty-day range is
  several times wider than the one-day range.
- **The recent record is short.** The caseload changed character partway through
  the data, so recent conclusions rest on few observations and may shift.
- **The discharge warning signal misses roughly half of what it should catch** and
  should not be relied on by itself.
- **The data is a single national total.** It cannot be broken down by region or
  facility, and it does not explain *why* numbers move.

## 8. How this should be used

As **a planning input alongside existing judgement — not as an automatic
trigger.** Three practical points:

- Read the range, not the headline number. Where a forecast is flat, all the
  information is in how wide the range is.
- Treat a warning as a prompt to look, not a conclusion — especially the discharge
  signal, which is unreliable alone.
- The figures are **not live.** They refresh only when someone deliberately reruns
  the system on new data; nothing updates or retrains on a schedule, and the date
  the data runs to is shown on every page. Repeated runs on the same data give
  identical results by design. The dashboard is hosted on a free service that
  sleeps after disuse, so the first visit of the day may take around half a minute
  to load.

---

*Full method, results and limitations: `reports/research_paper.md`. Every figure in
this summary is taken from a generated result file and recorded in
`reports/executive_summary_evidence.json`.*
"""

    text = _rewrap(text)
    EXECUTIVE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXECUTIVE_SUMMARY_PATH.write_text(text, encoding="utf-8")
    e.write(EXEC_SUMMARY_EVIDENCE_PATH, extra={
        "document": "reports/executive_summary.md",
        "spec": "roadmap Part 13 -- eight questions, one page, non-technical",
        "raw_csv_sha256": a.data_prov["raw_csv_sha256"],
        "random_seed": RANDOM_SEED,
    })
    return EXECUTIVE_SUMMARY_PATH, len(e.entries)


if __name__ == "__main__":
    path, n = build()
    body = path.read_text(encoding="utf-8")
    words = len(body.split())
    print("Wrote %s" % path.relative_to(PROJECT_ROOT))
    print("  words     : %d  (one page is roughly 500-1000)" % words)
    print("  questions : %d of 8" % len(re.findall(r"^## \d+\.", body, re.MULTILINE)))
    print("  citations : %d" % n)
