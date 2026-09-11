# Executive Summary

**Forecasting Care Load and Placement Demand — HHS Unaccompanied Alien Children
Program**

Data through 2025-12-21. Prepared for operational planning use.

---

## 1. The problem

Reporting today is descriptive: it says how many children are in care now, not
how many will be. By the time a surge shows in the daily count, much of the time
needed to respond — opening shelter capacity, scheduling medical staff,
assigning caseworkers — has already been spent. The scale of the risk is in the
programme's own history: over roughly three years the number of children in care
ranged from 1,972 to 11,516 — a 5.8-fold swing. Planning from today's count
alone has no way to see a move of that size coming.

## 2. What was built

A forecasting system covering the two quantities that drive resource decisions —
the number of **children in HHS care** and the number of **discharges** — at
three look-aheads, each given as a range rather than a single number, plus an
early-warning view that flags load heading toward unusually high levels by the
programme's own recent standards. It is delivered as a live eight-page dashboard
covering historical trends, both forecasts, the balance between arrivals and
exits, and a full account of the method.

## 3. What data was used

The programme's own published figures: roughly three years of reporting,
covering 720 reporting days from 2023-01-12 onward. Reporting runs Sunday
through Thursday rather than every calendar day, which is why look-aheads here
are counted in reporting days. For transparency, the raw file needed routine
cleaning first — blank rows at the end, and number formatting in one column —
all of it recorded and reversible. No figures were altered.

## 4. What the forecasting achieved

For **children in HHS care**, forecasts land within **10 children** of the
actual count one reporting day ahead, **70** about nine days ahead, and **148**
about twenty days ahead — against a current level of roughly 2,484. For
**discharges**, which run in the low tens rather than the thousands, forecasts
land within **4** one day ahead and **8** about nine days ahead.

One result should be stated rather than buried. Straightforward methods —
essentially, expecting tomorrow to resemble today — beat considerably more
sophisticated ones in all 6 of the 6 cases tested. That is a finding about the
data, not a shortcut: the count of children in care moves slowly enough day to
day that little is gained by modelling it more elaborately.

## 5. How early it can identify pressure

Backtested across 65 historical points, the signal gives a median of 1.0
reporting periods of notice for children in HHS care (about one day) and 2.5 for
discharges (about two to three days).

**The level that triggers a warning is not an official capacity figure.** No
such figure exists in the published data or documentation, so the system
compares against what has been unusually high for this programme recently. A
warning means "high by recent standards", never "capacity is about to be
exceeded".

## 6. What operational value it provides

Against the three questions the programme needs answered:

**How many children will be in care in the coming days?** Answered — the
strongest result. The near-term forecast lands within a fraction of a percent of
the current level, with a range showing how much to trust it.

**Will discharge capacity keep pace with arrivals?** Direction only. The
expected gap between arrivals and exits is smaller than the uncertainty around
it, so read it as "no clear signal either way", not as a forecast of relief or
pressure.

**When should shelters, staff, and caseworkers be scaled up ahead of a surge?**
**Not answered — this system cannot answer it.** Warning arrives one to three
days ahead; opening capacity, onboarding staff, and scaling sponsor vetting take
weeks. A better forecast would not close that gap, because the data supports
looking about twenty days ahead at most. Use this for near-term tempo —
rostering, transport, discharge-queue priority — not capacity planning.

## 7. Major limitations

- **There is no official capacity number.** Every "high load" statement is
  relative to recent history, not to a real capacity ceiling.
- **Published arrivals and exits do not fully account for the change in the
  number of children in care.** Where this can be checked, the count rises by an
  average of 42 more per period than they explain — at least one route into care
  is missing from this data. The forecasts work around that gap, and would be
  caught out if that unseen route changed.
- **Forecasts become markedly less certain further out.** The twenty-day range
  is several times wider than the one-day range.
- **The recent record is short.** The caseload changed character partway through
  the data, so recent conclusions rest on few observations and may shift.
- **The discharge warning signal misses roughly half of what it should catch**
  and should not be relied on by itself.
- **The data is a single national total.** It cannot be broken down by region or
  facility, and it does not explain *why* numbers move.

## 8. How this should be used

As **a planning input alongside existing judgement — not as an automatic
trigger.** Three practical points:

- Read the range, not the headline number. Where a forecast is flat, all the
  information is in how wide the range is.
- Treat a warning as a prompt to look, not a conclusion — especially the
  discharge signal, which is unreliable alone.
- The figures are **not live.** They refresh only when someone deliberately
  reruns the system on new data; nothing updates or retrains on a schedule, and
  the date the data runs to is shown on every page. Repeated runs on the same
  data give identical results by design. The dashboard is hosted on a free
  service that sleeps after disuse, so the first visit of the day may take
  around half a minute to load.

---

*Full method, results and limitations: `reports/research_paper.md`. Every figure
in this summary is taken from a generated result file and recorded in
`reports/executive_summary_evidence.json`.*
