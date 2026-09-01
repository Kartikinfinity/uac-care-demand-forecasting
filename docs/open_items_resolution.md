# Open-Items Resolution — Day-7 Handoff Cleanup

Every unresolved item carried out of Day 7, closed with repository evidence.
No item is left open. Nothing here implements Day 8.

---

## 1. `ROLLING_MIN_PERIODS` — RESOLVED, permissive setting kept

**Issue.** Day 4 emitted rolling features with `min_periods=1` (mean) and `2`
(variance) while a comment claimed strict full windows. The Day-5 audit fixed
the comment and moved the values into `config.py`, but flagged the *choice*
itself as an unresolved decision that had to be frozen before the ML stage.

**Evidence.**

| Question | Measurement |
|---|---|
| Do the sources mandate a minimum? | **No.** Zero occurrences of `min_period`, "minimum observation", "partial/incomplete window", "at least N" in either governing document. The official documentation asks only for "7-day and 14-day rolling mean and variance". |
| What does the choice actually affect? | **Flow-derived features only.** Stock columns are fully interpolated: 0.0% of their rolling cells fall below a full window under either setting. |
| What would strict windows cost? | `rolling_7` on a flow column: 253 values voided (32.9% of rows). `rolling_14`: 437 voided (56.8%). Feature rows carrying any NaN — exactly what Random Forest drops — rise from **131/755 (17.4%) to 424/755 (56.2%)**, costing RF a further **293 training rows**. |
| Which setting matches the addendum's stated expectation? | **Permissive.** The addendum expects ML feature NaN "from flow-column gaps". Verified: all 10 folds with a NaN in the RF prediction row were `lag_*` columns; no `rolling_*` column contributed. Strict windows would invert that. |
| Is the exact value even binding? | **No.** The smallest observation count the data actually reaches is **2** for `rolling_7` and **8** for `rolling_14`. A floor of 1 vs 2 changes nothing on this dataset. |

**Decision.** Keep `ROLLING_MIN_PERIODS_MEAN = 1`, `ROLLING_MIN_PERIODS_VAR = 2`.
Strict windows are not required by any source, would more than triple the NaN
row rate, and would turn the addendum's deliberately *bounded* Random Forest
handicap into the dominant effect on a dataset the roadmap already calls
moderate-sized.

**Changes.** `src/config.py` — the five evidence points recorded inline as the
justification. Four tests added in `tests/test_features.py` pinning the
semantics, the stock-column non-effect, the non-binding floor, and the presence
of the `rolling_{w}_nobs_{col}` companion columns that expose the true sample
size behind every rolling statistic.

**Verification.** No feature regeneration or Day-7 re-run was required, because
the resolution is to keep current behaviour. `tests/test_features.py` — 7 passed.

---

## 2. Three of six recent-regime cells won by a baseline — RESOLVED

**Issue.** Day 7 reported that baselines won 3 of 6 recent-regime
target/horizon cells, leaving champion selection ambiguous going into Day 8.

**Evidence.** A paired bootstrap (10,000 resamples, seed 42) over
per-observation absolute errors on the recent-regime pools, comparing the best
baseline against the best non-baseline candidate in each cell:

| Target | h | n | best baseline | best complex | difference | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| Discharged | 1 | 14 | naive 4.14 | ETS 3.82 | −0.33 | [−2.71, +2.29] | not distinguishable |
| Discharged | 7 | 13 | mov-avg 6.61 | ETS 6.78 | +0.17 | [−1.07, +1.63] | not distinguishable |
| Discharged | 14 | 12 | s-naive 5.33 | ETS 7.33 | +2.00 | [+0.08, +4.05] | **distinguishable — baseline better** |
| HHS Care | 1 | 15 | naive 10.13 | SARIMA 11.44 | +1.31 | [−2.94, +5.51] | not distinguishable |
| HHS Care | 7 | 14 | naive 70.21 | ETS 61.52 | −8.70 | [−35.36, +23.05] | not distinguishable |
| HHS Care | 14 | 14 | naive 147.64 | ETS 108.74 | −38.90 | [−97.82, +0.59] | not distinguishable |

**Finding.** At n = 12–15 per cell, **five of six differences are
indistinguishable from zero**, and the single distinguishable one **favours the
baseline**. The apparent "3 of 6 baseline wins" is not evidence that baselines
are superior, nor that they are inferior — it is evidence that these pools are
too small to separate most candidates. Selecting the numerically lowest error
would be selecting on noise.

**Decision — the champion-selection rule.** Implemented in
`src/evaluation/selection.py`:

1. **Gate** (roadmap Part 6). A non-baseline candidate is viable only if it
   beats **both** naive and seasonal-naive, strictly. A tie is not an
   improvement.
2. **Practical equivalence** (addendum §5). Among viable candidates, any whose
   paired-bootstrap difference from the numerical leader spans zero is tied
   with it. The margin is a bootstrap, **not a fixed percentage** — a fixed
   margin cannot know that one cell has 12 observations and another has 65.
3. **Tie-break by simplicity.** Within the tied set the lowest
   `MODEL_COMPLEXITY_ORDER` rank wins, implementing the addendum's "the
   simpler/more stable candidate wins over the numerically lowest one".
4. **Baselines are eligible champions.** If nothing clears the gate, the best
   baseline *is* the champion and is reported as the result — never overridden
   by a more complex model that failed.

The complexity ordering is measured, not asserted — read from the persisted
artifacts: baselines 0 estimated parameters; SARIMA 4–6; ETS 8–11; Random
Forest 300 trees / ~29,700 nodes; HistGradientBoosting 300 boosted trees.

**Changes.** `src/evaluation/selection.py` (new), `src/config.py`
(`MODEL_COMPLEXITY_ORDER`, `BASELINE_MODELS`, bootstrap parameters,
`BASELINES_ARE_ELIGIBLE_CHAMPIONS`), `tests/test_selection.py` (new, 20 tests).

**Verification.** 20 tests pass, including: a baseline winning is preserved; a
genuinely better complex model still wins; a noisy tie breaks toward
simplicity; a small but *perfectly consistent* edge is still detected as real
(the counterpart case — pairing removes shared variation, so a consistent 2%
edge is signal, not noise); and a test that reproduces the measured
Discharged-h=14 finding from the real artifacts.

**Not done here:** applying the rule to pick champions. That is Day 8.

---

## 3. Stale generated artifacts — RESOLVED

**Issue.** The Day-7 harness fix (models must forecast to the date they are
scored on, `effective_lead` rather than nominal horizon) changed behaviour
*after* the Day-5 and Day-6 artifacts were generated.

**Evidence.** Fold 57 is the only fold where lead ≠ horizon (leads 3/9/16 vs
horizons 1/7/14). Reconstructing what each artifact's seasonal-naive forecast
must have been:

| Artifact | Verdict |
|---|---|
| `baseline_predictions.csv` | PRE-FIX → stale |
| `statistical_predictions.csv` | PRE-FIX → stale |
| `ml_predictions.csv` | POST-FIX → current |

Regenerating Day 5 confirmed the impact: **40 of 144 metric cells changed**,
exclusively on `seasonal_naive` — the only baseline whose forecast depends on
the horizon. Max ΔMAE 1.667; one cell's scored N moved 63 → 62 because the
deeper lookback landed on a true-missing flow slot.

**Decision.** Regenerate every artifact so the committed documentation matches
what current code produces.

**Changes.** `docs/day5_baseline_metrics.md`, `docs/day6_statistical_metrics.md`
and `docs/day7_ml_metrics.md` regenerated from a full re-run of all three
drivers; `forecasts/*.csv` regenerated (gitignored).

---

## 4. Stale configuration comment — RESOLVED

`src/config.py` carried `"These will be populated during Days 5-7 ... Stubs here
for structure"` above `MODEL_FAMILIES`, long after all seven families were
implemented and evaluated. Replaced with an accurate description.

---

## 5. Silently swallowed exceptions — RESOLVED

`run_statistical.py` and `run_ml.py` contained `except Exception: pass` around
artifact summary statistics, model slimming, and provenance loading. A failure
there would have produced a silently incomplete artifact. All three now print an
explicit warning, and a failed summary read is recorded on the artifact itself
as `summary_error` so a downstream reader sees *why* a field is empty.

---

## 6. Dead no-op branch — RESOLVED

`selection.py` contained an `if ...: pass` placeholder written during this
pass. Removed on review before commit.

---

## 7. Items checked and confirmed NOT to require action

| Checked | Finding |
|---|---|
| TODO / FIXME / XXX / HACK markers | None in `src/`, `tests/`, `app/`, `docs/`. |
| Commented-out workarounds | None. |
| Skipped tests | Two conditional `pytest.skip`s, both guarding integration checks against gitignored artifacts. Neither skips in the current tree; both would skip on a fresh clone before the pipeline is run. Deliberate — the alternative is a 30-minute unit-test suite. Documented here rather than hidden. |
| Remaining broad `except Exception` | 8, all in model fit/forecast paths where the addendum explicitly requires "log and continue past an individual model-fit failure". Each records a reason string surfaced in the reports, and each is covered by an abstention test. |
| `docs/AUDIT_REPORT.md` staleness | No such file in this repository (that path belongs to a different project). |
| Holdout integrity | Untouched by every change in this pass; re-verified by the validation suite. |
| Frozen parameters | All 12 re-verified unchanged. |
