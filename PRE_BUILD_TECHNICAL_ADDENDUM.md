# PRE_BUILD_TECHNICAL_ADDENDUM

This addendum resolves every decision from the pre-build technical review (Phases 0–5) into one frozen implementation specification. It supplements — and does not replace — `UAC_Forecasting_Execution_Roadmap.md`. Where earlier phases left two options open for empirical testing, exactly one is selected below as the pre-registered default, grounded in diagnostic evidence already gathered (structural-break tests, ADF/KPSS, row-count arithmetic, library-behavior checks, correlation checks). No walk-forward model-performance comparison has actually been run yet — nothing below claims otherwise.

## 1. LOCKED roadmap decisions

Original-roadmap commitments that survived five phases of scrutiny unchanged:

- Two primary targets — `Children in HHS Care` (stock) and `Children discharged from HHS Care` (flow) — modeled as two independent tracks, not a combined multi-output model.
- Six required model families, evaluated per target and per horizon: naive/seasonal-naive, moving average, Exponential Smoothing, ARIMA/SARIMA, Random Forest, Gradient Boosting.
- Chronological-only splitting; no random sampling at any stage.
- The intake/exit imbalance is a derived signal (Transferred Out − Discharged), not a third independently-optimized target.
- No fabricated absolute capacity threshold; capacity-stress signals use a relative/statistical proxy, labeled as such everywhere it appears.
- 4 Streamlit Core Modules + 3 User Capabilities, delivered as an 8-page app reading only from pre-generated artifacts — no live training in a page callback.
- Batch-generated artifacts; a single offline `generate.py` as sole producer.
- Streamlit Community Cloud deployment; minimal, pinned dependency set (no deep-learning framework).
- Three deliverables — research paper (25-section structure), Streamlit dashboard, executive summary.
- P0/P1/P2 priority framing; the 15-day, solo-delivery schedule shape, including buffer days at 8 and 14.
- The baseline-beats-gate as the validation floor: every statistical/ML model must beat naive and seasonal-naive on held-out folds.

## 2. Required methodology changes

Everything below is a confirmed delta from the original documentation, finalized here.

**Data construction:** master series built on a period-position index — 767 canonical Sun–Thu template positions plus the 2 confirmed off-template Friday observations (2024-09-13, 2025-04-11), retained as standalone rows at their real dates — 769 positions total, 720 fully real.

**Missing-data treatment:** stock columns (`HHS Care`, `CBP Custody`) interpolated across the 49 gaps, flagged per-column (`is_imputed_<column>`). Flow columns (`Apprehended`, `Transferred Out`, `Discharged`) left as true-missing at gap slots — never zero-filled or interpolated.

**Features:** lags (t-1/7/14), rolling windows (7/14), and forecast horizons (h=1/7/14) are all integer offsets on the period-position index — never calendar-date arithmetic. Rolling statistics use strictly-prior periods only. ML feature rows with NaN inputs (from flow-column gaps) are dropped for Random Forest, count logged; not required for Gradient Boosting once implemented as `HistGradientBoostingRegressor`. Baselines computed on the identical index/masking/target-availability treatment as the six models — no separate code path.

**Forecasting:** Random Forest and Gradient Boosting use direct multi-horizon fitting (3 models per target). ARIMA/SARIMA and Exponential Smoothing are each fit once per target; forecasts at h=1/7/14 come from that single model's native multi-step forecast function — implemented as `SARIMAX` (confirmed native missing-data handling) and `ETSModel` with `missing='drop'` where the input is the masked Discharged series (confirmed necessary: classic `ExponentialSmoothing` silently returns an all-NaN forecast on NaN input). Seasonal period is confirmed at Day 3 against actual within-week seasonality in each target's level, not assumed from the reporting schedule. Training data for every model/fold with origin on or after **2025-02-05** is capped to start no earlier than that date; folds with earlier origins are unaffected. Exception: if a capped training set falls below ~60–80 usable rows for a given fold/target/horizon, that instance uses full-expanding-window data instead. The differenced/relative-target idea for RF/GBR is dropped — the training-window cap already addresses the regime-heterogeneity concern at lower complexity.

**Validation:** one walk-forward harness, fixed fold cadence, evaluates all six models plus a post-hoc ensemble on identical folds — full protocol in Section 5.

**Uncertainty:** empirical residual-quantile intervals restricted to the same 2025-02-05-onward regime already used for training — no separate weighting scheme. Full protocol in Section 6.

**Early-warning logic:** three-tier trigger off the existing h=1/7/14 forecasts, trailing-only 90th-percentile threshold. Full protocol in Section 8.

## 3. Required architecture changes

- `config.py` holds every parameter fixed in this addendum: the 2025-02-05 cutoff, the ~60–80 row floor, the 60-period final test window, the 10-period fold step, the 50-period minimum initial training size, the 90th-percentile early-warning threshold, and one global `RANDOM_SEED` applied to every stochastic fit (Random Forest, HistGB).
- Model/forecast registry carries, per artifact: a content hash of the raw CSV and of the cleaned master series, a generation timestamp, and the winning model/window/ensemble-or-not combination per target/horizon.
- Test suite extended with: period-position row counts (769/720) and the two named Friday dates present as standalone rows; column-specific imputation flags; lag-7 correctness verified row-based, not calendar-based; no current-row leakage into rolling features; NaN-row-drop counts logged; RF/GBR floor enforcement; SARIMAX/ETSModel missing-data behavior (regression test against the exact silent-failure mode found); intervals built only from out-of-sample residuals; coverage check returns a number with a confidence band; early-warning threshold at a historical origin unaffected by appending later rows.
- Fail-fast validation gate at the start of `generate.py`, reusing the above assertions as a runtime guard.
- Failure handling: fail loud at the data-validation/cleaning stage; log and continue past an individual model-fit failure within the batch; dashboard shows an explicit "unavailable" state for any missing/failed combination, never a crash or a silently wrong number.
- Deployment smoke test extended to confirm the deployed app's provenance readout matches the local run.
- One-line provenance readout ("data as of [date], forecasts generated [timestamp]") on the Methodology page; manual-only refresh policy stated explicitly in the README and on that page; one-line note on the hosting platform's free-tier cold-start behavior.

## 4. Final model architecture

Six required families only — no additional family is added. Phase 2 evaluated dynamic regression/SARIMAX-with-exogenous, regularized lag regression, external modern boosting libraries, an additional statistical baseline, and (proactively) a regime-switching model; none cleared the bar for a required addition at this sample size (145–720 usable rows depending on window).

- Naive, seasonal-naive, moving average — computed on the identical treatment as the six evaluated models.
- Exponential Smoothing → `ETSModel` (state-space), `missing='drop'` where input is the masked Discharged series.
- ARIMA/SARIMA → `SARIMAX`.
- Random Forest → `RandomForestRegressor`, NaN-affected feature rows dropped, count logged.
- Gradient Boosting → `HistGradientBoostingRegressor` (not the classic class) — confirmed native NaN support, zero new dependencies.
- A simple post-hoc average of the champion statistical and champion ML forecast is computed and evaluated on the same folds and final test window as every other candidate, for every target/horizon — included in the comparison, not pre-declared a winner.
- Excluded from the required set (Section 10): SARIMAX-with-exogenous, regularized regression, external boosting libraries, regime-switching models, additional baselines.

## 5. Final validation protocol

- **Top-level split:** final held-out test window = the most recent 60 real observations (≈12 weeks, late Sept – Dec 21, 2025), reserved and touched exactly once, after champions are frozen. Development portion = the ~660 real observations before it, used for all walk-forward CV and selection.
- **Walk-forward folds:** expanding-window, chronological, within the development portion. Minimum initial training size 50 periods; step size 10 periods between successive fold origins. No fold has training data later than its own test data.
- **Fold spacing:** fixed at 10 periods for every model/target — not decided ad hoc at implementation time.
- **Recent-regime evaluation:** for every fold, compute predictions under both the full-expanding-window rule and the 2025-02-05-capped rule (identical for folds whose origin precedes the cutoff — not a separate fold set). Report aggregate metrics both across the full development portion and restricted to folds with origin on or after 2025-02-05. If the two rankings disagree, the restricted (recent-regime) ranking governs champion selection.
- **Model comparison:** all six models plus the ensemble evaluated on identical folds and horizons per target; every candidate must beat naive and seasonal-naive to be considered.
- **Champion selection:** best aggregate error among candidates that clear the baseline gate, subject to a practical-equivalence margin — within that margin, the simpler/more stable candidate wins over the numerically lowest one. Decided independently per target and per horizon.
- **Leakage prevention:** rolling/lag features verified strictly backward-shifted by an automated test; the training-window cap is never applied retroactively to a fold whose origin predates 2025-02-05; stock-column interpolation is a one-time global step performed before folding, and any fold whose training-cutoff or near-term test window falls inside or adjacent to an interpolated gap is flagged and checked at Day 5, not assumed negligible; the early-warning threshold is always computed trailing-only.

## 6. Final uncertainty architecture

Empirical residual-quantile intervals, built from out-of-sample walk-forward residuals restricted to folds with origin on or after **2025-02-05** — the same cutoff as training, not a separately-tuned weighting scheme. Native ARIMA/SARIMAX and ETS confidence intervals are computed and shown as a secondary diagnostic only.

**Coverage evaluation:** primary evidence is the pooled outcome across all post-cutoff walk-forward fold test observations per target/horizon/model — actual-vs-interval hit rate compared to the nominal level, reported with a binomial confidence band sized to the actual achieved N (expected to be small — low tens, not hundreds — and disclosed as such). Secondary, confirmatory check on the single final held-out test window. A gap between nominal and empirical coverage outside the confidence band is disclosed in the Limitations section, not silently widened after the fact.

**Derived imbalance signal:** Transferred Out's forward value, for this purpose only, uses the same baseline (seasonal-naive/moving-average) treatment already required for every series — not a full champion-selection track, keeping it a derived signal rather than a third target. Combined uncertainty uses Var(A−B) = Var(A) + Var(B) − 2·Cov(A,B), with Cov measured from the two components' actual paired out-of-sample residuals once they exist at Day 8. The simplified independence form is used only if that measured correlation supports it. The raw-series proxy correlation already checked (0.657 raw / 0.074 first-difference / 0.112 detrended) sets the pre-registered expectation of near-independence — it is a prior, not the final answer.

## 7. Final production architecture

Training → six models plus ensemble fit inside the Day 5–8 walk-forward harness under the rules above → `generate.py` produces flat forecast/metrics/interval files plus a provenance sidecar (data hash, timestamp, winning configuration) → Streamlit app reads only from these artifacts, no live fitting → Streamlit Community Cloud deployment, smoke-tested for provenance match → no automated monitoring or retraining. Two optional, manual, on-demand scripts (reusing the Day-3 break-scan and the coverage-calibration check) let a future maintainer re-check for a new regime shift or re-verify calibration if the CSV is ever replaced — not scheduled, not automatic. The manual-only refresh policy is stated plainly wherever a reader might otherwise assume the dashboard is continuously live.

## 8. Final early-warning methodology

**Trigger:** at each forecast origin, three tiers from that origin's h=1/7/14 forecasts — Watch (h=14 crosses threshold), Warning (h=7 crosses), Alert (h=1 crosses). Watch carries the least confidence (thinnest OOS residual pool) and is presented as advance/lower-confidence notice, not equal-weight with Alert.

**Threshold:** trailing 90th percentile of recent observed load, computed using only data available as of each origin — never full-history hindsight. Fixed at the 90th percentile as a stated convention, never searched or tuned against the window used to report the final KPI numbers.

**Evaluation:** Surge Lead Time = periods between the earliest tier firing at a historical origin and the actual series subsequently crossing the same threshold, summarized (median) across historical origins restricted to the development portion only — the final test window is never used to tune or report this number. False-positive/false-negative rate is tracked alongside Surge Lead Time as a required metric, not optional.

**Dashboard presentation:** the Capacity Breach Probability card uses a qualitative tier label ("Elevated"/"High") rather than a bare percentage, so the proxy cannot read as an official figure regardless of its caption.

**Derived-signal handling:** per Section 6.

## 9. 15-day roadmap amendments

- **Day 1:** add `RANDOM_SEED` and the Section 3 parameter list to `config.py`.
- **Day 2:** build the 769-slot period-position master series; apply per-column imputation flags; compute the data hash; add the fail-fast validation gate.
- **Day 3:** run the structural-break scan (already confirmed: dominant break 2025-01-12, secondary break 2025-02-05 — now locked as the training/residual cutoff); test actual within-week seasonality per target before fixing the SARIMA seasonal order.
- **Day 4:** implement lags/rolling/horizons as period-based, strictly backward; drop NaN-affected RF feature rows, log count; compute baselines on the identical treatment as the six models.
- **Day 5:** fix fold cadence (step 10, minimum initial training 50) and the 60-period final test window; implement the 2025-02-05 training cap with its ~60–80-row fallback; enforce the RF/GBR floor; check interpolation-adjacent fold contamination.
- **Day 6:** implement `SARIMAX` and `ETSModel` (`missing='drop'` for Discharged) as single-fit-per-target; confirm seasonal order from Day 3.
- **Day 7:** implement `HistGradientBoostingRegressor` and `RandomForestRegressor`, direct multi-horizon; persist OOS residuals per fold with origin dates.
- **Day 8:** champion selection with the practical-equivalence margin and recent-regime-restricted ranking; compute and evaluate the ensemble; measure component-residual correlation for the imbalance signal; confirm baseline-based Transferred-Out forward value.
- **Day 9:** implement the imbalance-signal uncertainty formula; run the coverage-calibration check (pooled-fold primary, final-holdout confirmatory); implement the 3-tier early-warning trigger with trailing 90th-percentile threshold and required false-positive tracking, evaluated only within the development portion.
- **Day 10–11:** qualitative-tier capacity card; horizon labels show calendar-day equivalents alongside period counts; provenance readout on the Methodology page.
- **Day 12:** extend the deployment smoke test to check provenance match.
- **Day 13:** report states the horizon-vs-decision-timescale gap plainly; reports the full comparison matrix, not just the winning path; caveats any RF/GBR feature importances for collinearity; gives the omitted-variable/2.5%-reconciliation limitation prominent, unhedged placement.
- **Day 14:** README/executive summary finalize seed, versioning, refresh-policy, and hosting cold-start disclosures.
- **Day 15:** unchanged — verification and submission only.

## 10. Explicit non-goals

Regime-switching/Markov-switching models. External boosting libraries (LightGBM/XGBoost/CatBoost). Regularized regression and SARIMAX-with-exogenous as required models, or even as optional side analyses — excluded entirely to keep frozen scope unambiguous. Additional statistical baselines (Theta, drift, STL+naive). Bootstrap resampling and conformal-prediction frameworks. Separate quantile-loss model variants. Any live data feed, live model-performance monitoring, or automated drift detection. A database, feature store, orchestration framework, container/Kubernetes deployment, message queue, or dedicated experiment-tracking platform. Scheduled or automatic retraining. A continuously-weighted recency scheme for residual pooling (a single hard cutoff is used instead — Section 6).

## 11. Implementation invariants

- Never compute a lag, rolling window, or forecast horizon using calendar-date arithmetic — period-position offsets only.
- Never zero-fill or interpolate a flow column at a gap slot — true-missing only.
- Never include the current row in its own rolling-window feature.
- Never apply the 2025-02-05 training cap to a fold whose origin predates it.
- Never build a prediction interval from in-sample residuals.
- Never compute the early-warning threshold using data past the historical origin being evaluated.
- Never tune the early-warning percentile against the same window used to report the final Surge Lead Time or Capacity Breach Probability numbers.
- Never present the capacity-proxy figure as an official threshold.
- Never let the Streamlit app fit or refit a model as part of serving a page.
- Never report a model-comparison "winner" without the practical-equivalence margin check.
- Never claim a walk-forward result, coverage number, or performance figure that has not actually been produced by running the pipeline.

## 12. Final GO / NO-GO

**GO.** Every open fork identified across Phases 0–5 is resolved to exactly one rule here, each grounded in evidence already gathered — structural-break tests, ADF/KPSS, row-count arithmetic, library-behavior checks, correlation checks — rather than in walk-forward results that require the pipeline to exist first. Nothing here requires technology beyond what Phase 2 already justified (`SARIMAX`, `ETSModel`, `HistGradientBoostingRegressor` — all standard components of the already-approved stack). The parameters fixed in Sections 3 and 5 (the 2025-02-05 cutoff, the 60/50/10 validation numbers, the 90th-percentile threshold, the ~60–80 row floor) are frozen for this build and should not be reopened during implementation without new evidence of the same rigor used to reach them. Proceed to build.
