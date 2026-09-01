"""
selection.py -- The champion-selection rule.

This module defines the RULE. It does not run it: applying it to pick champions
per target/horizon is Day 8's job. What lives here is the machinery that makes
that decision unambiguous, reproducible and testable.

Why this exists
---------------
Day 7 closed with an open question: three of six recent-regime target/horizon
cells were won by a BASELINE rather than by a statistical or ML model. Left
unresolved, Day 8 would have faced a tempting shortcut -- pick the numerically
lowest error, or quietly prefer the more sophisticated family.

Measurement settles it. A paired bootstrap over per-observation absolute errors
on the recent-regime pools (n = 12-15 per cell) gives:

    target / horizon              best complex - best baseline    95% CI
    Discharged h=1                        -0.33            [ -2.71, +2.29]
    Discharged h=7                        +0.17            [ -1.07, +1.63]
    Discharged h=14                       +2.00            [ +0.08, +4.05]  *
    HHS Care  h=1                         +1.31            [ -2.94, +5.51]
    HHS Care  h=7                         -8.70            [-35.36, +23.05]
    HHS Care  h=14                       -38.90            [-97.82,  +0.59]

Five of six cells are NOT distinguishable at these sample sizes. The one that
is (*) favours the BASELINE. So the honest conclusion is not "baselines are
better" nor "complex models are better" -- it is that at 12-15 observations per
cell most of these differences are noise, and a rule that ignores that would be
selecting on noise.

The rule
--------
1. GATE (roadmap Part 6). A non-baseline candidate is viable only if it beats
   BOTH naive and seasonal-naive. Beating one is not a pass.
2. PRACTICAL EQUIVALENCE (addendum Section 5). Among viable candidates, any
   whose paired-bootstrap difference from the best performer includes zero is
   treated as tied with it.
3. TIE-BREAK BY SIMPLICITY. Within the tied set, the lowest
   `MODEL_COMPLEXITY_ORDER` rank wins -- "the simpler/more stable candidate wins
   over the numerically lowest one".
4. BASELINES ARE ELIGIBLE CHAMPIONS. If no candidate clears the gate, the best
   baseline is the champion and is reported as such. A baseline winning is a
   RESULT, not a failure to be overridden.

The margin is deliberately not a fixed percentage. A fixed margin cannot know
that one cell has 12 observations and another has 65; a paired bootstrap adapts
to the sample actually achieved, which is the whole difficulty here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    BASELINE_MODELS,
    BASELINES_ARE_ELIGIBLE_CHAMPIONS,
    MODEL_COMPLEXITY_ORDER,
    PRACTICAL_EQUIVALENCE_LEVEL,
    PRACTICAL_EQUIVALENCE_RESAMPLES,
    PRACTICAL_EQUIVALENCE_SEED,
)

__all__ = [
    "complexity_rank",
    "paired_absolute_errors",
    "paired_bootstrap_difference",
    "is_practically_equivalent",
    "passes_baseline_gate",
    "select_champion",
]


def complexity_rank(model: str) -> int:
    """Position in the measured complexity ordering; unknown models rank last."""
    try:
        return MODEL_COMPLEXITY_ORDER.index(model)
    except ValueError:
        return len(MODEL_COMPLEXITY_ORDER)


def paired_absolute_errors(predictions: pd.DataFrame, model_a: str, model_b: str):
    """
    Absolute errors for two models on EXACTLY the same test points.

    Pairing is what makes the comparison powerful at small N: the two models see
    identical actuals, so the fold-to-fold variation that dominates the raw
    error distribution cancels. Rows are matched on (fold_id, horizon) and only
    points where both models produced a scorable forecast are kept.
    """
    needed = {"model", "fold_id", "horizon", "y_true", "y_pred"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError("predictions is missing column(s): %s" % sorted(missing))

    frames = {}
    for name in (model_a, model_b):
        sub = predictions[predictions["model"] == name]
        if sub.empty:
            raise ValueError("no rows for model %r" % name)
        frames[name] = sub.set_index(["fold_id", "horizon"])[["y_true", "y_pred"]]

    joined = frames[model_a].join(frames[model_b], lsuffix="_a", rsuffix="_b", how="inner")
    keep = (
        joined[["y_true_a", "y_pred_a", "y_true_b", "y_pred_b"]].notna().all(axis=1)
    )
    joined = joined[keep]
    if joined.empty:
        return np.array([]), np.array([])

    if not np.allclose(joined["y_true_a"], joined["y_true_b"]):
        raise ValueError("paired rows disagree on the actual value -- not the same test points")

    return (
        np.abs(joined["y_true_a"] - joined["y_pred_a"]).to_numpy(),
        np.abs(joined["y_true_b"] - joined["y_pred_b"]).to_numpy(),
    )


def paired_bootstrap_difference(
    errors_a,
    errors_b,
    resamples: int = PRACTICAL_EQUIVALENCE_RESAMPLES,
    level: float = PRACTICAL_EQUIVALENCE_LEVEL,
    seed: int = PRACTICAL_EQUIVALENCE_SEED,
) -> dict:
    """
    Bootstrap confidence interval for mean(|e_a|) - mean(|e_b|).

    Resamples PAIRS, so the interval reflects uncertainty in the DIFFERENCE
    rather than in either model's error separately. Negative means model A has
    the lower error. Seeded, so the answer is reproducible.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired errors must be the same length: %d vs %d" % (a.size, b.size))
    n = a.size
    if n == 0:
        return {"n": 0, "mean_difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "distinguishable": False}

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diff[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(boot, [tail, 100.0 - tail])
    return {
        "n": int(n),
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "distinguishable": bool(lo > 0 or hi < 0),
    }


def is_practically_equivalent(errors_a, errors_b, **kwargs) -> bool:
    """True when the paired difference is not distinguishable from zero."""
    return not paired_bootstrap_difference(errors_a, errors_b, **kwargs)["distinguishable"]


def passes_baseline_gate(metrics_row: dict, baseline_maes: dict) -> bool:
    """
    Roadmap Part 6: a candidate must beat BOTH naive and seasonal-naive.

    Compared against the BETTER of the two, and strictly -- a tie is not an
    improvement. Baselines themselves are not gated against each other.
    """
    mae = float(metrics_row["MAE"])
    refs = [float(v) for k, v in baseline_maes.items()
            if k in ("naive", "seasonal_naive") and np.isfinite(v)]
    if not refs or not np.isfinite(mae):
        return False
    return mae < min(refs)


def select_champion(predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """
    Apply the rule to ONE target/horizon/window-rule cell.

    `predictions` must already be restricted to that cell and to scorable,
    common-support rows; `metrics` must carry one row per model with an `MAE`
    column. Returns the champion plus the full evidence trail behind the
    decision -- never a bare name.

    This function is the RULE. Day 8 calls it; nothing here runs it.
    """
    maes = metrics.set_index("model")["MAE"].astype(float).to_dict()
    baseline_maes = {m: v for m, v in maes.items() if m in BASELINE_MODELS}
    finite = {m: v for m, v in maes.items() if np.isfinite(v)}
    if not finite:
        return {"champion": None, "reason": "no candidate produced a finite MAE"}

    viable = [m for m, v in finite.items()
              if m in BASELINE_MODELS or passes_baseline_gate({"MAE": v}, baseline_maes)]

    if not any(m not in BASELINE_MODELS for m in viable):
        if not BASELINES_ARE_ELIGIBLE_CHAMPIONS:
            return {"champion": None, "reason": "no non-baseline candidate cleared the gate"}
        best_baseline = min(baseline_maes, key=lambda m: baseline_maes[m])
        return {
            "champion": best_baseline,
            "champion_mae": baseline_maes[best_baseline],
            "gate_cleared_by": [],
            "reason": "no candidate beat both naive and seasonal-naive; the best "
                      "baseline is the champion and is preserved as the result",
            "tied_with_best": [best_baseline],
        }

    leader = min(viable, key=lambda m: finite[m])
    tied = []
    evidence = {}
    for model in viable:
        if model == leader:
            tied.append(model)
            continue
        try:
            ea, eb = paired_absolute_errors(predictions, model, leader)
        except ValueError:
            continue
        stats = paired_bootstrap_difference(ea, eb)
        evidence[model] = stats
        if not stats["distinguishable"]:
            tied.append(model)

    champion = min(tied, key=complexity_rank)
    return {
        "champion": champion,
        "champion_mae": finite[champion],
        "numerical_leader": leader,
        "numerical_leader_mae": finite[leader],
        "gate_cleared_by": sorted(m for m in viable if m not in BASELINE_MODELS),
        "tied_with_best": sorted(tied, key=complexity_rank),
        "bootstrap_evidence": evidence,
        "reason": ("champion is the simplest candidate not distinguishable from the "
                   "numerical leader" if champion != leader else
                   "champion is the numerical leader and no simpler candidate ties it"),
    }
