"""
early_warning.py -- Capacity-stress early-warning signal (finalised at Day 9).

WHAT THIS IS NOT
----------------
There is no official capacity threshold for the UAC programme in either source
document. Neither the Unified Mentor documentation nor the dataset states how
many children the system can hold. Every number this module produces is a
RELATIVE, DATA-DERIVED PROXY for "load is unusually high by this programme's own
recent standards" -- never an official capacity figure, and never a statement
that capacity has been or will be breached.

That is why `capacity_tier()` returns a qualitative label rather than a
percentage: addendum Section 8 requires the dashboard card to read "Elevated" or
"High" precisely so the proxy cannot be mistaken for an official figure
regardless of how it is captioned.

THE TRIGGER (addendum Section 8)
--------------------------------
At each forecast origin, three tiers off that origin's h=1/7/14 forecasts:

    Watch    h=14 forecast crosses the threshold  -- earliest, least confident
    Warning  h=7  crosses
    Alert    h=1  crosses                          -- latest, most confident

Watch rests on the thinnest out-of-sample residual pool and the longest horizon,
so it is presented as advance, lower-confidence notice -- never as equal in
weight to Alert.

THE THRESHOLD
-------------
Trailing 90th percentile of recent observed load, computed using ONLY
observations at or before the origin. This is the load-bearing invariant here:
"never compute the early-warning threshold using data past the historical origin
being evaluated". A threshold built with hindsight would make backtested lead
times meaningless, so `trailing_threshold` slices strictly to `[.. origin]` and
a test asserts that appending later rows cannot change a historical value.

Only genuinely observed values feed the threshold. Interpolated stock slots are
excluded: a percentile of invented values would set the alarm level from a
straight line the cleaning step drew.

Day 8 built the threshold, tiers and honest labelling. Day 9 adds the
backtested KPIs -- Surge Lead Time and the required false-positive/
false-negative rates -- evaluated ONLY within the development portion, because
"the final test window is never used to tune or report this number".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    CAPACITY_TIER_LABELS,
    EARLY_WARNING_PERCENTILE,
    EARLY_WARNING_TIERS,
    EARLY_WARNING_TRAILING_WINDOW,
)

__all__ = [
    "trailing_threshold",
    "classify_tier",
    "evaluate_origin",
    "capacity_tier",
    "backtest_origin",
    "summarise_backtest",
    "PROXY_DISCLAIMER",
]

PROXY_DISCLAIMER = (
    "Relative, data-derived proxy for unusually high load by this programme's "
    "own recent standards. No official capacity threshold exists in the source "
    "documentation or the dataset; this is not an official capacity figure and "
    "does not indicate that capacity has been or will be breached."
)


def trailing_threshold(
    values,
    origin_pos: int,
    is_observed=None,
    percentile: float = EARLY_WARNING_PERCENTILE,
    window: int = EARLY_WARNING_TRAILING_WINDOW,
) -> float:
    """
    Trailing percentile of observed load as of `origin_pos`, inclusive.

    Strictly backward-looking: the slice ends at the origin, so no value from
    after it can influence the result. Returns NaN rather than a guess when the
    trailing window holds no observed values.

    `is_observed`, when given, excludes interpolated slots -- a percentile taken
    over invented values would set the alarm level from a straight line rather
    than from reported load.
    """
    values = np.asarray(values, dtype=float)
    if origin_pos < 0 or origin_pos >= values.size:
        raise IndexError("origin_pos %d outside the series (size %d)" % (origin_pos, values.size))

    start = max(0, origin_pos - window + 1)
    window_values = values[start : origin_pos + 1]

    if is_observed is not None:
        mask = np.asarray(is_observed, dtype=bool)[start : origin_pos + 1]
        window_values = window_values[mask]

    window_values = window_values[np.isfinite(window_values)]
    if window_values.size == 0:
        return float("nan")
    return float(np.percentile(window_values, percentile))


def classify_tier(forecasts: dict, threshold: float, tiers: dict = EARLY_WARNING_TIERS) -> dict:
    """
    Map one origin's {horizon: forecast} to the tiers it fires.

    Every crossing horizon is reported, plus `highest_tier` -- the most
    confident tier that fired, i.e. the one from the SHORTEST crossing horizon.
    Alert outranks Warning outranks Watch, because a nearer forecast rests on a
    thicker residual pool.
    """
    if not np.isfinite(threshold):
        return {"threshold": float("nan"), "fired": {}, "highest_tier": None,
                "earliest_tier": None, "confidence_note": "threshold unavailable"}

    fired = {}
    for horizon, forecast in forecasts.items():
        if horizon in tiers and np.isfinite(forecast) and forecast > threshold:
            fired[int(horizon)] = tiers[horizon]

    if not fired:
        return {"threshold": threshold, "fired": {}, "highest_tier": None,
                "earliest_tier": None, "confidence_note": "no tier fired"}

    highest = tiers[min(fired)]          # shortest crossing horizon -> most confident
    earliest = tiers[max(fired)]         # longest crossing horizon -> earliest notice
    note = ("Watch is advance, lower-confidence notice: it rests on the longest "
            "horizon and the thinnest out-of-sample residual pool."
            if highest == "Watch" else
            "%s fired from a %d-period-ahead forecast." % (highest, min(fired)))
    return {"threshold": threshold, "fired": fired, "highest_tier": highest,
            "earliest_tier": earliest, "confidence_note": note}


def evaluate_origin(
    values,
    origin_pos: int,
    forecasts: dict,
    is_observed=None,
    percentile: float = EARLY_WARNING_PERCENTILE,
    window: int = EARLY_WARNING_TRAILING_WINDOW,
) -> dict:
    """One origin end-to-end: trailing threshold, then tier classification."""
    threshold = trailing_threshold(values, origin_pos, is_observed, percentile, window)
    result = classify_tier(forecasts, threshold)
    result.update({
        "origin_pos": int(origin_pos),
        "percentile": percentile,
        "window": window,
        "is_proxy": True,
        "disclaimer": PROXY_DISCLAIMER,
    })
    return result


def capacity_tier(forecast: float, threshold: float, elevated_fraction: float = 0.9) -> str:
    """
    Qualitative label for the dashboard's Capacity Breach card.

    Addendum Section 8: "the Capacity Breach Probability card uses a qualitative
    tier label ('Elevated'/'High') rather than a bare percentage, so the proxy
    cannot read as an official figure regardless of its caption." This returns
    that label and nothing numeric.
    """
    if not (np.isfinite(forecast) and np.isfinite(threshold)) or threshold <= 0:
        return CAPACITY_TIER_LABELS[0]
    if forecast > threshold:
        return CAPACITY_TIER_LABELS[2]          # High
    if forecast >= threshold * elevated_fraction:
        return CAPACITY_TIER_LABELS[1]          # Elevated
    return CAPACITY_TIER_LABELS[0]              # Normal


# ======================================================================
# DAY 9 -- BACKTESTED KPIs
# ======================================================================
# Surge Lead Time and the false-positive/false-negative rates are evaluated
# ONLY within the development portion. Addendum Section 8: "the final test
# window is never used to tune or report this number", and invariant: "never
# tune the early-warning percentile against the same window used to report the
# final Surge Lead Time or Capacity Breach Probability numbers." The percentile
# is a frozen convention (90), never searched -- so nothing is tuned here at
# all; the backtest only measures what the frozen rule would have done.


def _first_crossing_after(values, is_observed, start_pos: int, threshold: float,
                          limit: int = None) -> int:
    """
    First position AFTER `start_pos` where the observed series exceeds
    `threshold`. Returns None if it never does within `limit` periods.

    Only genuinely observed values count as a crossing -- an interpolated slot
    is not an event that happened.
    """
    values = np.asarray(values, dtype=float)
    observed = (np.ones(values.size, dtype=bool) if is_observed is None
                else np.asarray(is_observed, dtype=bool))
    end = values.size if limit is None else min(values.size, start_pos + 1 + limit)
    for pos in range(start_pos + 1, end):
        if observed[pos] and np.isfinite(values[pos]) and values[pos] > threshold:
            return pos
    return None


def backtest_origin(values, is_observed, origin_pos: int, forecasts: dict,
                    horizon_limit: int = None, **kwargs) -> dict:
    """
    One historical origin: did the signal fire, and did the series actually
    cross the same threshold afterwards?

    `lead_time` is the addendum's definition verbatim -- "periods between the
    earliest tier firing at a historical origin and the actual series
    subsequently crossing the same threshold". Measured in period-positions, not
    calendar days, consistent with every other offset in this project.

    Outcome is one of:
        true_positive   fired, and the series subsequently crossed
        false_positive  fired, and it never did within the lookahead
        false_negative  did not fire, but the series crossed anyway
        true_negative   did not fire, and it did not cross
    """
    result = evaluate_origin(values, origin_pos, forecasts, is_observed, **kwargs)
    threshold = result["threshold"]
    fired = bool(result["fired"])

    if not np.isfinite(threshold):
        result.update({"outcome": None, "lead_time": None, "crossing_pos": None})
        return result

    limit = horizon_limit if horizon_limit is not None else max(EARLY_WARNING_TIERS)
    crossing = _first_crossing_after(values, is_observed, origin_pos, threshold, limit)

    if fired and crossing is not None:
        outcome, lead = "true_positive", crossing - origin_pos
    elif fired:
        outcome, lead = "false_positive", None
    elif crossing is not None:
        outcome, lead = "false_negative", None
    else:
        outcome, lead = "true_negative", None

    result.update({"outcome": outcome, "lead_time": lead, "crossing_pos": crossing,
                   "horizon_limit": limit})
    return result


def summarise_backtest(origin_results) -> dict:
    """
    Surge Lead Time (median) plus the false-positive/false-negative rates the
    addendum makes "a required metric, not optional".

    The median is used rather than the mean because the lead-time distribution
    is bounded below by 1 and right-skewed, and at these sample sizes one late
    crossing would drag a mean badly.
    """
    results = [r for r in origin_results if r.get("outcome") is not None]
    n = len(results)
    if n == 0:
        return {"n_origins": 0, "median_surge_lead_time": None,
                "false_positive_rate": None, "false_negative_rate": None}

    counts = {k: sum(1 for r in results if r["outcome"] == k)
              for k in ("true_positive", "false_positive",
                        "false_negative", "true_negative")}
    leads = [r["lead_time"] for r in results if r["lead_time"] is not None]
    fired = counts["true_positive"] + counts["false_positive"]
    did_not_fire = counts["true_negative"] + counts["false_negative"]

    return {
        "n_origins": n,
        **counts,
        "n_fired": fired,
        "median_surge_lead_time": float(np.median(leads)) if leads else None,
        "min_surge_lead_time": int(np.min(leads)) if leads else None,
        "max_surge_lead_time": int(np.max(leads)) if leads else None,
        # Of the times it fired, how often was it wrong.
        "false_positive_rate": (counts["false_positive"] / fired) if fired else None,
        # Of the times it stayed silent, how often it should not have.
        "false_negative_rate": (counts["false_negative"] / did_not_fire) if did_not_fire else None,
        "precision": (counts["true_positive"] / fired) if fired else None,
        "lead_time_units": "period-positions (not calendar days)",
    }
