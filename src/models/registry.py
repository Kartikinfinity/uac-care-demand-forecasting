"""
registry.py -- Which model wins per target/horizon, and why.

The roadmap's Day-8 artifact. A registry entry is never a bare model name: it
carries the metric numbers, the gate outcome, the practical-equivalence
evidence, the bias and stability diagnostics, and the data provenance the
decision was made against. The Day-8 validation checkpoint is that "every
selection claim traces to a specific row in the Day 5-7 metrics tables", and
that is only auditable if the trace ships with the decision.

The registry is written once by `src/evaluation/run_selection.py` and read
thereafter -- by `generate.py` at Day 9 and by the dashboard as metadata. It is
a record, not a decision-maker: the rule lives in `src/evaluation/selection.py`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    MODEL_REGISTRY_PATH,
    PRACTICAL_EQUIVALENCE_LEVEL,
    PRACTICAL_EQUIVALENCE_RESAMPLES,
    SELECTION_SCOPE,
    SELECTION_WINDOW_RULE,
)

__all__ = ["build_registry", "write_registry", "read_registry", "champion_for"]


def build_registry(entries: list, provenance: dict = None) -> dict:
    """
    Assemble the registry document.

    `entries` is one record per (target, horizon) as returned by the selection
    driver, each already carrying its own evidence trail.
    """
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": {
            "gate": "must beat BOTH naive and seasonal-naive on MAE, strictly",
            "ranking_scope": SELECTION_SCOPE,
            "window_rule": SELECTION_WINDOW_RULE,
            "practical_equivalence": (
                "paired bootstrap over per-observation absolute errors, "
                "%d resamples at the %.0f%% level; a candidate whose interval "
                "spans zero is tied with the numerical leader"
                % (PRACTICAL_EQUIVALENCE_RESAMPLES, PRACTICAL_EQUIVALENCE_LEVEL * 100)
            ),
            "tie_break": "prefer un-biased, then prefer lower complexity rank",
            "baselines_eligible": True,
        },
        "provenance": provenance,
        "entries": entries,
    }


def write_registry(registry: dict, path: Path = MODEL_REGISTRY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_registry(path: Path = MODEL_REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def champion_for(registry: dict, target: str, horizon: int) -> dict:
    """The registry entry for one target/horizon, or None."""
    for entry in registry.get("entries", []):
        if entry["target"] == target and int(entry["horizon"]) == int(horizon):
            return entry
    return None
