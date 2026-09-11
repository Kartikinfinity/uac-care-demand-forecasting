"""
evidence.py -- The ledger that makes the Day 13 checkpoint mechanical.

The checkpoint reads: "every number in the paper is cross-checked against an
actual output file -- zero figures written from memory or assumption."

Proofreading cannot establish that. A number typed from memory looks exactly
like a number read from a CSV once it is on the page. So the paper is GENERATED,
and every numeral in it is produced by `Evidence.cite(...)`, which returns the
rendered string AND records where the value came from.

That gives two things a human read-through cannot:

  1. `reports/paper_evidence.json` -- a claim-by-claim audit trail: rendered
     text, source artifact, and the locator within it.
  2. A test that scans the finished paper for numeric tokens and fails if one
     appears that the ledger never issued. A figure typed straight into the
     prose does not survive that scan.

The prose is authored. The numbers are not.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["Evidence"]


class Evidence:
    """
    Records every value the paper interpolates, then writes the audit trail.

    `cite` is deliberately the only way to get a number into the document. It is
    slightly awkward to call, and that is the point: the friction is what stops
    a figure being typed inline "just this once".
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def cite(self, value: Any, source: str, locator: str,
             fmt: str = "{:,.0f}") -> str:
        """
        Render `value` for the paper and record its provenance.

        source   -- repository-relative path of the artifact it was read from.
        locator  -- where inside that artifact, precisely enough that a reader
                    can go and check it by hand (e.g. "target=... horizon=7 -> MAE").
        """
        if value is None:
            rendered = "n/a"
        elif isinstance(value, str):
            rendered = value
        elif isinstance(value, bool):
            rendered = "yes" if value else "no"
        else:
            rendered = fmt.format(value)
        self._entries.append({
            "rendered": rendered,
            "raw_value": self._jsonable(value),
            "source": source,
            "locator": locator,
        })
        return rendered

    def text(self, value: str, source: str, locator: str) -> str:
        """A non-numeric string lifted verbatim from an artifact."""
        return self.cite(value, source, locator)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, (str, bool, int, float, type(None))):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @property
    def rendered_tokens(self) -> set[str]:
        """Every string the ledger has issued, for the paper-scan test."""
        return {entry["rendered"] for entry in self._entries}

    def write(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "purpose": (
                "Audit trail for reports/research_paper.md. Every numeral in the "
                "paper was issued by Evidence.cite and appears here with the "
                "artifact it was read from. Nothing was typed from memory."
            ),
            "n_citations": len(self._entries),
            "n_distinct_sources": len({e["source"] for e in self._entries}),
            "sources": sorted({e["source"] for e in self._entries}),
            "citations": self._entries,
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
