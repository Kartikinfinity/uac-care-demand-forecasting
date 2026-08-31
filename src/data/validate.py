"""
validate.py -- Data-provenance hashing and the fail-fast validation gate.

Implements two PRE_BUILD_TECHNICAL_ADDENDUM Day-2 deliverables that were named
in Section 9 ("build the 769-slot period-position master series; apply
per-column imputation flags; COMPUTE THE DATA HASH; add the FAIL-FAST
VALIDATION GATE") and required again by Section 3:

  * "Model/forecast registry carries, per artifact: a content hash of the raw
     CSV and of the cleaned master series, a generation timestamp [...]"
  * "Fail-fast validation gate [...] reusing the above assertions as a runtime
     guard."

The hashes are content hashes over a canonical text serialisation, not hashes of
the parquet bytes: parquet embeds writer metadata, so hashing the file itself
would change between identical runs and make the provenance record useless for
the Day-12 deployment smoke test ("confirm the deployed app's provenance readout
matches the local run").

Nothing here is exploratory. Every assertion restates a rule that already exists
in the addendum, so a violation means the data no longer matches the frozen
specification and the pipeline must stop rather than produce a plausible-looking
artifact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    COL_DATE,
    DATE_FORMAT,
    EXPECTED_GAP_COUNT,
    EXPECTED_REAL_ROWS,
    EXPECTED_TEMPLATE_POSITIONS,
    EXPECTED_TOTAL_POSITIONS,
    FLOW_COLS,
    NUMERIC_COLS,
    OFF_TEMPLATE_FRIDAYS,
    PROVENANCE_PATH,
    RAW_CSV_PATH,
    REPORTING_WEEKDAYS,
    STOCK_COLS,
)

__all__ = [
    "hash_file",
    "hash_frame",
    "validate_master_series",
    "build_provenance",
    "write_provenance",
    "read_provenance",
]


def hash_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes. Used for the untouched source CSV."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_frame(df: pd.DataFrame) -> str:
    """
    SHA-256 over a canonical CSV serialisation of a DataFrame.

    Deterministic across runs and across parquet writer versions, which hashing
    the parquet file would not be.
    """
    payload = df.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_master_series(df: pd.DataFrame) -> None:
    """
    Fail-fast gate. Raises ValueError on the first violated invariant.

    Deliberately raises rather than asserts, so the check survives `python -O`.
    """

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("master-series validation failed: " + message)

    check(len(df) == EXPECTED_TOTAL_POSITIONS,
          "expected %d period-positions, got %d" % (EXPECTED_TOTAL_POSITIONS, len(df)))

    dates = pd.to_datetime(df["parsed_date"])
    check(dates.is_monotonic_increasing, "period-position index is not chronological")
    check(dates.is_unique, "duplicate dates in the period-position index")

    n_real = int((~df["is_imputed"]).sum())
    check(n_real == EXPECTED_REAL_ROWS,
          "expected %d real observations, got %d" % (EXPECTED_REAL_ROWS, n_real))
    check(int(df["is_imputed"].sum()) == EXPECTED_GAP_COUNT,
          "gap count is not %d" % EXPECTED_GAP_COUNT)

    # Index composition: canonical Sun-Thu template plus the two named Fridays.
    on_template = dates.dt.dayofweek.isin(REPORTING_WEEKDAYS)
    check(int(on_template.sum()) == EXPECTED_TEMPLATE_POSITIONS,
          "template positions != %d" % EXPECTED_TEMPLATE_POSITIONS)
    for friday in OFF_TEMPLATE_FRIDAYS:
        ts = pd.Timestamp(friday)
        check(int((dates == ts).sum()) == 1, "off-template Friday %s missing" % friday)
        check(not bool(df.loc[dates == ts, "is_imputed"].iloc[0]),
              "off-template Friday %s must be a real observation" % friday)

    # Invariant 2: flow columns are true-missing at gaps, never filled.
    for col in FLOW_COLS:
        check(df.loc[df["is_imputed"], col].isna().all(),
              "flow column %r was filled at a gap slot" % col)
        check(df.loc[~df["is_imputed"], col].notna().all(),
              "flow column %r is missing on a real observation" % col)

    # Stock columns are interpolated everywhere and carry a per-column flag.
    for col in STOCK_COLS:
        check(df[col].notna().all(), "stock column %r has an unfilled slot" % col)
        check("is_imputed_%s" % col in df.columns,
              "missing required per-column flag is_imputed_%s" % col)

    for col in NUMERIC_COLS:
        check((df[col].dropna() >= 0).all(), "negative values in %r" % col)


def build_provenance(raw_csv_path: Path, master: pd.DataFrame) -> dict:
    """The provenance record carried alongside every downstream artifact."""
    return {
        "raw_csv_filename": Path(raw_csv_path).name,
        "raw_csv_sha256": hash_file(raw_csv_path),
        "master_series_sha256": hash_frame(master),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_period_positions": int(len(master)),
        "n_real_observations": int((~master["is_imputed"]).sum()),
        "n_gap_slots": int(master["is_imputed"].sum()),
        "data_as_of": str(pd.to_datetime(master["parsed_date"]).max().date()),
        "series_starts": str(pd.to_datetime(master["parsed_date"]).min().date()),
    }


def write_provenance(record: dict, path: Path = PROVENANCE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_provenance(path: Path = PROVENANCE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    from src.config import MASTER_SERIES_PATH

    master = pd.read_parquet(MASTER_SERIES_PATH)
    validate_master_series(master)
    record = build_provenance(RAW_CSV_PATH, master)
    write_provenance(record)
    print(json.dumps(record, indent=2))
