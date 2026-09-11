"""
load.py -- Read the raw CSV, and nothing else (Day 1).

Deliberately thin. This module's only job is to get the delivered file into a
DataFrame with no interpretation applied: no type coercion, no reindexing, no
gap handling. Everything that makes a judgement about the data belongs in
`clean.py`, so that there is exactly one place to look when asking "what was
done to this data before I saw it".

The raw file is treated as read-only throughout the project and is hashed on
every run by `validate.py`.
"""
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    RAW_CSV_PATH,
    EXPECTED_RAW_ROWS,
    EXPECTED_REAL_ROWS,
    EXPECTED_BLANK_ROWS,
    COL_DATE,
    NUMERIC_COLS,
)

def load_raw_data(filepath: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and fail loudly if it is not the file this pipeline was
    built against (addendum Section 3: "fail fast at the data-validation stage").

    The previous version of this check evaluated a condition and then did
    nothing with it, so a substituted or truncated CSV would have flowed
    silently into cleaning.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(
            "raw CSV not found at %s -- see README for the expected location" % filepath
        )
    df = pd.read_csv(filepath)

    missing = [c for c in [COL_DATE] + NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError("raw CSV is missing expected column(s): %s" % missing)

    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(
            "raw CSV has %d rows, expected %d (%d data rows + %d blank trailing rows). "
            "If the source file was legitimately updated, update EXPECTED_RAW_ROWS and "
            "the counts in config.py, and re-run the full pipeline."
            % (len(df), EXPECTED_RAW_ROWS, EXPECTED_REAL_ROWS, EXPECTED_BLANK_ROWS)
        )
    return df
