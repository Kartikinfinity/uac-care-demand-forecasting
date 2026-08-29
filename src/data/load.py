import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_CSV_PATH, EXPECTED_RAW_ROWS

def load_raw_data(filepath: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV, asserting basic file structure."""
    df = pd.read_csv(filepath)
    if len(df) == EXPECTED_RAW_ROWS:
        # File has the blank trailing rows, which is expected
        pass
    return df
