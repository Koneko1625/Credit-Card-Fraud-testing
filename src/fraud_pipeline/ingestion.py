from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw CSV into a DataFrame. Raises loudly if it's missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data at {path}. Copy creditcard.csv into that location "
            "before running the pipeline (see README.md -> Setup)."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_data()
    print(df.shape)
