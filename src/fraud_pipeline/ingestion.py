from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

from .config import CONFIG, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def download_from_kaggle(dest_dir: Path = RAW_DATA_PATH.parent) -> Path:

    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset = CONFIG["data"]["kaggle_dataset"]

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest_dir)],
        check=True,
    )
    zip_path = dest_dir / "creditcardfraud.zip"
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest_dir)], check=True)

    logger.info("Downloaded and extracted %s to %s", dataset, dest_dir)
    return RAW_DATA_PATH


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"No raw data at {path}. Run download_from_kaggle() first, or "
            "place creditcard.csv there manually."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_from_kaggle()
    df = load_data()
    print(df.shape)
