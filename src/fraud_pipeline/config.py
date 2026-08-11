"""Loads configs/config.yaml into a single object every stage imports from.

Keeping this in one place is what makes the pipeline "configurable, not
hardcoded" — Person A's notebook had split ratios, model hyperparameters,
and thresholds scattered across cells. Here they live in one YAML file.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# Convenience absolute paths used across stages
RAW_DATA_PATH = REPO_ROOT / CONFIG["data"]["raw_path"]
PROCESSED_DIR = REPO_ROOT / CONFIG["data"]["processed_dir"]
MODEL_DIR = REPO_ROOT / CONFIG["artifacts"]["model_dir"]
