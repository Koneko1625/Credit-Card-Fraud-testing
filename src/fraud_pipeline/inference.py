"""Batch inference interface — this is the handoff to Person C.

C needs to feed in simulated batches (for drift testing) without caring
about how the model was trained. This module is the one contract they
should depend on:

    from fraud_pipeline.inference import load_inference_artifacts, predict_batch

    model, scaler, threshold = load_inference_artifacts()
    scored_df = predict_batch(df, model, scaler, threshold)

`predict_batch` takes a raw, unscored DataFrame shaped like the
original data (same columns, minus "Class" if present) and returns it
with two extra columns: fraud_probability and fraud_prediction. It
re-validates the batch against the same pandera schema training used,
so a malformed simulated batch fails loudly instead of silently
producing garbage predictions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR
from .validation import SCHEMA

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


def save_inference_artifacts(
    model: Any, scaler: StandardScaler, threshold: float, model_dir: Path = MODEL_DIR
) -> None:
    """Called at the end of the training pipeline so the exact scaler
    and chosen threshold ship alongside the model — C should never have
    to re-derive these."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    with open(model_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)
    logger.info("Saved model, scaler, and threshold to %s", model_dir)


def load_inference_artifacts(model_dir: Path = MODEL_DIR) -> Tuple[Any, StandardScaler, float]:
    model = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    with open(model_dir / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, threshold


def predict_batch(
    df: pd.DataFrame,
    model: Any,
    scaler: StandardScaler,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Scores a batch of transactions. `df` must contain the same feature
    columns used in training (Time, V1..V28, Amount) — "Class" is optional
    and ignored if present (useful when C is scoring labeled batches to
    measure drift against ground truth).
    """
    has_class = "Class" in df.columns
    validation_input = df if has_class else df.assign(Class=0)  # schema requires Class; dummy it
    SCHEMA.validate(validation_input[["Time", "Amount", "Class"]])

    features = df.drop(columns=["Class"]) if has_class else df.copy()
    features["Amount"] = scaler.transform(features[["Amount"]])

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    scored = df.copy()
    scored["fraud_probability"] = probabilities
    scored["fraud_prediction"] = predictions
    return scored
