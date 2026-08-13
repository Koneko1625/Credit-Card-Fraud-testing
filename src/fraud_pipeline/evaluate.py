from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display available in pipeline/CI runs
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CONFIG

logger = logging.getLogger(__name__)

THRESH_MIN = CONFIG["evaluation"]["threshold_min"]
THRESH_MAX = CONFIG["evaluation"]["threshold_max"]
THRESH_STEP = CONFIG["evaluation"]["threshold_step"]


def compute_core_metrics(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def threshold_sweep(y_test: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Same sweep as Part A: score every threshold from THRESH_MIN to
    THRESH_MAX in THRESH_STEP increments, keep precision/recall/F1/FP/FN."""
    thresholds = np.arange(THRESH_MIN, THRESH_MAX + 0.01, THRESH_STEP)
    rows = []
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1 Score": f1_score(y_test, preds, zero_division=0),
                "False Positives": fp,
                "False Negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(threshold_df: pd.DataFrame) -> pd.Series:
    return threshold_df.loc[threshold_df["F1 Score"].idxmax()]


def evaluate_and_log(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path,
    log_to_active_run: bool = True,
) -> dict:
    """Runs the full evaluation stage and logs metrics + artifacts to
    MLflow. Pass log_to_active_run=True when called right after train.py
    inside the same `with mlflow.start_run()` block; otherwise it opens
    its own run.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    core_metrics = compute_core_metrics(y_test, y_pred, y_prob)
    threshold_df = threshold_sweep(y_test, y_prob)
    best = best_threshold(threshold_df)

    threshold_csv = artifact_dir / "threshold_sweep.csv"
    threshold_df.to_csv(threshold_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
    ax.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
    ax.plot(threshold_df["Threshold"], threshold_df["F1 Score"], label="F1 Score")
    ax.axvline(best["Threshold"], color="grey", linestyle="--", label="Best F1 threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    plot_path = artifact_dir / "threshold_sweep.png"
    fig.savefig(plot_path)
    plt.close(fig)

    def _log():
        mlflow.log_metrics(core_metrics)
        mlflow.log_metric("best_threshold", float(best["Threshold"]))
        mlflow.log_metric("best_f1", float(best["F1 Score"]))
        mlflow.log_artifact(str(threshold_csv))
        mlflow.log_artifact(str(plot_path))

    if log_to_active_run and mlflow.active_run() is not None:
        _log()
    else:
        with mlflow.start_run(run_name="evaluation"):
            _log()

    logger.info("Evaluation complete: %s", core_metrics)
    return {"metrics": core_metrics, "best_threshold": best.to_dict()}
