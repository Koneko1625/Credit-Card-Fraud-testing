
import numpy as np
import pandas as pd

from src.fraud_pipeline.evaluate import best_threshold, compute_core_metrics, threshold_sweep


def test_compute_core_metrics_perfect_predictions():
    y_test = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.95])
    metrics = compute_core_metrics(y_test, y_pred, y_prob)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_threshold_sweep_returns_expected_columns():
    y_test = pd.Series([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    df = threshold_sweep(y_test, y_prob)
    expected_cols = {"Threshold", "Precision", "Recall", "F1 Score", "False Positives", "False Negatives"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0


def test_best_threshold_picks_max_f1():
    df = pd.DataFrame(
        {
            "Threshold": [0.1, 0.5, 0.9],
            "Precision": [0.5, 0.8, 0.9],
            "Recall": [0.9, 0.7, 0.3],
            "F1 Score": [0.6, 0.75, 0.45],
            "False Positives": [10, 5, 1],
            "False Negatives": [1, 3, 8],
        }
    )
    best = best_threshold(df)
    assert best["Threshold"] == 0.5
    assert best["F1 Score"] == 0.75
