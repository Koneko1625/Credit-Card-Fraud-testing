from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import CONFIG

logger = logging.getLogger(__name__)

TRAIN_FRACTION = CONFIG["split"]["train_fraction"]


def time_ordered_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sorts by Time and splits the first `train_fraction` rows into train."""
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    X = df_sorted.drop("Class", axis=1)
    y = df_sorted["Class"]

    split_index = int(len(df_sorted) * train_fraction)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    logger.info("Train: %s rows, Test: %s rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def scale_amount(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fits StandardScaler on train's Amount column only, applies to both.

    Returns the fitted scaler too — it must be saved alongside the model
    so inference.py can apply the identical transform to new batches.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    return X_train, X_test, scaler


def preprocess(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Full preprocessing stage: split then scale. This is what the
    orchestration flow calls."""
    X_train, X_test, y_train, y_test = time_ordered_split(df, train_fraction)
    X_train, X_test, scaler = scale_amount(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
