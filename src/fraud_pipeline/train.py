from __future__ import annotations

import logging
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import CONFIG

logger = logging.getLogger(__name__)

RANDOM_STATE = CONFIG["model"]["random_state"]
XGB_PARAMS = CONFIG["model"]["xgboost_tuned"]
REGISTERED_MODEL_NAME = CONFIG["mlflow"]["registered_model_name"]


@contextmanager
def _run_or_reuse_active(run_name: str):
    """Yields the currently active MLflow run if there is one; otherwise
    opens a new one (and closes it on exit). Prevents the "Run ... is
    already active" error when called from inside an outer
    `with mlflow.start_run():` block, while still working standalone."""
    if mlflow.active_run() is not None:
        yield
    else:
        with mlflow.start_run(run_name=run_name):
            yield


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, run_name: str = "logistic_regression_baseline"
) -> LogisticRegression:
    with _run_or_reuse_active(run_name):
        params = {"random_state": RANDOM_STATE, "max_iter": 1000}
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "logistic_regression")

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Logistic Regression trained and logged to MLflow")
        return model


def train_xgboost_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    register: bool = True,
    run_name: str = "xgboost_tuned",
) -> XGBClassifier:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    with _run_or_reuse_active(run_name):
        params = {**XGB_PARAMS, "random_state": RANDOM_STATE, "scale_pos_weight": scale_pos_weight}
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "xgboost")

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME if register else None,
        )
        logger.info("Tuned XGBoost trained and logged to MLflow (registered=%s)", register)
        return model
