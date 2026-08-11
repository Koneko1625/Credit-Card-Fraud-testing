"""Stage orchestration with Prefect.

Turns the notebook's top-to-bottom cell execution into a real pipeline:
ingestion -> validation -> preprocessing -> training -> evaluation ->
save artifacts, each as its own @task so Prefect gives us retries,
logging, and a run graph for free. Run directly with:

    python -m fraud_pipeline.flow

or deploy it (`prefect deploy`) to run on a schedule / trigger.

If Prefect isn't available in your environment, `Makefile` in the repo
root chains the same stages as plain `python -m` calls — functionally
equivalent, just without retries/scheduling/observability.
"""
from __future__ import annotations

import logging

import mlflow
from prefect import flow, task

from .config import CONFIG, MODEL_DIR, RAW_DATA_PATH
from .evaluate import evaluate_and_log
from .ingestion import load_data
from .inference import save_inference_artifacts
from .preprocessing import preprocess
from .train import train_xgboost_tuned
from .validation import validate_data

logger = logging.getLogger(__name__)


@task(retries=2, retry_delay_seconds=10)
def ingest_task():
    return load_data(RAW_DATA_PATH)


@task
def validate_task(df):
    return validate_data(df)


@task
def preprocess_task(df):
    return preprocess(df)


@task
def train_task(X_train, y_train):
    return train_xgboost_tuned(X_train, y_train)


@task
def evaluate_task(model, X_test, y_test):
    return evaluate_and_log(model, X_test, y_test, artifact_dir=MODEL_DIR / "eval")


@task
def persist_task(model, scaler, best_threshold):
    save_inference_artifacts(model, scaler, best_threshold, MODEL_DIR)


@flow(name="fraud-detection-pipeline")
def fraud_detection_pipeline():
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    raw_df = ingest_task()
    validated_df = validate_task(raw_df)
    X_train, X_test, y_train, y_test, scaler = preprocess_task(validated_df)

    with mlflow.start_run(run_name="pipeline_run"):
        model = train_task(X_train, y_train)
        results = evaluate_task(model, X_test, y_test)
        persist_task(model, scaler, results["best_threshold"]["Threshold"])

    logger.info("Pipeline complete: %s", results["metrics"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fraud_detection_pipeline()
