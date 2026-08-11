"""Stage orchestration — plain Python chain, no external scheduler.

Runs ingestion -> validation -> preprocessing -> training -> evaluation
-> persist artifacts, in that order, inside one MLflow run.

This replaces an earlier Prefect-based version: once ingestion moved
to reading a local file instead of calling an external API, Prefect's
retry/scheduling machinery wasn't earning its dependency weight for
this project's scope. If you outgrow this later — scheduled runs, a
run graph UI, automatic retries on a flaky step — reintroducing
Prefect here is a small change: wrap each stage call below in a
`@task` and this function in a `@flow`.

Run with:
    python -m src.fraud_pipeline.pipeline
or:
    make pipeline
"""
from __future__ import annotations

import logging

import mlflow

from .config import CONFIG, MODEL_DIR, RAW_DATA_PATH
from .evaluate import evaluate_and_log
from .inference import save_inference_artifacts
from .ingestion import load_data
from .preprocessing import preprocess
from .train import train_xgboost_tuned
from .validation import validate_data

logger = logging.getLogger(__name__)


def run_pipeline() -> dict:
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    logger.info("Stage 1/5: ingestion")
    raw_df = load_data(RAW_DATA_PATH)

    logger.info("Stage 2/5: validation")
    validated_df = validate_data(raw_df)

    logger.info("Stage 3/5: preprocessing")
    X_train, X_test, y_train, y_test, scaler = preprocess(validated_df)

    with mlflow.start_run(run_name="pipeline_run"):
        logger.info("Stage 4/5: training")
        model = train_xgboost_tuned(X_train, y_train)

        logger.info("Stage 5/5: evaluation")
        results = evaluate_and_log(model, X_test, y_test, artifact_dir=MODEL_DIR / "eval")

        save_inference_artifacts(
            model, scaler, results["best_threshold"]["Threshold"], MODEL_DIR
        )

    logger.info("Pipeline complete: %s", results["metrics"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
