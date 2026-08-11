# Stage-by-stage orchestration via plain Python, no external scheduler.
# `make pipeline` chains ingestion -> validation -> preprocessing ->
# training -> evaluation -> persist by calling src/fraud_pipeline/pipeline.py.

.PHONY: install pipeline test lint docker-build clean

install:
	pip install -r requirements.txt

pipeline:
	python -m src.fraud_pipeline.pipeline

test:
	pytest tests/ -v

lint:
	python -m pyflakes src/ tests/

docker-build:
	docker build -t fraud-detection-pipeline .

clean:
	rm -rf mlruns mlflow.db models/*.pkl models/eval __pycache__ .pytest_cache
