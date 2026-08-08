

.PHONY: install data validate train pipeline test lint docker-build clean

install:
	pip install -r requirements.txt

data:
	python -m src.fraud_pipeline.ingestion

pipeline:
	python -m src.fraud_pipeline.flow

test:
	pytest tests/ -v

lint:
	python -m pyflakes src/ tests/

docker-build:
	docker build -t fraud-detection-pipeline .

clean:
	rm -rf mlruns models/*.pkl models/eval __pycache__ .pytest_cache
