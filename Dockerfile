FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost/matplotlib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

# data/ and models/ are mounted as volumes at run time — see README
RUN mkdir -p data/raw data/processed models

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.fraud_pipeline.pipeline"]
