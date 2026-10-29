# Explainable Anomaly Detection & Model Monitoring (MVP)

A reproducible **synthetic-data** anomaly detection demo using Isolation Forest,
FastAPI, MLflow, SHAP, and a Kolmogorov–Smirnov drift test.

> This is **not** a real fraud detection benchmark. Synthetic data and unsupervised
> anomaly flags cannot establish real-world fraud detection performance.

## Setup

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m app.model
python -m app.explain
uvicorn app.main:app --reload --port 8001
```

Open http://127.0.0.1:8001/docs

Example POST `/predict`:

```json
{"amount":2500,"hour":2,"account_age_days":5,"transactions_last_24h":40}
```

POST `/drift` accepts at least 30 objects of the same schema. Use a batch of new transactions.
The drift endpoint compares feature distributions with a KS test and Bonferroni correction.
It does **not** automatically run on a schedule.

Run tests: `pytest -q`. Inspect MLflow runs with `mlflow ui`.
Docker: `docker build -t anomaly-monitor .` then `docker run -p 8001:8001 anomaly-monitor`.

## Architecture

Synthetic transactions → feature preprocessing → Isolation Forest → model artifact →
FastAPI prediction; optional SHAP score explanation; reference-vs-current KS drift checks;
MLflow training experiment logging.

## Limitations and improvements

No real fraud labels, so precision/recall are not reported. Add a public labeled
dataset, held-out evaluation, temporal validation, background monitoring jobs,
alerting, and API authentication before claiming production readiness.

## Responsible resume description

"Developed an unsupervised anomaly detection prototype using Isolation Forest,
with FastAPI inference, SHAP score explanations, MLflow tracking, and statistical
feature-drift checks." Use only after running the components yourself.
