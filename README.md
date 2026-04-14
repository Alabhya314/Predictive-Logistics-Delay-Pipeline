# Predictive Logistics Delay Pipeline

> A production-grade ML system that predicts shipping delays by fusing
> logistics data with real-time weather/traffic signals.
> Built to demonstrate the patterns that separate **Senior MLEs** from Notebook Warriors.

```
docker-compose up
```

That single command starts Airflow, MLflow, and the full pipeline.

---

## What This System Does

Shipping delays cost the logistics industry an estimated **$1.3 trillion annually**.
This pipeline ingests raw trip records, validates them against a **Data Contract**,
engineers temporal and external-signal features, trains an XGBoost regressor,
and gates the result behind quality thresholds — all fully orchestrated and observable.

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌───────┐    ┌──────────┐
│  Ingest │ ─► │ Validate │ ─► │ Transform │ ─► │ Train │ ─► │ Evaluate │
└─────────┘    └──────────┘    └───────────┘    └───────┘    └──────────┘
     │               │                                │              │
  DVC hash    GE contract             MLflow tracks params,      Promotion
  manifest    enforcement            metrics + data version      threshold
```

---

## Architecture

```
logistics-pipeline/
├── docker-compose.yml          ← 1-step startup (Airflow + MLflow + Postgres)
├── Makefile                    ← Local development shortcuts
├── requirements.txt
│
├── dags/
│   └── logistics_pipeline_dag.py  ← Airflow DAG with retry logic
│
├── src/
│   ├── ingestion/
│   │   └── ingest.py           ← API simulation, DVC-compatible manifest
│   ├── validation/
│   │   └── validate.py         ← Data Contract (Great Expectations)
│   ├── features/
│   │   └── engineer.py         ← Modular sklearn-compatible transformers
│   └── training/
│       ├── train.py            ← XGBoost + MLflow tracking
│       └── evaluate.py         ← Promotion gating
│
├── tests/
│   └── test_pipeline.py        ← Unit + integration tests
│
└── data/
    ├── raw/                    ← DVC-tracked raw parquet files
    ├── validated/              ← Post-contract parquet + validation report
    ├── processed/              ← Feature-engineered parquet
    └── models/                 ← Trained model + evaluation report
```

---

## Quick Start

### Option A — Full stack (recommended)

```bash
git clone <this-repo>
cd logistics-pipeline
docker-compose up
```

| Service  | URL                      | Credentials    |
|----------|--------------------------|----------------|
| Airflow  | http://localhost:8080    | admin / admin  |
| MLflow   | http://localhost:5000    | —              |

Enable the `logistics_delay_pipeline` DAG in the Airflow UI and trigger a run.

### Option B — Local (no Docker)

```bash
pip install -r requirements.txt

# Run stages individually
make ingest
make validate
make transform
make train
make evaluate

# Or run everything
make pipeline
```

### Running Tests

```bash
make test
# or
PYTHONPATH=src pytest tests/ -v
```

---

## The Senior MLE Checklist

| Criterion | Implementation |
|-----------|---------------|
| **No Notebooks** | All logic in `.py` modules under `src/`. Zero `.ipynb` files in the pipeline. |
| **Schema Enforcement** | `validate.py` rejects data failing any Data Contract expectation and writes a JSON alert. |
| **Reproducibility** | DVC manifest ties every model to its exact input data hash. MLflow stores data version hash alongside hyperparameters. `git checkout <sha> && dvc pull` recreates any historical run. |
| **Logging** | Every module uses Python's `logging` with structured output. Failure alerts written to `data/alerts/` as JSON. Airflow captures task stdout/stderr per run. |

---

## Technical Decisions

### Why XGBoost over Deep Learning?

**Decision**: XGBoost (gradient-boosted trees) for the primary model.

| Dimension | XGBoost | Deep Learning |
|-----------|---------|---------------|
| Inference latency | < 1 ms | 5–50 ms (CPU) |
| Interpretability | Native SHAP support | Requires post-hoc attribution |
| Data requirement | Effective at 5k–100k rows | Often needs 100k+ rows |
| Infrastructure | CPU-only, no GPU | GPU strongly preferred |
| Tabular performance | State-of-the-art on benchmarks | Rarely outperforms GBMs on tabular |

For structured logistics data with temporal patterns, XGBoost consistently matches or beats
neural approaches while being faster to iterate on and easier to explain to operations teams.
A `LightGBM` fallback is coded in `train.py` for environments where XGBoost is unavailable.

### Why Airflow over cron / scripts?

**Decision**: Apache Airflow for orchestration.

**The problem cron doesn't solve**: If the API pull times out at 03:00, a cron job silently
produces no output. Downstream jobs either crash or run on stale data — with no record of why.

Airflow provides:
- **Task dependency DAG**: `validate` won't run if `ingest` failed — no silent stale-data runs
- **Automatic retries with exponential backoff**: API timeouts are transient; three retries with
  5/10/20-minute delays handle most real-world intermittency
- **Execution history**: every run logged with duration, exit status, and captured stdout
- **XCom**: tasks pass file paths, not side effects — easier to test and debug

**Trade-off accepted**: Airflow is operationally heavier than a simple Python script.
For a team of one running weekly, a simple cron + alerting script would suffice.
For anything that needs audit trails, retry logic, or parallel tasks, Airflow pays for itself.

### Why Great Expectations over manual assertions?

**Decision**: Data Contract defined in `validate.py` using GE semantics (built-in fallback when GE is not installed).

Manual `assert df['col'].ge(0).all()` scattered across scripts has no memory — each run
re-invents the contract. GE formalises expectations as versioned, reusable artifacts:
- The contract is readable by non-engineers
- Failures produce structured JSON reports (not Python tracebacks)
- The contract evolves with the schema, not with the model

### Why DVC for data versioning?

**Decision**: DVC-compatible SHA-256 manifest instead of `data_v2_final_FINAL.csv`.

Git tracks code. DVC tracks data. A `manifest.json` entry for every ingested file means:
- Any model can be reproduced by checking out the commit and pulling the matching data
- Storage-efficient (data lives outside Git, only the hash is committed)
- Compatible with S3/GCS remotes for team workflows

### Cyclic encoding for time features

**Decision**: `sin`/`cos` encoding for `hour_of_day` and `day_of_week`.

Without cyclic encoding, a tree model sees midnight (hour=0) and 11pm (hour=23) as
maximally far apart. In reality they are one hour apart. Sin/cos encoding preserves
the circular distance, letting the model find rush-hour patterns correctly.

---

## Data Contract

The following expectations are enforced on every ingested dataset:

```
Schema
  ✓ All required columns present (trip_id, pickup_datetime, …)
  ✓ Numeric columns have correct dtypes

Completeness
  ✓ < 1% null values in any required column

Logical bounds
  ✓ trip_distance ∈ [0, 200]
  ✓ fare_amount ≥ 0
  ✓ delay_minutes ≥ 0
  ✓ traffic_index ∈ [0, 1]
  ✓ weather_code ∈ {1, …, 10}

Referential integrity
  ✓ trip_id values are unique
  ✓ pickup_datetime < dropoff_datetime
  ✓ Row count ≥ 100
```

A failed contract writes `data/validated/validation_report.json` and raises
`RuntimeError`, halting the pipeline before the model can be poisoned.

---

## MLflow Experiment Tracking

Every training run logs:

| Category | Logged values |
|----------|--------------|
| Parameters | All XGBoost hyperparameters |
| Data | SHA-256 hash of raw file (links model ↔ dataset) |
| Metrics | Train/test MAE, RMSE, R² |
| Artefacts | Serialised model, feature column list |

To compare runs:
```bash
# Start MLflow UI (if running locally)
mlflow ui --port 5000
```
Or open http://localhost:5000 when running via Docker.

---

## Fault Tolerance

| Failure mode | Response |
|--------------|----------|
| API timeout during ingestion | Airflow retries 3× with exponential backoff (5m, 10m, 20m) |
| Data contract violation | Pipeline halts, JSON alert written, downstream tasks skipped |
| Model quality below threshold | `evaluate` task raises `RuntimeError`, run marked failed |
| Container crash | `restart: unless-stopped` in docker-compose |
| Simultaneous DAG runs | `max_active_runs: 1` prevents data races |

---

## Extending This Project

- **Real data**: Replace `_simulate_api_response()` with a real HTTP client pointing at the NYC TLC API
- **Feature store**: Replace `LagFeatureBuilder` with reads from a Redis/Feast feature store
- **Model registry**: Add an `mlflow.register_model()` call in `evaluate.py` after successful promotion
- **Serving**: Wrap the saved model in FastAPI; add a `serve` task to the DAG
- **Monitoring**: Add a `monitor` task that compares live prediction distributions to training distributions (data drift)
