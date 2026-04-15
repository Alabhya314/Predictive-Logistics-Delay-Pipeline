"""
logistics_pipeline_dag.py
--------------------------
Apache Airflow DAG: Predictive Logistics Delay Pipeline

Chain: Ingest → Validate → Transform → Train → Evaluate

Key production features demonstrated here:
  • retries=3 / retry_delay=5min  → Fault tolerance for transient API failures
  • on_failure_callback            → Structured alert logging
  • XCom passing of file paths     → Decoupled task communication
  • dagrun_timeout                 → SLA enforcement
  • templated data paths           → Each run's data is isolated by execution date
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger("logistics_dag")

# ---------------------------------------------------------------------------
# Default task arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,         # set to True and configure smtp in prod
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,  # 5m, 10m, 20m backoffs
    "max_retry_delay": timedelta(hours=1),
    "execution_timeout": timedelta(minutes=30),
}


# ---------------------------------------------------------------------------
# Alert callbacks
# ---------------------------------------------------------------------------
def _on_failure_alert(context: dict) -> None:
    """
    Called by Airflow on any task failure.
    In production: push to Slack/PagerDuty. Here we write a structured log.
    """
    task_id   = context["task_instance"].task_id
    dag_id    = context["dag"].dag_id
    exec_date = context["execution_date"]
    exception = context.get("exception", "Unknown error")

    alert = {
        "alert_type": "PIPELINE_FAILURE",
        "dag_id": dag_id,
        "task_id": task_id,
        "execution_date": str(exec_date),
        "exception": str(exception),
    }
    log.error("PIPELINE ALERT: %s", json.dumps(alert, indent=2))

    # Write to disk so monitoring systems can pick it up
    alert_dir = Path("/opt/airflow/data/alerts")
    alert_dir.mkdir(parents=True, exist_ok=True)
    alert_path = alert_dir / f"alert_{task_id}_{exec_date.strftime('%Y%m%dT%H%M%S')}.json"
    with open(alert_path, "w") as fh:
        json.dump(alert, fh, indent=2)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def task_ingest(**context) -> str:
    """
    Pull raw logistics data.
    Returns the path to the written parquet file via XCom.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from ingestion.ingest import run_ingestion

    exec_date = context["execution_date"]
    filename  = f"logistics_{exec_date.strftime('%Y%m%dT%H%M%SZ')}.parquet"

    output_path = run_ingestion(
        n_rows=10_000,
        corruption_rate=0.0,
        output_filename=filename,
    )
    log.info("Ingestion complete: %s", output_path)
    return str(output_path)


def task_validate(**context) -> str:
    """
    Run data contract validation.
    Halts the DAG if any expectation fails — retry logic will kick in.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from validation.validate import run_validation

    raw_path = context["ti"].xcom_pull(task_ids="ingest")
    if not raw_path:
        raise ValueError("No raw data path from ingest task")

    validated_path = run_validation(raw_path, output_dir="/opt/airflow/data/validated")
    log.info("Validation complete: %s", validated_path)
    return str(validated_path)


def task_transform(**context) -> str:
    """
    Execute the feature engineering pipeline.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from features.engineer import run_feature_engineering

    validated_path = context["ti"].xcom_pull(task_ids="validate")
    if not validated_path:
        raise ValueError("No validated data path from validate task")

    processed_path, feature_cols = run_feature_engineering(
        validated_path,
        output_dir="/opt/airflow/data/processed",
    )
    log.info("Transformation complete: %s (%d features)", processed_path, len(feature_cols))
    return str(processed_path)


def task_train(**context) -> str:
    """
    Train the XGBoost model and log to MLflow.
    Returns the MLflow run_id via XCom.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from training.train import run_training

    processed_path = context["ti"].xcom_pull(task_ids="transform")
    if not processed_path:
        raise ValueError("No processed data path from transform task")

    results = run_training(
        processed_path,
        data_root="/opt/airflow/data",
    )
    run_id = results["run_id"]
    log.info(
        "Training complete — run_id=%s | test_MAE=%.3f",
        run_id,
        results["metrics"]["test"]["mae"],
    )
    return run_id


def task_evaluate(**context) -> None:
    """
    Evaluate the trained model against promotion thresholds.
    Raises if quality is insufficient.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from training.evaluate import run_evaluation

    processed_path = context["ti"].xcom_pull(task_ids="transform")
    if not processed_path:
        raise ValueError("No processed data path from transform task")

    report = run_evaluation(
        processed_path,
        model_dir="/opt/airflow/data/models",
        output_dir="/opt/airflow/data/models",
    )
    log.info("Evaluation result: %s", report["promotion_decision"])


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="logistics_delay_pipeline",
    description="End-to-end logistics delay prediction pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 3 * * *",     # 03:00 UTC daily
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=["mlops", "logistics", "production"],
    on_failure_callback=_on_failure_alert,
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
        on_failure_callback=_on_failure_alert,
        doc_md="""
        ## Ingest
        Simulates an API pull of NYC TLC trip records merged with weather/traffic signals.
        Writes a timestamped parquet file to `data/raw/` and updates the DVC manifest.
        **Retries**: 3× with exponential backoff (handles transient API timeouts).
        """,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=task_validate,
        on_failure_callback=_on_failure_alert,
        doc_md="""
        ## Validate
        Enforces the Data Contract via Great Expectations (or built-in validator).
        Halts the pipeline and writes an alert if any expectation fails.
        **Fail-safe**: A corrupt upstream data dump will not poison downstream models.
        """,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_transform,
        on_failure_callback=_on_failure_alert,
        doc_md="""
        ## Transform
        Runs the modular feature engineering pipeline:
        temporal extraction → lag features → external signal encoding → scaling.
        """,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        on_failure_callback=_on_failure_alert,
        doc_md="""
        ## Train
        Fits an XGBoost regressor. Every run is logged to MLflow with:
        - Hyperparameters
        - Train/test MAE, RMSE, R²
        - Data version hash (links model to exact dataset)
        """,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
        on_failure_callback=_on_failure_alert,
        doc_md="""
        ## Evaluate
        Checks the trained model against promotion thresholds (configurable via env vars).
        PROMOTE → model is ready for serving. REJECT → pipeline fails with clear error message.
        """,
    )

    # Define the DAG topology
    ingest >> validate >> transform >> train >> evaluate
