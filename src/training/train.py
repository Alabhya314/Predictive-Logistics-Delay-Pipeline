"""
training/train.py
-----------------
Trains an XGBoost regression model to predict logistics delays.
Every run is tracked in MLflow: parameters, metrics, artefacts,
and the data version hash (from the manifest) for reproducibility.

Why XGBoost over Deep Learning?
  • Better interpretability for tabular logistics data (SHAP values)
  • Lower inference latency (<1 ms per prediction vs GPU-required DL)
  • No GPU infrastructure needed in production
  • Handles missing values natively
  • Comparable accuracy on structured tabular data (many benchmarks)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

log = logging.getLogger("training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "logistics-delay-prediction"
TARGET_COL = "delay_minutes"


def _load_data_version_hash(data_dir: Path) -> str:
    """Reads the SHA-256 hash of the most recently ingested file from the manifest."""
    manifest_path = data_dir / "raw" / "manifest.json"
    if not manifest_path.exists():
        log.warning("No manifest found at %s — data version unknown", manifest_path)
        return "unknown"
    with open(manifest_path) as fh:
        manifest: list[dict] = json.load(fh)
    if not manifest:
        return "unknown"
    return manifest[-1].get("sha256", "unknown")[:16]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def run_training(
    processed_path: Path | str,
    data_root: Path | str = "data",
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Trains an XGBoost model and logs the run to MLflow.

    Returns a dict with keys: run_id, metrics, model_path.
    """
    try:
        import mlflow
        import mlflow.xgboost
        MLFLOW_AVAILABLE = True
    except ImportError:
        log.warning("MLflow not installed — metrics will be logged to stdout only")
        MLFLOW_AVAILABLE = False

    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        log.warning("XGBoost not installed — falling back to LightGBM")
        XGB_AVAILABLE = False

    processed_path = Path(processed_path)
    data_root = Path(data_root)

    log.info("Loading processed features from %s …", processed_path)
    df = pd.read_parquet(processed_path)

    feature_list_path = processed_path.parent / "feature_columns.json"
    if feature_list_path.exists():
        with open(feature_list_path) as fh:
            feature_cols = json.load(fh)
    else:
        feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(
        "Train: %d rows | Test: %d rows | Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    params = model_params or {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    data_version = _load_data_version_hash(data_root)
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow tracking ────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception:
            pass

    def _train_and_evaluate():
        if XGB_AVAILABLE:
            import xgboost as xgb
            model = xgb.XGBRegressor(**params)
        else:
            try:
                import lightgbm as lgb
                lgbm_params = {k: v for k, v in params.items()
                               if k not in ("colsample_bytree", "reg_alpha", "reg_lambda")}
                lgbm_params["feature_fraction"] = params.get("colsample_bytree", 0.8)
                lgbm_params["lambda_l1"] = params.get("reg_alpha", 0.1)
                lgbm_params["lambda_l2"] = params.get("reg_lambda", 1.0)
                model = lgb.LGBMRegressor(**lgbm_params)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                sklearn_params = {k: v for k, v in params.items()
                                  if k in ("n_estimators", "max_depth", "learning_rate",
                                           "subsample", "random_state")}
                model = GradientBoostingRegressor(**sklearn_params)

        log.info("Training model: %s", type(model).__name__)
        model.fit(X_train, y_train)

        train_metrics = _compute_metrics(y_train, model.predict(X_train))
        test_metrics  = _compute_metrics(y_test,  model.predict(X_test))

        log.info(
            "Train metrics: MAE=%.3f | RMSE=%.3f | R²=%.3f",
            train_metrics["mae"], train_metrics["rmse"], train_metrics["r2"],
        )
        log.info(
            "Test  metrics: MAE=%.3f | RMSE=%.3f | R²=%.3f",
            test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )
        return model, train_metrics, test_metrics

    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("data_version_hash", data_version)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                for k, v in params.items():
                    mlflow.log_param(k, v)

                model, train_metrics, test_metrics = _train_and_evaluate()

                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                model_path = output_dir / "model.json"
                if XGB_AVAILABLE:
                    model.save_model(str(model_path))
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    import pickle
                    with open(str(model_path).replace(".json", ".pkl"), "wb") as fh:
                        pickle.dump(model, fh)

                mlflow.log_artifact(str(feature_list_path), artifact_path="features")

                run_id = run.info.run_id
                log.info("MLflow run_id: %s", run_id)
        except Exception as mlflow_exc:
            log.warning(
                "MLflow tracking unavailable (%s) — falling back to local-only mode.",
                mlflow_exc,
            )
            MLFLOW_AVAILABLE = False  # fall through to the block below

    if not MLFLOW_AVAILABLE:
        model, train_metrics, test_metrics = _train_and_evaluate()
        import pickle
        model_path = output_dir / "model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        run_id = "no-mlflow"

    results = {
        "run_id": run_id,
        "data_version_hash": data_version,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "model_path": str(model_path),
        "feature_cols": feature_cols,
    }

    results_path = output_dir / "last_run_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Training results written → %s", results_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("processed_data", help="Path to processed parquet file")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    run_training(args.processed_data, args.data_root)
