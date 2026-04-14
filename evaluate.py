"""
training/evaluate.py
---------------------
Loads the last trained model and evaluates it against a held-out slice.
Writes a JSON report that Airflow can inspect to decide whether to
promote the model to a 'registry' slot.

Promotion criteria (configurable via env vars):
  EVAL_MAX_MAE   – maximum acceptable MAE  (default: 15 minutes)
  EVAL_MIN_R2    – minimum acceptable R²   (default: 0.50)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger("evaluation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

MAX_MAE = float(os.getenv("EVAL_MAX_MAE", "15.0"))
MIN_R2  = float(os.getenv("EVAL_MIN_R2",  "0.50"))
TARGET_COL = "delay_minutes"


def run_evaluation(
    processed_path: Path | str,
    model_dir: Path | str = "data/models",
    output_dir: Path | str = "data/models",
) -> dict:
    """
    Evaluates the most recent model and writes an evaluation report.

    Raises
    ------
    RuntimeError if model quality is below promotion thresholds.
    """
    import pickle

    processed_path = Path(processed_path)
    model_dir      = Path(model_dir)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results from training
    results_path = model_dir / "last_run_results.json"
    if results_path.exists():
        with open(results_path) as fh:
            train_results = json.load(fh)
        test_metrics = train_results.get("metrics", {}).get("test", {})
        feature_cols = train_results.get("feature_cols", [])
    else:
        test_metrics = {}
        feature_cols = []

    log.info("Loading processed data for evaluation …")
    df = pd.read_parquet(processed_path)

    if not feature_cols:
        feature_list_path = processed_path.parent / "feature_columns.json"
        if feature_list_path.exists():
            with open(feature_list_path) as fh:
                feature_cols = json.load(fh)
        else:
            feature_cols = [c for c in df.columns if c != TARGET_COL]

    from sklearn.model_selection import train_test_split
    _, X_test_df, _, y_test = train_test_split(
        df[feature_cols], df[TARGET_COL], test_size=0.2, random_state=42
    )

    # Try loading XGBoost model first, then pickle fallback
    model = None
    xgb_path = model_dir / "model.json"
    pkl_path  = model_dir / "model.pkl"

    if xgb_path.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(xgb_path))
            log.info("Loaded XGBoost model from %s", xgb_path)
        except Exception as e:
            log.warning("Could not load XGBoost model: %s", e)

    if model is None and pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)
        log.info("Loaded pickled model from %s", pkl_path)

    if model is None:
        raise RuntimeError(f"No model found in {model_dir}")

    y_pred = model.predict(X_test_df.values)
    fresh_metrics = {
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2":   float(r2_score(y_test, y_pred)),
    }

    log.info(
        "Evaluation metrics — MAE=%.3f | RMSE=%.3f | R²=%.3f",
        fresh_metrics["mae"], fresh_metrics["rmse"], fresh_metrics["r2"],
    )

    passed_mae = fresh_metrics["mae"] <= MAX_MAE
    passed_r2  = fresh_metrics["r2"]  >= MIN_R2

    promotion_decision = "PROMOTE" if (passed_mae and passed_r2) else "REJECT"
    log.info(
        "Promotion thresholds — MAE≤%.1f: %s | R²≥%.2f: %s → %s",
        MAX_MAE, "PASS" if passed_mae else "FAIL",
        MIN_R2,  "PASS" if passed_r2  else "FAIL",
        promotion_decision,
    )

    report = {
        "promotion_decision": promotion_decision,
        "thresholds": {"max_mae": MAX_MAE, "min_r2": MIN_R2},
        "metrics": fresh_metrics,
        "passed_mae": passed_mae,
        "passed_r2":  passed_r2,
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Evaluation report → %s", report_path)

    if promotion_decision == "REJECT":
        raise RuntimeError(
            f"Model quality below thresholds: MAE={fresh_metrics['mae']:.2f} "
            f"(max={MAX_MAE}), R²={fresh_metrics['r2']:.3f} (min={MIN_R2})"
        )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("processed_data", help="Path to processed parquet file")
    parser.add_argument("--model-dir", default="data/models")
    args = parser.parse_args()

    run_evaluation(args.processed_data, args.model_dir)
