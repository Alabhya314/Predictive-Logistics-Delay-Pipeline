"""
features/engineer.py
---------------------
Modular feature engineering pipeline.
Every transformer is a scikit-learn compatible class with fit/transform.
This makes the whole pipeline:
  1. Serialisable (joblib/pickle)
  2. Composable (sklearn Pipeline)
  3. Testable (each class in isolation)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("features")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# Individual transformer classes
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts calendar and cyclic time features from datetime columns.

    Cyclic encoding (sin/cos) for hour-of-day and day-of-week ensures the
    model sees midnight and 11pm as 'close together' rather than at opposite
    ends of a linear scale.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        pickup = pd.to_datetime(df["pickup_datetime"])

        df["hour_of_day"]   = pickup.dt.hour
        df["day_of_week"]   = pickup.dt.dayofweek
        df["month"]         = pickup.dt.month
        df["is_weekend"]    = (pickup.dt.dayofweek >= 5).astype(int)
        df["is_rush_hour"]  = (
            pickup.dt.hour.between(7, 9) | pickup.dt.hour.between(16, 19)
        ).astype(int)
        df["trip_duration_min"] = (
            pd.to_datetime(df["dropoff_datetime"]) - pickup
        ).dt.total_seconds() / 60

        # Cyclic encoding — prevents artificial discontinuity at day/week boundaries
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

        log.debug("TemporalFeatureExtractor: added 10 features")
        return df


class LagFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Creates rolling-window lag statistics grouped by pickup location.

    In production these would be computed against a feature store; here
    we approximate using the same batch (no data leakage because we sort
    by pickup_datetime before computing).
    """

    def __init__(self, lag_windows: list[int] | None = None) -> None:
        self.lag_windows = lag_windows or [3, 6, 12]

    def fit(self, X: pd.DataFrame, y=None) -> "LagFeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values("pickup_datetime").copy()

        for window in self.lag_windows:
            col = f"delay_rolling_mean_{window}h"
            df[col] = (
                df.groupby("pickup_location_id")["delay_minutes"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            df[col] = df[col].fillna(df["delay_minutes"].mean())

        log.debug("LagFeatureBuilder: added %d lag features", len(self.lag_windows))
        return df


class ExternalSignalEncoder(BaseEstimator, TransformerMixin):
    """
    Enriches the dataset with external signal interaction terms.

    Business insight: weather × traffic interaction often explains delay
    variance better than either feature alone.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ExternalSignalEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["weather_traffic_interaction"] = df["weather_code"] * df["traffic_index"]
        df["distance_per_minute"] = df["trip_distance"] / (
            df["trip_duration_min"].clip(lower=1)
        )
        df["is_bad_weather"]  = (df["weather_code"] >= 7).astype(int)
        df["is_high_traffic"] = (df["traffic_index"] >= 0.7).astype(int)
        log.debug("ExternalSignalEncoder: added 4 interaction features")
        return df


class NumericScaler(BaseEstimator, TransformerMixin):
    """
    Fits StandardScaler on a fixed set of numeric columns.
    Skips the target column and datetime/id columns.
    """

    SCALE_COLS = [
        "trip_distance", "fare_amount", "traffic_index",
        "trip_duration_min", "distance_per_minute",
    ]

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "NumericScaler":
        self._fitted_cols = [c for c in self.SCALE_COLS if c in X.columns]
        self._scaler.fit(X[self._fitted_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[self._fitted_cols] = self._scaler.transform(df[self._fitted_cols])
        log.debug("NumericScaler: scaled %d columns", len(self._fitted_cols))
        return df


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops columns not needed downstream.
    Keeping this explicit makes the feature set self-documenting.
    """

    DROP_COLS = [
        "trip_id", "pickup_datetime", "dropoff_datetime",
        "pickup_location_id", "dropoff_location_id",
        "hour_of_day", "day_of_week",          # replaced by cyclic encodings
    ]

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.DROP_COLS if c in X.columns]
        return X.drop(columns=drop)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> list[tuple[str, Any]]:
    """
    Returns an ordered list of (name, transformer) tuples.
    Wrap in sklearn.pipeline.Pipeline at the call site if needed.
    """
    return [
        ("temporal",  TemporalFeatureExtractor()),
        ("lag",       LagFeatureBuilder(lag_windows=[3, 6, 12])),
        ("external",  ExternalSignalEncoder()),
        ("scaler",    NumericScaler()),
        ("dropper",   ColumnDropper()),
    ]


def run_feature_engineering(
    input_path: Path | str,
    output_dir: Path | str = "data/processed",
) -> tuple[Path, list[str]]:
    """
    Executes the full feature engineering pipeline.

    Returns
    -------
    (output_path, feature_names)
    """
    import pickle

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading validated data from %s …", input_path)
    df = pd.read_parquet(input_path)
    log.info("Input shape: %s", df.shape)

    steps = build_feature_pipeline()
    for name, transformer in steps:
        log.info("Applying transformer: %s", name)
        if name == "scaler":
            transformer.fit(df)
        df = transformer.transform(df)

    TARGET = "delay_minutes"
    feature_cols = [c for c in df.columns if c != TARGET]

    log.info(
        "Feature engineering complete. Output shape: %s | Features: %d",
        df.shape,
        len(feature_cols),
    )

    # Persist processed dataset
    output_path = output_dir / input_path.name
    df.to_parquet(output_path, index=False, engine="pyarrow")

    # Persist fitted transformers (for inference-time reproduction)
    scaler_path = output_dir / "scaler.pkl"
    scaler_transformer = dict(steps)["scaler"]
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler_transformer, fh)

    # Persist feature list
    feature_list_path = output_dir / "feature_columns.json"
    import json
    with open(feature_list_path, "w") as fh:
        json.dump(feature_cols, fh, indent=2)

    log.info("Processed data → %s", output_path)
    log.info("Feature list   → %s", feature_list_path)
    return output_path, feature_cols


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("input", help="Path to validated parquet file")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    run_feature_engineering(args.input, args.output_dir)
