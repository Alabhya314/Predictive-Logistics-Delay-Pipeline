"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for each pipeline stage.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------
class TestIngestion:
    def test_synthetic_data_shape(self):
        from ingestion.ingest import _simulate_api_response

        df = _simulate_api_response(n_rows=500, seed=0)
        assert len(df) == 500, "Wrong number of rows"
        assert "trip_id" in df.columns
        assert "delay_minutes" in df.columns

    def test_required_columns_present(self):
        from ingestion.ingest import _simulate_api_response, RAW_SCHEMA

        df = _simulate_api_response(n_rows=100)
        for col in RAW_SCHEMA:
            assert col in df.columns, f"Missing column: {col}"

    def test_distance_non_negative_by_default(self):
        from ingestion.ingest import _simulate_api_response

        df = _simulate_api_response(n_rows=1000, seed=42)
        assert (df["trip_distance"] >= 0).all(), "Found negative distances"

    def test_corruption_injects_bad_rows(self):
        from ingestion.ingest import _simulate_api_response

        df = _simulate_api_response(n_rows=1000, corruption_rate=0.1, seed=42)
        # Some rows should have negative values
        n_bad = (df["trip_distance"] < 0).sum() + (df["fare_amount"] < 0).sum()
        assert n_bad > 0, "Corruption did not inject bad rows"

    def test_run_ingestion_writes_parquet(self, tmp_path):
        from ingestion.ingest import run_ingestion

        out = run_ingestion(
            n_rows=200,
            output_filename="test.parquet",
        )
        # Override DATA_DIR for test
        import ingestion.ingest as ing_mod
        original = ing_mod.DATA_DIR
        ing_mod.DATA_DIR = tmp_path
        ing_mod.MANIFEST_PATH = tmp_path / "manifest.json"
        try:
            out = run_ingestion(n_rows=200, output_filename="test.parquet")
            assert out.exists(), "Output parquet not created"
            df = pd.read_parquet(out)
            assert len(df) == 200
        finally:
            ing_mod.DATA_DIR = original


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:
    def _make_clean_df(self, n: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "trip_id": [f"T-{i}" for i in range(n)],
            "pickup_datetime":     pd.date_range("2024-01-01", periods=n, freq="10min"),
            "dropoff_datetime":    pd.date_range("2024-01-01 00:30:00", periods=n, freq="10min"),
            "pickup_location_id":  rng.integers(1, 265, n),
            "dropoff_location_id": rng.integers(1, 265, n),
            "trip_distance":       rng.exponential(3, n).round(2),
            "fare_amount":         rng.gamma(3, 5, n).round(2),
            "delay_minutes":       np.abs(rng.normal(8, 5, n)).round(1),
            "weather_code":        rng.integers(1, 10, n),
            "traffic_index":       rng.uniform(0, 1, n).round(3),
        })

    def test_clean_data_passes(self, tmp_path):
        from validation.validate import _validate_dataframe

        df = self._make_clean_df()
        result = _validate_dataframe(df)
        assert result.success, f"Failures: {result.failures}"

    def test_negative_distance_fails(self):
        from validation.validate import _validate_dataframe

        df = self._make_clean_df()
        df.loc[0, "trip_distance"] = -5.0
        result = _validate_dataframe(df)
        assert not result.success
        checks = [f["check"] for f in result.failures]
        assert "trip_distance_non_negative" in checks

    def test_negative_fare_fails(self):
        from validation.validate import _validate_dataframe

        df = self._make_clean_df()
        df.loc[0:2, "fare_amount"] = -10.0
        result = _validate_dataframe(df)
        assert not result.success

    def test_duplicate_trip_id_fails(self):
        from validation.validate import _validate_dataframe

        df = self._make_clean_df()
        df.loc[1, "trip_id"] = df.loc[0, "trip_id"]
        result = _validate_dataframe(df)
        assert not result.success
        checks = [f["check"] for f in result.failures]
        assert "trip_id_unique" in checks

    def test_bad_traffic_index_fails(self):
        from validation.validate import _validate_dataframe

        df = self._make_clean_df()
        df.loc[0, "traffic_index"] = 1.5   # outside [0, 1]
        result = _validate_dataframe(df)
        assert not result.success

    def test_run_validation_raises_on_bad_data(self, tmp_path):
        from validation.validate import run_validation

        df = self._make_clean_df()
        df.loc[0:5, "trip_distance"] = -1.0  # inject failures

        input_path = tmp_path / "raw.parquet"
        df.to_parquet(input_path)

        with pytest.raises(RuntimeError, match="Data contract violated"):
            run_validation(input_path, output_dir=tmp_path / "validated")

    def test_run_validation_returns_path_on_clean_data(self, tmp_path):
        from validation.validate import run_validation

        df = self._make_clean_df()
        input_path = tmp_path / "raw.parquet"
        df.to_parquet(input_path)

        output = run_validation(input_path, output_dir=tmp_path / "validated")
        assert output.exists()


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------
class TestFeatures:
    def _make_validated_df(self, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "trip_id": [f"T-{i}" for i in range(n)],
            "pickup_datetime":     pd.date_range("2024-01-01", periods=n, freq="5min"),
            "dropoff_datetime":    pd.date_range("2024-01-01 00:30", periods=n, freq="5min"),
            "pickup_location_id":  rng.integers(1, 265, n),
            "dropoff_location_id": rng.integers(1, 265, n),
            "trip_distance":       rng.exponential(3, n).round(2),
            "fare_amount":         rng.gamma(3, 5, n).round(2),
            "delay_minutes":       np.abs(rng.normal(8, 5, n)).round(1),
            "weather_code":        rng.integers(1, 10, n),
            "traffic_index":       rng.uniform(0, 1, n).round(3),
        })

    def test_temporal_extractor_adds_cyclic_features(self):
        from features.engineer import TemporalFeatureExtractor

        df = self._make_validated_df()
        df = TemporalFeatureExtractor().transform(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
            assert col in df.columns, f"Missing: {col}"

    def test_cyclic_encoding_range(self):
        from features.engineer import TemporalFeatureExtractor

        df = self._make_validated_df()
        df = TemporalFeatureExtractor().transform(df)
        assert df["hour_sin"].between(-1, 1).all(), "hour_sin out of [-1, 1]"
        assert df["hour_cos"].between(-1, 1).all(), "hour_cos out of [-1, 1]"

    def test_lag_feature_builder(self):
        from features.engineer import TemporalFeatureExtractor, LagFeatureBuilder

        df = self._make_validated_df()
        df = TemporalFeatureExtractor().transform(df)
        df = LagFeatureBuilder(lag_windows=[3, 6]).transform(df)
        assert "delay_rolling_mean_3h" in df.columns
        assert "delay_rolling_mean_6h" in df.columns

    def test_interaction_features(self):
        from features.engineer import TemporalFeatureExtractor, ExternalSignalEncoder

        df = self._make_validated_df()
        df = TemporalFeatureExtractor().transform(df)
        df = ExternalSignalEncoder().transform(df)
        assert "weather_traffic_interaction" in df.columns
        assert "is_bad_weather" in df.columns

    def test_column_dropper_removes_ids(self):
        from features.engineer import (
            TemporalFeatureExtractor, LagFeatureBuilder,
            ExternalSignalEncoder, NumericScaler, ColumnDropper
        )

        df = self._make_validated_df()
        for T in [TemporalFeatureExtractor(), LagFeatureBuilder(), ExternalSignalEncoder()]:
            df = T.transform(df)
        scaler = NumericScaler()
        scaler.fit(df)
        df = scaler.transform(df)
        df = ColumnDropper().transform(df)

        assert "trip_id" not in df.columns
        assert "pickup_datetime" not in df.columns
        assert "delay_minutes" in df.columns  # target must survive


# ---------------------------------------------------------------------------
# Utility: run all tests standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
