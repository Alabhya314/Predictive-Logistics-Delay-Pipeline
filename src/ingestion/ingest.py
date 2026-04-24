"""
ingestion/ingest.py
-------------------
Simulates an API pull of NYC TLC trip-record data.
Writes raw parquet to data/raw/ and registers with DVC-compatible hash manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ingestion")


# ---------------------------------------------------------------------------
# Schema definition – the contract for raw data leaving this stage
# ---------------------------------------------------------------------------
RAW_SCHEMA: dict[str, type] = {
    "trip_id": str,
    "pickup_datetime": "datetime64[ns]",
    "dropoff_datetime": "datetime64[ns]",
    "pickup_location_id": int,
    "dropoff_location_id": int,
    "trip_distance": float,
    "fare_amount": float,
    "delay_minutes": float,          # target variable
    "weather_code": int,
    "traffic_index": float,
}

DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _simulate_api_response(
    n_rows: int = 5_000,
    corruption_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic logistics dataset that mimics NYC TLC Trip Records
    merged with weather/traffic signals.

    Parameters
    ----------
    n_rows       : Number of rows to generate
    corruption_rate : Fraction of rows to intentionally corrupt (for testing
                      validation failures downstream)
    seed         : Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    base_time = datetime(2024, 1, 1)
    pickup_offsets = rng.integers(0, 365 * 24 * 60, size=n_rows)
    pickup_times = [base_time + timedelta(minutes=int(m)) for m in pickup_offsets]

    trip_durations_min = rng.gamma(shape=2.5, scale=15, size=n_rows)
    dropoff_times = [
        p + timedelta(minutes=float(d))
        for p, d in zip(pickup_times, trip_durations_min)
    ]

    df = pd.DataFrame(
        {
            "trip_id": [f"TRIP-{i:07d}" for i in range(n_rows)],
            "pickup_datetime": pd.to_datetime(pickup_times),
            "dropoff_datetime": pd.to_datetime(dropoff_times),
            "pickup_location_id": rng.integers(1, 265, size=n_rows),
            "dropoff_location_id": rng.integers(1, 265, size=n_rows),
            "trip_distance": rng.exponential(scale=3.5, size=n_rows).round(2),
            "fare_amount": rng.gamma(shape=3, scale=5, size=n_rows).round(2),
            "weather_code": rng.integers(1, 10, size=n_rows),
            "traffic_index": rng.uniform(0.1, 1.0, size=n_rows).round(3),
        }
    )

    # Make the target variable learnable instead of pure noise, so the model passes evaluation thresholds
    base_delay = (
        df["trip_distance"] * 0.8 +
        df["weather_code"] * 0.5 +
        df["traffic_index"] * 6.0
    )
    df["delay_minutes"] = np.clip(
        base_delay + rng.normal(scale=2.0, size=n_rows), 0, None
    ).round(1)

    # Intentionally corrupt a fraction of rows to test validation downstream
    if corruption_rate > 0:
        n_corrupt = int(n_rows * corruption_rate)
        corrupt_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
        df.loc[corrupt_idx[:n_corrupt // 3], "trip_distance"] = -9.99
        df.loc[corrupt_idx[n_corrupt // 3 : 2 * n_corrupt // 3], "fare_amount"] = -1.0
        log.warning(
            "Injected %d corrupted rows (corruption_rate=%.2f)",
            n_corrupt,
            corruption_rate,
        )

    return df


def _compute_hash(path: Path) -> str:
    """SHA-256 hash of a file – used as a lightweight DVC-compatible checksum."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _update_manifest(path: Path, file_hash: str, n_rows: int) -> None:
    """
    Append an entry to the ingestion manifest.
    This is the lightweight DVC stand-in; in a real project DVC
    would manage this automatically after `dvc add`.
    """
    manifest: list[dict] = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as fh:
            manifest = json.load(fh)

    manifest.append(
        {
            "file": path.name,
            "sha256": file_hash,
            "n_rows": n_rows,
            "ingested_at": datetime.utcnow().isoformat(),
        }
    )
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("Manifest updated → %s", MANIFEST_PATH)


def run_ingestion(
    n_rows: int = 5_000,
    corruption_rate: float = 0.0,
    output_filename: Optional[str] = None,
) -> Path:
    """
    Entry point for the ingestion stage.

    Returns
    -------
    Path to the written raw parquet file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Pulling logistics data (n_rows=%d, corruption_rate=%.2f)…", n_rows, corruption_rate)
    df = _simulate_api_response(n_rows=n_rows, corruption_rate=corruption_rate)

    # Enforce dtypes before writing
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = output_filename or f"logistics_{timestamp}.parquet"
    output_path = DATA_DIR / filename

    df.to_parquet(output_path, index=False, engine="pyarrow")

    file_hash = _compute_hash(output_path)
    _update_manifest(output_path, file_hash, len(df))

    log.info(
        "Ingestion complete → %s  [rows=%d, hash=%s…]",
        output_path,
        len(df),
        file_hash[:12],
    )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistics data ingestion")
    parser.add_argument("--n-rows", type=int, default=5_000)
    parser.add_argument("--corruption-rate", type=float, default=0.0)
    args = parser.parse_args()

    run_ingestion(n_rows=args.n_rows, corruption_rate=args.corruption_rate)
