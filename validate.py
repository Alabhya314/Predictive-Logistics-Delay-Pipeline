"""
validation/validate.py
----------------------
Implements the Data Contract using Great Expectations.
If any expectation fails the pipeline is halted and a structured alert
is written to data/validated/validation_report.json.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("validation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
def _build_expectations(validator: Any) -> None:
    """
    Define the full Data Contract for raw logistics data.

    Rules fall into four categories:
      1. Schema – column presence and types
      2. Completeness – no unexpected nulls
      3. Logical bounds – domain constraints
      4. Referential integrity – business rules
    """
    # ── 1. Schema ──────────────────────────────────────────────────────────
    required_columns = [
        "trip_id", "pickup_datetime", "dropoff_datetime",
        "pickup_location_id", "dropoff_location_id",
        "trip_distance", "fare_amount", "delay_minutes",
        "weather_code", "traffic_index",
    ]
    validator.expect_table_columns_to_match_set(
        column_set=required_columns,
        exact_match=False,
    )

    validator.expect_column_values_to_be_of_type("trip_distance", "float64")
    validator.expect_column_values_to_be_of_type("fare_amount",   "float64")
    validator.expect_column_values_to_be_of_type("delay_minutes", "float64")
    validator.expect_column_values_to_be_of_type("traffic_index", "float64")

    # ── 2. Completeness ────────────────────────────────────────────────────
    for col in required_columns:
        validator.expect_column_values_to_not_be_null(col, mostly=0.99)

    # ── 3. Logical bounds ──────────────────────────────────────────────────
    # Distance cannot be negative
    validator.expect_column_values_to_be_between(
        "trip_distance", min_value=0.0, max_value=200.0,
        mostly=0.999,
    )
    # Fare cannot be negative
    validator.expect_column_values_to_be_between(
        "fare_amount", min_value=0.0, max_value=5_000.0,
        mostly=0.999,
    )
    # Delay cannot be negative
    validator.expect_column_values_to_be_between(
        "delay_minutes", min_value=0.0,
        mostly=0.999,
    )
    # Traffic index must be in [0, 1]
    validator.expect_column_values_to_be_between(
        "traffic_index", min_value=0.0, max_value=1.0,
        mostly=0.999,
    )
    # Weather code must be a valid integer range
    validator.expect_column_values_to_be_between(
        "weather_code", min_value=1, max_value=10,
    )
    # Location IDs
    validator.expect_column_values_to_be_between(
        "pickup_location_id", min_value=1, max_value=265,
    )
    validator.expect_column_values_to_be_between(
        "dropoff_location_id", min_value=1, max_value=265,
    )

    # ── 4. Referential integrity ───────────────────────────────────────────
    # trip_id must be unique
    validator.expect_column_values_to_be_unique("trip_id")

    # Row count sanity check
    validator.expect_table_row_count_to_be_between(min_value=100, max_value=10_000_000)


# ---------------------------------------------------------------------------
# Core validation logic (without Great Expectations dependency)
# ---------------------------------------------------------------------------
class ValidationResult:
    """Lightweight validation result container."""

    def __init__(self):
        self.failures: list[dict] = []
        self.success_count: int = 0
        self.total_count: int = 0

    @property
    def success(self) -> bool:
        return len(self.failures) == 0

    def add_check(self, name: str, passed: bool, details: str = "") -> None:
        self.total_count += 1
        if passed:
            self.success_count += 1
        else:
            self.failures.append({"check": name, "details": details})
            log.warning("FAILED: %s | %s", name, details)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "evaluated_at": datetime.utcnow().isoformat(),
            "statistics": {
                "evaluated_expectations": self.total_count,
                "successful_expectations": self.success_count,
                "unsuccessful_expectations": len(self.failures),
            },
            "failures": self.failures,
        }


def _validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Run the data contract checks against a pandas DataFrame.
    This is the framework-agnostic implementation that works
    without Great Expectations installed.
    """
    result = ValidationResult()

    # ── 1. Schema ──────────────────────────────────────────────────────────
    required_columns = [
        "trip_id", "pickup_datetime", "dropoff_datetime",
        "pickup_location_id", "dropoff_location_id",
        "trip_distance", "fare_amount", "delay_minutes",
        "weather_code", "traffic_index",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    result.add_check(
        "required_columns_present",
        len(missing) == 0,
        f"Missing columns: {missing}" if missing else "",
    )

    # ── 2. Completeness ────────────────────────────────────────────────────
    for col in required_columns:
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        result.add_check(
            f"nulls_{col}",
            null_rate <= 0.01,
            f"null_rate={null_rate:.4f} (threshold=0.01)",
        )

    # ── 3. Logical bounds ──────────────────────────────────────────────────
    if "trip_distance" in df.columns:
        bad = (df["trip_distance"] < 0).sum()
        result.add_check(
            "trip_distance_non_negative",
            bad == 0,
            f"{bad} rows with trip_distance < 0",
        )
        too_far = (df["trip_distance"] > 200).sum()
        result.add_check(
            "trip_distance_max_200",
            too_far == 0,
            f"{too_far} rows with trip_distance > 200",
        )

    if "fare_amount" in df.columns:
        bad = (df["fare_amount"] < 0).sum()
        result.add_check(
            "fare_amount_non_negative",
            bad == 0,
            f"{bad} rows with fare_amount < 0",
        )

    if "delay_minutes" in df.columns:
        bad = (df["delay_minutes"] < 0).sum()
        result.add_check(
            "delay_minutes_non_negative",
            bad == 0,
            f"{bad} rows with delay_minutes < 0",
        )

    if "traffic_index" in df.columns:
        bad = ((df["traffic_index"] < 0) | (df["traffic_index"] > 1)).sum()
        result.add_check(
            "traffic_index_in_0_1",
            bad == 0,
            f"{bad} rows outside [0,1]",
        )

    if "weather_code" in df.columns:
        bad = (~df["weather_code"].between(1, 10)).sum()
        result.add_check(
            "weather_code_in_1_10",
            bad == 0,
            f"{bad} invalid weather codes",
        )

    # ── 4. Referential integrity ───────────────────────────────────────────
    if "trip_id" in df.columns:
        dupes = df["trip_id"].duplicated().sum()
        result.add_check(
            "trip_id_unique",
            dupes == 0,
            f"{dupes} duplicate trip_ids",
        )

    if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
        try:
            bad = (
                pd.to_datetime(df["pickup_datetime"])
                >= pd.to_datetime(df["dropoff_datetime"])
            ).sum()
            result.add_check(
                "pickup_before_dropoff",
                bad == 0,
                f"{bad} rows where pickup >= dropoff",
            )
        except Exception as exc:
            result.add_check("pickup_before_dropoff", False, str(exc))

    result.add_check(
        "row_count_minimum",
        len(df) >= 100,
        f"Only {len(df)} rows (minimum=100)",
    )

    return result


def _try_great_expectations(df: pd.DataFrame) -> "ValidationResult | None":
    """
    Attempt to run Great Expectations if available.
    Returns None if GE is not installed.
    """
    try:
        import great_expectations as ge  # type: ignore

        gdf = ge.from_pandas(df)
        _build_expectations(gdf)

        results = gdf.validate()
        vr = ValidationResult()

        for er in results.results:
            expectation_type = er.expectation_config.expectation_type
            passed = er.success
            details = ""
            if not passed and er.result:
                details = json.dumps(
                    {k: v for k, v in er.result.items() if k != "partial_unexpected_list"},
                    default=str,
                )
            vr.add_check(expectation_type, passed, details)

        return vr

    except ImportError:
        log.info(
            "great_expectations not installed — using built-in validator."
        )
        return None


def run_validation(
    input_path: Path | str,
    output_dir: Path | str = "data/validated",
) -> Path:
    """
    Validate raw data against the Data Contract.
    Halts (raises RuntimeError) if any expectation fails.

    Returns
    -------
    Path to the validated parquet file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw data from %s …", input_path)
    df = pd.read_parquet(input_path)
    log.info("Loaded %d rows × %d columns", *df.shape)

    # Try GE first, fall back to built-in
    result = _try_great_expectations(df)
    if result is None:
        result = _validate_dataframe(df)

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    log.info("Validation report written → %s", report_path)

    if not result.success:
        n_fail = len(result.failures)
        msg = (
            f"Data contract violated: {n_fail} expectation(s) failed. "
            f"See {report_path} for details.\n"
            + "\n".join(f"  • {f['check']}: {f['details']}" for f in result.failures)
        )
        log.error(msg)
        raise RuntimeError(msg)

    log.info(
        "All %d expectations passed — writing validated dataset …",
        result.total_count,
    )

    validated_path = output_dir / input_path.name
    df.to_parquet(validated_path, index=False, engine="pyarrow")
    log.info("Validated data written → %s", validated_path)
    return validated_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Contract validation")
    parser.add_argument("input", help="Path to raw parquet file")
    parser.add_argument("--output-dir", default="data/validated")
    args = parser.parse_args()

    run_validation(args.input, args.output_dir)
