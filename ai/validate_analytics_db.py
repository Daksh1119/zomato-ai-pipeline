"""Validate the DuckDB serving layer after an Airflow refresh."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


EXPECTED_TABLES = {
    "app_metadata",
    "dim_restaurants",
    "mart_customer_segment",
    "mart_daily_city_revenue",
    "mart_delivery_sla",
    "mart_income_cuisine",
    "mart_payment_cancellation",
    "mart_restaurant_performance",
    "mart_review_insights",
    "mart_weekday_performance",
    "reviews",
}


def validate_database(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"DuckDB artifact not found: {path}")

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()
        }
        missing = EXPECTED_TABLES - tables
        if missing:
            raise RuntimeError(f"Missing required tables: {', '.join(sorted(missing))}")

        total_reviews = con.execute("SELECT total_reviews FROM app_metadata").fetchone()[0]
        actual_reviews = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        city_days = con.execute("SELECT COUNT(*) FROM mart_daily_city_revenue").fetchone()[0]
        if total_reviews != actual_reviews or actual_reviews == 0 or city_days == 0:
            raise RuntimeError(
                "Serving-layer row-count validation failed: "
                f"metadata_reviews={total_reviews}, actual_reviews={actual_reviews}, city_days={city_days}"
            )

        print(
            "DuckDB validation passed: "
            f"reviews={actual_reviews:,}, city_day_rows={city_days:,}, tables={len(tables)}"
        )
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Zomato DuckDB serving layer.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).with_name("zomato_analytics.duckdb"),
    )
    args = parser.parse_args()
    validate_database(args.database)


if __name__ == "__main__":
    main()
