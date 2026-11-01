"""Build the compact DuckDB database served by the Streamlit dashboard.

The source CSVs stay outside Git and can be several gigabytes.  This script
derives only the aggregates the analytics UI needs, which makes the resulting
database small enough to deploy with the application.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = Path(__file__).with_name("zomato_analytics.duckdb")


def sql_path(path: Path) -> str:
    """Return a SQL string literal-safe absolute path."""
    return path.resolve().as_posix().replace("'", "''")


def build_database(data_dir: Path, output_path: Path) -> None:
    required_files = ["orders.csv", "reviews.csv", "users.csv", "restaurant.csv"]
    missing = [name for name in required_files if not (data_dir / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing source CSV files in {data_dir}: {missing_list}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling file makes the builder safe even if an operator
    # accidentally starts a second run while the DAG-level concurrency guard
    # is in effect. Only a completed build replaces the published artifact.
    staging_path = output_path.with_name(f"{output_path.name}.{uuid4().hex}.building")

    # Build beside the active artifact and publish only after a successful
    # checkpoint. This prevents a failed scheduled run from breaking Streamlit.
    con = None
    try:
        con = duckdb.connect(str(staging_path))
        orders = sql_path(data_dir / "orders.csv")
        reviews = sql_path(data_dir / "reviews.csv")
        users = sql_path(data_dir / "users.csv")
        restaurants = sql_path(data_dir / "restaurant.csv")

        # CSVs are read as strings first so the transformation is explicit and
        # resilient to the known messy source fields.
        con.execute(f"""
            CREATE VIEW source_orders AS
            SELECT
                TRY_CAST(order_id AS BIGINT) AS order_id,
                TRY_CAST(order_timestamp AS TIMESTAMP) AS order_timestamp,
                TRY_CAST(order_date AS DATE) AS order_date,
                TRY_CAST(user_id AS BIGINT) AS customer_id,
                TRY_CAST(r_id AS BIGINT) AS restaurant_id,
                TRIM(regexp_extract(restaurant_city, '[^,]+$')) AS city,
                cuisine,
                payment_method,
                order_status,
                order_status = 'Delivered' AS is_delivered,
                TRY_CAST(sales_amount AS DOUBLE) AS sales_amount,
                TRY_CAST(customer_rating AS DOUBLE) AS customer_rating,
                TRY_CAST(delivery_time_min AS DOUBLE) AS delivery_time_min
            FROM read_csv('{orders}', header = true, all_varchar = true)
            WHERE TRY_CAST(order_id AS BIGINT) IS NOT NULL
        """)
        con.execute(f"""
            CREATE VIEW source_users AS
            SELECT
                TRY_CAST(user_id AS BIGINT) AS customer_id,
                CASE
                    WHEN TRY_CAST(age AS INTEGER) < 25 THEN 'Gen Z'
                    WHEN TRY_CAST(age AS INTEGER) < 40 THEN 'Millennial'
                    WHEN TRY_CAST(age AS INTEGER) < 55 THEN 'Gen X'
                    WHEN TRY_CAST(age AS INTEGER) IS NULL THEN 'Unknown'
                    ELSE 'Boomer'
                END AS age_segment,
                "Monthly Income" AS income_band
            FROM read_csv('{users}', header = true, all_varchar = true)
            WHERE TRY_CAST(user_id AS BIGINT) IS NOT NULL
        """)
        con.execute(f"""
            CREATE TABLE dim_restaurants AS
            SELECT
                TRY_CAST(id AS BIGINT) AS restaurant_id,
                name AS restaurant_name,
                TRIM(regexp_extract(city, '[^,]+$')) AS city,
                cuisine,
                TRY_CAST(NULLIF(rating, '--') AS DOUBLE) AS rating,
                TRY_CAST(regexp_extract(rating_count, '[0-9]+') AS BIGINT) AS rating_count,
                TRY_CAST(regexp_extract(cost, '[0-9]+') AS BIGINT) AS cost_for_two
            FROM read_csv('{restaurants}', header = true, all_varchar = true)
            WHERE TRY_CAST(id AS BIGINT) IS NOT NULL
        """)

        con.execute("""
            CREATE TABLE mart_daily_city_revenue AS
            SELECT
                order_date,
                city,
                COUNT(*) AS orders,
                COUNT(*) FILTER (WHERE is_delivered) AS delivered_orders,
                ROUND(COUNT(*) FILTER (WHERE order_status = 'Cancelled') / COUNT(*)::DOUBLE, 4) AS cancel_rate,
                ROUND(SUM(CASE WHEN is_delivered THEN sales_amount ELSE 0 END), 2) AS gmv,
                ROUND(SUM(CASE WHEN is_delivered THEN sales_amount ELSE 0 END)
                    / NULLIF(COUNT(*) FILTER (WHERE is_delivered), 0), 2) AS aov
            FROM source_orders
            GROUP BY order_date, city
        """)
        con.execute("""
            CREATE TABLE mart_restaurant_performance AS
            SELECT
                o.restaurant_id,
                COALESCE(r.restaurant_name, 'Unknown') AS restaurant_name,
                COALESCE(r.city, o.city) AS city,
                COALESCE(r.cuisine, o.cuisine) AS cuisine,
                COUNT(*) AS orders,
                ROUND(SUM(CASE WHEN o.is_delivered THEN o.sales_amount ELSE 0 END), 2) AS revenue,
                ROUND(AVG(o.customer_rating), 2) AS avg_customer_rating,
                ROUND(AVG(o.delivery_time_min), 1) AS avg_delivery_min
            FROM source_orders AS o
            LEFT JOIN dim_restaurants AS r USING (restaurant_id)
            GROUP BY ALL
        """)
        con.execute("""
            CREATE TABLE mart_delivery_sla AS
            SELECT
                city,
                EXTRACT(HOUR FROM order_timestamp) AS order_hour,
                COUNT(*) AS delivered_orders,
                ROUND(MEDIAN(delivery_time_min), 1) AS p50,
                ROUND(quantile_cont(delivery_time_min, 0.9), 1) AS p90
            FROM source_orders
            WHERE is_delivered AND delivery_time_min IS NOT NULL
            GROUP BY city, order_hour
        """)
        con.execute("""
            CREATE TABLE mart_payment_cancellation AS
            SELECT
                payment_method,
                COUNT(*) AS orders,
                COUNT(*) FILTER (WHERE order_status = 'Cancelled') AS cancelled_orders,
                ROUND(COUNT(*) FILTER (WHERE order_status = 'Cancelled') / COUNT(*)::DOUBLE, 4) AS cancel_rate
            FROM source_orders
            GROUP BY payment_method
        """)
        con.execute("""
            CREATE TABLE mart_customer_segment AS
            SELECT
                u.age_segment,
                o.cuisine,
                COUNT(*) AS orders,
                COUNT(*) FILTER (WHERE o.is_delivered) AS delivered_orders,
                ROUND(SUM(CASE WHEN o.is_delivered THEN o.sales_amount ELSE 0 END), 2) AS gmv,
                ROUND(SUM(CASE WHEN o.is_delivered THEN o.sales_amount ELSE 0 END)
                    / NULLIF(COUNT(*) FILTER (WHERE o.is_delivered), 0), 2) AS aov
            FROM source_orders AS o
            LEFT JOIN source_users AS u USING (customer_id)
            GROUP BY u.age_segment, o.cuisine
        """)
        con.execute("""
            CREATE TABLE mart_income_cuisine AS
            SELECT
                COALESCE(u.income_band, 'Unknown') AS income_band,
                o.cuisine,
                COUNT(*) AS orders,
                ROUND(SUM(CASE WHEN o.is_delivered THEN o.sales_amount ELSE 0 END), 2) AS gmv
            FROM source_orders AS o
            LEFT JOIN source_users AS u USING (customer_id)
            GROUP BY u.income_band, o.cuisine
        """)
        con.execute("""
            CREATE TABLE mart_weekday_performance AS
            SELECT
                CASE WHEN dayofweek(order_date) IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                COUNT(*) AS orders,
                COUNT(*) FILTER (WHERE is_delivered) AS delivered_orders,
                ROUND(SUM(CASE WHEN is_delivered THEN sales_amount ELSE 0 END), 2) AS gmv,
                ROUND(SUM(CASE WHEN is_delivered THEN sales_amount ELSE 0 END)
                    / NULLIF(COUNT(*) FILTER (WHERE is_delivered), 0), 2) AS aov
            FROM source_orders
            GROUP BY day_type
        """)

        # Keep the entire review corpus in the serving layer. The Streamlit app
        # builds a bounded sparse index over this table at runtime.
        con.execute(f"""
            CREATE TABLE reviews AS
            WITH restaurant_city AS (
                SELECT restaurant_id, city FROM dim_restaurants
            )
            SELECT
                TRY_CAST(r.review_id AS BIGINT) AS review_id,
                COALESCE(rc.city, 'Unknown') AS city,
                TRY_CAST(r.rating AS INTEGER) AS rating,
                r.comment
            FROM read_csv('{reviews}', header = true, all_varchar = true) AS r
            LEFT JOIN restaurant_city AS rc
                ON TRY_CAST(r.restaurant_id AS BIGINT) = rc.restaurant_id
            WHERE r.comment IS NOT NULL AND TRIM(r.comment) <> ''
            ORDER BY TRY_CAST(r.review_id AS BIGINT)
        """)
        con.execute("""
            CREATE TABLE mart_review_insights AS
            WITH classified_reviews AS (
                SELECT
                    city,
                    CASE
                        WHEN rating >= 4 THEN 'Positive'
                        WHEN rating <= 2 THEN 'Negative'
                        ELSE 'Neutral'
                    END AS sentiment_label,
                    CASE
                        WHEN lower(comment) LIKE '%deliver%' THEN 'Delivery'
                        WHEN lower(comment) LIKE '%packag%' THEN 'Packaging'
                        WHEN lower(comment) LIKE '%price%' OR lower(comment) LIKE '%cost%' THEN 'Pricing'
                        WHEN lower(comment) LIKE '%food%' OR lower(comment) LIKE '%taste%' THEN 'Food Quality'
                        ELSE 'General'
                    END AS topic,
                    CASE
                        WHEN rating <= 2 THEN 1 ELSE 0
                    END AS flagged_issue
                FROM reviews
            )
            SELECT
                city,
                topic,
                sentiment_label,
                COUNT(*) AS reviews,
                ROUND(AVG(CASE sentiment_label WHEN 'Positive' THEN 1 WHEN 'Negative' THEN -1 ELSE 0 END), 3) AS avg_sentiment_score,
                SUM(flagged_issue) AS flagged_issues
            FROM classified_reviews
            GROUP BY ALL
        """)

        # The final artifact is self-contained: it must not retain views that
        # reference the large, local-only CSV source paths.
        con.execute("DROP VIEW source_orders")
        con.execute("DROP VIEW source_users")
        con.execute("CREATE TABLE app_metadata AS SELECT COUNT(*) AS total_reviews FROM reviews")
        con.execute("CHECKPOINT")
    except Exception:
        # Keep the previous published artifact available to Streamlit and avoid
        # a stale staging file blocking the next scheduled Airflow run.
        if con is not None:
            con.close()
        staging_path.unlink(missing_ok=True)
        raise
    else:
        assert con is not None
        con.close()
        staging_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deployable Zomato DuckDB analytics database.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    build_database(args.data_dir, args.output)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"Built {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
