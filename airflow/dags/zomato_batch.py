"""Daily local analytics refresh: CSV source files -> validated DuckDB artifact."""

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


BUILD_COMMAND = """
set -euo pipefail
python /opt/airflow/ai/build_analytics_db.py \
  --data-dir /opt/airflow/data \
  --output /opt/airflow/ai/zomato_analytics.duckdb
"""

VALIDATE_COMMAND = """
set -euo pipefail
python /opt/airflow/ai/validate_analytics_db.py \
  --database /opt/airflow/ai/zomato_analytics.duckdb
"""

with DAG(
    dag_id="zomato_duckdb_refresh",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["zomato", "duckdb", "serving-layer"],
    doc_md=__doc__,
) as dag:

    build_duckdb_serving_layer = BashOperator(
        task_id="build_duckdb_serving_layer",
        bash_command=BUILD_COMMAND,
    )
    validate_duckdb_serving_layer = BashOperator(
        task_id="validate_duckdb_serving_layer",
        bash_command=VALIDATE_COMMAND,
    )

    build_duckdb_serving_layer >> validate_duckdb_serving_layer
