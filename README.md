# Zomato Data Engineering Pipeline

An end-to-end batch data pipeline built on Snowflake, Apache Airflow 3, dbt, and large language models. Raw food-delivery data is ingested from cloud storage, transformed through a dimensional model in Snowflake, enriched with AI-generated sentiment analysis, and exposed through a conversational analytics application.

---

## Architecture

![Pipeline Architecture](architecture.png)

---

## Overview

The pipeline processes seven source datasets representing the core entities of a food delivery platform — restaurants, users, menus, food items, orders, order line items, and customer reviews. Data moves through three distinct Snowflake schemas (`RAW` → `STAGING` → `MARTS`) and a dedicated `AI` schema for LLM-enriched outputs. The entire workflow is orchestrated by a daily Airflow DAG running inside Docker.

**Key engineering characteristics:**

- Incremental loading with merge deduplication on `FACT_ORDERS`
- Dimensional modelling following the Kimball methodology
- Tag-based DAG splitting to separate core and AI-dependent build phases
- LLM enrichment isolated in its own pipeline step with idempotent upsert logic
- Retrieval-Augmented Generation (RAG) over vectorised review embeddings for conversational analytics
- Natural-language-to-SQL translation backed by a schema-aware system prompt

---

## Technology Stack

| Layer | Technology |
|---|---|
| Cloud Data Warehouse | Snowflake |
| Orchestration | Apache Airflow 3 (Docker, LocalExecutor) |
| Transformation | dbt-snowflake 1.8 |
| LLM Inference | Groq API (OpenAI-compatible, `openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Cache | Apache Parquet (local disk via `pyarrow`) |
| Analytics Application | Streamlit |
| Containerisation | Docker Compose |
| Source Control | GitHub |
| Cloud Deployment | Streamlit Community Cloud |

---

## Repository Structure

```
zomato-de-pipeline/
├── airflow/
│   ├── Dockerfile                  # Custom Airflow 3 image with providers and dbt venv
│   ├── docker-compose.yaml         # Services: postgres, airflow-init, apiserver, scheduler, dag-processor
│   └── dags/
│       └── zomato_batch.py         # Primary orchestration DAG (daily schedule)
│
├── zomato/                         # dbt project root
│   ├── dbt_project.yml             # Project config: materialization defaults, schema targets
│   ├── profiles.yml                # Snowflake connection profile (git-ignored)
│   └── models/
│       ├── staging/                # Source-aligned views: type casting, column renaming, filtering
│       │   ├── _sources.yml        # RAW schema source declarations
│       │   ├── _ai_sources.yml     # AI schema source declaration (REVIEW_ENRICHED)
│       │   ├── _staging.yml        # Staging model data quality tests
│       │   ├── stg_orders.sql
│       │   ├── stg_order_items.sql
│       │   ├── stg_restaurants.sql
│       │   ├── stg_users.sql
│       │   ├── stg_food.sql
│       │   ├── stg_menu.sql
│       │   └── stg_reviews.sql
│       └── marts/                  # Business-layer tables and aggregate marts
│           ├── _marts.yml          # Referential integrity and accepted-values tests
│           ├── fact_orders.sql     # Incremental fact table (merge strategy)
│           ├── fact_order_items.sql
│           ├── dim_customer.sql    # Includes derived age_segment column
│           ├── dim_restaurants.sql
│           ├── dim_date.sql
│           ├── mart_daily_city_revenue.sql
│           ├── mart_restaurant_performance.sql
│           ├── mart_delivery_sla.sql
│           └── mart_review_insights.sql   # tag:ai — built after LLM enrichment
│
├── ai/
│   ├── enrich_reviews.py           # Batch LLM enrichment: classifies review sentiment and topics
│   ├── rag_chat.py                 # Standalone RAG Streamlit application (per-app entrypoint)
│   ├── text_to_sql.py              # Standalone Text-to-SQL Streamlit application
│   ├── app.py                      # Unified multi-tab Streamlit hub (primary entrypoint)
│   └── requirements.txt            # Python dependencies for AI layer
│
├── data/                           # Source CSVs (git-ignored; loaded to Snowflake stage)
├── architecture.png                # End-to-end pipeline diagram
├── requirements.txt                # Root-level requirements for Streamlit Cloud
├── app.py                          # Root entrypoint that delegates to ai/app.py
└── .gitignore
```

---

## Data Pipeline: Stage-by-Stage

### 1. Ingestion — Snowflake COPY INTO

Raw CSV files are uploaded to a named Snowflake internal stage (`ZOMATO.RAW.ZOMATO_RAW_STAGE`) partitioned by entity. The Airflow `reload_raw` task issues `COPY INTO` statements for all seven tables using Snowflake's bulk loader, which parallelises file reads, enforces schema, and logs per-file load metadata in `INFORMATION_SCHEMA.LOAD_HISTORY`.

**Source datasets (approximate sizes):**

| Table | File Size |
|---|---|
| `orders` | ~1.3 GB |
| `order_items` | ~955 MB |
| `menu` | ~66 MB |
| `restaurant` | ~47 MB |
| `reviews` | ~25 MB |
| `food` | ~17 MB |
| `users` | ~11 MB |

### 2. Staging — dbt Views (`ZOMATO.STAGING`)

Staging models are materialised as **views** — they add no storage cost and always reflect the current state of the RAW layer. Responsibilities at this layer:

- **Type coercion**: casting raw strings to `NUMBER`, `DATE`, `STRING` using explicit Snowflake cast operators.
- **Column standardisation**: renaming source-specific keys (`user_id` → `customer_id`, `r_id` → `restaurant_id`) to a unified vocabulary used downstream.
- **Derived boolean flags**: `(order_status = 'Delivered') AS is_delivered` replaces repeated `CASE` expressions across downstream models.
- **City extraction**: `REGEXP_SUBSTR` + `TRIM` normalises inconsistently formatted city strings from the orders source.
- **Null filtering**: `stg_reviews` excludes rows where `comment IS NULL`, preventing empty strings from entering the LLM enrichment pipeline.
- **Cross-entity join at staging**: `stg_reviews` joins to `stg_restaurants` to propagate the `city` attribute onto reviews, enabling city-level sentiment aggregation in the AI mart without requiring a runtime join.

**Data quality tests** are declared in `_staging.yml` and `_marts.yml`:

```yaml
# Examples
- name: order_id
  tests: [unique, not_null]
- name: order_status
  tests:
    - accepted_values: { values: ['Delivered', 'Cancelled', 'Refunded'] }
- name: customer_id
  tests:
    - relationships: { to: ref('dim_customer'), field: customer_id }
```

### 3. Marts — dbt Tables (`ZOMATO.MARTS`)

Marts are materialised as **tables** in Snowflake, pre-computed and query-optimised for analytical access patterns.

#### Dimensional Model

The mart layer follows a **star schema** design:

| Model | Type | Description |
|---|---|---|
| `FACT_ORDERS` | Fact (incremental) | One row per order. Central transactional table. |
| `FACT_ORDER_ITEMS` | Fact | One row per line item within an order. |
| `DIM_CUSTOMER` | Dimension | Demographic profile with derived `age_segment` (Gen Z / Millennial / Gen X / Boomer). |
| `DIM_RESTAURANTS` | Dimension | Restaurant attributes: name, city, cuisine, rating, cost. |
| `DIM_DATE` | Dimension | Calendar spine with `is_weekend` flag for temporal analysis. |

#### Incremental Strategy on `FACT_ORDERS`

```sql
{{ config(
    materialized='incremental',
    unique_key='order_id',
    incremental_strategy='merge',
    on_schema_change='append_new_columns'
) }}
...
{% if is_incremental() %}
  where order_timestamp > (select coalesce(max(order_timestamp), '1900-01-01'::timestamp) from {{ this }})
{% endif %}
```

On each pipeline run, dbt computes a high-water mark from `max(order_timestamp)` in the existing table and only processes rows newer than that threshold. The `merge` strategy ensures late-arriving or corrected records upsert correctly via `order_id` as the unique key.

#### Pre-Aggregated Mart Tables

| Mart | Grain | Key Metrics |
|---|---|---|
| `MART_DAILY_CITY_REVENUE` | `order_date, city` | GMV, AOV, order count, cancellation rate |
| `MART_RESTAURANT_PERFORMANCE` | `restaurant_id` | Revenue, average rating, average delivery time |
| `MART_DELIVERY_SLA` | `city, order_hour` | P50 (median) and P90 delivery minutes |
| `MART_REVIEW_INSIGHTS` | `city, topic, sentiment_label` | Average sentiment score, average star rating, flagged issue count |

**Business metric definitions encoded in the mart layer:**

- **GMV** (Gross Merchandise Value): `SUM(sales_amount) WHERE is_delivered = TRUE` — delivered revenue only, excluding cancelled and refunded orders.
- **AOV** (Average Order Value): `DIV0(GMV, delivered_orders)` — Snowflake's null-safe division prevents divide-by-zero errors.
- **Cancel Rate**: `DIV0(COUNT_IF(order_status = 'Cancelled'), COUNT(*))` — computed at daily city grain.
- **P90 Delivery SLA**: `PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY delivery_time_min)` — 90th-percentile delivery time per city per hour of day.

### 4. LLM Enrichment — `ZOMATO.AI.REVIEW_ENRICHED`

`ai/enrich_reviews.py` runs as the third Airflow task (`enrich_reviews`). It:

1. Connects to Snowflake and creates `ZOMATO.AI` schema and `REVIEW_ENRICHED` table if they do not exist.
2. Queries `ZOMATO.RAW.REVIEWS` for reviews not yet present in `REVIEW_ENRICHED` (idempotent exclusion filter).
3. For each review, sends the comment text to the Groq API with a structured JSON prompt requesting:
   - `sentiment_label`: `positive`, `negative`, or `neutral`
   - `sentiment_score`: float between -1.0 and 1.0
   - `topic`: one of `food quality`, `delivery`, `pricing`, `service`, `packaging`, `other`
   - `key_issue`: a short phrase describing the primary complaint, or `null`
4. Inserts all results in a single `executemany` call.

The `SAMPLE_N` environment variable controls how many reviews are processed per run, allowing the batch size to be tuned without a code change.

### 5. AI Mart Build — `MART_REVIEW_INSIGHTS`

After enrichment completes, the final Airflow task (`dbt_build_ai`) runs `dbt build --select tag:ai`, which targets only models tagged with `ai`. `MART_REVIEW_INSIGHTS` joins `REVIEW_ENRICHED` to `stg_reviews` on `review_id` and aggregates sentiment metrics by `city`, `topic`, and `sentiment_label`.

The deliberate tag-based split prevents this mart from being attempted during `dbt_build_core`, when the AI schema may not yet contain fresh data.

---

## Orchestration — Apache Airflow 3

### DAG: `zomato_batch`

```
reload_raw  >>  dbt_build_core  >>  enrich_reviews  >>  dbt_build_ai
```

| Task | Operator | Description |
|---|---|---|
| `reload_raw` | `SQLExecuteQueryOperator` | Issues COPY INTO for all 7 raw tables via `snowflake_default` connection |
| `dbt_build_core` | `BashOperator` | Runs `dbt build --exclude tag:ai` inside the container's dbt venv |
| `enrich_reviews` | `BashOperator` | Executes `python /opt/airflow/ai/enrich_reviews.py` |
| `dbt_build_ai` | `BashOperator` | Runs `dbt build --select tag:ai` to build `MART_REVIEW_INSIGHTS` |

**Schedule**: `@daily`, `catchup=False` — runs once per calendar day, does not backfill missed runs.

### Airflow 3 Architecture

This project uses **Apache Airflow 3**, which introduces a decomposed service model compared to Airflow 2:

| Service | Role |
|---|---|
| `apiserver` | Replaces the legacy `webserver`; exposes the REST API and serves the React UI on port 8080 |
| `scheduler` | Parses no DAGs; solely handles scheduling, triggering, and state management |
| `dag-processor` | Dedicated process for parsing and serialising DAG files |
| `postgres` | Airflow metadata database |

Inter-component communication uses JWT-signed requests. The `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` must point to the `apiserver` service by its Docker network hostname, not `localhost`, to avoid task state mismatches in multi-container deployments.

### Custom Docker Image

dbt is installed in an isolated Python virtual environment (`/opt/airflow/dbt_venv`) to avoid dependency conflicts with Airflow's pinned packages:

```dockerfile
FROM apache/airflow:3.0.3
RUN pip install --no-cache-dir \
      apache-airflow-providers-snowflake \
      apache-airflow-providers-fab \
      openai \
      python-dotenv

RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir "dbt-snowflake==1.8.*"
```

---

## Analytics Application — Streamlit

`ai/app.py` is the unified application exposing two tools through a tab interface:

### Tab 1: Review Intelligence (RAG)

A Retrieval-Augmented Generation system over customer review text.

**Embedding and retrieval:**

1. On first load, `load_reviews()` either reads a local `review_embeddings.parquet` cache or fetches reviews from `ZOMATO.STAGING.STG_REVIEWS` via Snowflake.
2. The `sentence-transformers` model (`all-MiniLM-L6-v2`) encodes review text to 384-dimensional normalised float vectors, stored in the parquet cache for sub-second subsequent loads.
3. At query time, the user's question is embedded using the same model. Semantic similarity is computed via **matrix dot product** (`numpy.dot` against the full embedding matrix), leveraging BLAS-accelerated linear algebra. Because vectors are L2-normalised, the dot product equals cosine similarity.
4. The top-K most similar reviews are retrieved and passed as structured context to the Groq LLM, which synthesises a grounded natural-language answer.

**Performance characteristics**: The parquet cache and vectorised dot product make similarity search over hundreds of reviews operate in under 50 ms on CPU, without requiring a vector database.

### Tab 2: SQL Data Analyst (Text-to-SQL)

A natural-language-to-SQL interface against the full Snowflake MARTS schema.

- A schema-aware system prompt encodes all 10 table definitions, column names, join keys, and business metric definitions (GMV, AOV, cancel rate, SLA percentiles).
- The Groq LLM returns a JSON object `{"sql": "..."}` which is parsed and cleaned (schema prefix stripping, semicolon removal).
- A **SQL safety guard** (`is_safe_sql`) rejects any statement that does not begin with `SELECT` or `WITH`, or contains DML/DDL keywords, before execution.
- Results are rendered as an interactive `st.dataframe`. Two-column numeric results automatically generate a bar chart.

---

## Snowflake Schema Design

```
ZOMATO (database)
├── RAW          ← COPY INTO from internal stage; raw source tables
├── STAGING      ← dbt views; typed, renamed, filtered
├── MARTS        ← dbt tables; star schema + pre-aggregated marts
└── AI           ← LLM enrichment output (REVIEW_ENRICHED)
```

---

## Installation and Setup

### Prerequisites

- Docker Desktop
- Python 3.11+
- A Snowflake account with a warehouse, database, and a role with `CREATE SCHEMA`, `CREATE TABLE`, and `COPY INTO` privileges
- A Groq API key ([console.groq.com](https://console.groq.com))
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/Daksh1119/zomato-de-pipeline.git
cd zomato-de-pipeline
```

### 2. Configure Snowflake

In your Snowflake account, run the following setup script:

```sql
CREATE DATABASE IF NOT EXISTS ZOMATO;
CREATE WAREHOUSE IF NOT EXISTS ZOMATO_WH WAREHOUSE_SIZE = 'X-SMALL' AUTO_SUSPEND = 60;
CREATE ROLE IF NOT EXISTS DBT_ROLE;

CREATE SCHEMA IF NOT EXISTS ZOMATO.RAW;
CREATE SCHEMA IF NOT EXISTS ZOMATO.STAGING;
CREATE SCHEMA IF NOT EXISTS ZOMATO.MARTS;

GRANT USAGE ON WAREHOUSE ZOMATO_WH TO ROLE DBT_ROLE;
GRANT ALL ON DATABASE ZOMATO TO ROLE DBT_ROLE;
GRANT ALL ON ALL SCHEMAS IN DATABASE ZOMATO TO ROLE DBT_ROLE;

-- Create internal stage for CSV ingestion
USE SCHEMA ZOMATO.RAW;
CREATE STAGE IF NOT EXISTS ZOMATO_RAW_STAGE;
```

Upload the source CSVs to the stage, partitioned by folder:

```bash
# Using SnowSQL
PUT file://data/orders.csv       @ZOMATO.RAW.ZOMATO_RAW_STAGE/orders/
PUT file://data/order_items.csv  @ZOMATO.RAW.ZOMATO_RAW_STAGE/order_items/
PUT file://data/restaurants.csv  @ZOMATO.RAW.ZOMATO_RAW_STAGE/restaurants/
PUT file://data/users.csv        @ZOMATO.RAW.ZOMATO_RAW_STAGE/users/
PUT file://data/food.csv         @ZOMATO.RAW.ZOMATO_RAW_STAGE/food/
PUT file://data/menu.csv         @ZOMATO.RAW.ZOMATO_RAW_STAGE/menu/
PUT file://data/reviews.csv      @ZOMATO.RAW.ZOMATO_RAW_STAGE/reviews/
```

Create the raw table definitions that match the source CSV schemas before running `COPY INTO`.

### 3. Configure Environment Variables

Create an `.env` file in the `airflow/` directory. Use the example below as a template:

```bash
cp airflow/.env.example airflow/.env
```

**`airflow/.env.example`:**

```dotenv
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
GROQ_API_KEY=gsk_your_groq_api_key
OPENAI_API_KEY=
SAMPLE_N=500
```

Create an `.env` file in `ai/` for local Streamlit development:

```bash
cp ai/.env.example ai/.env
```

**`ai/.env.example`:**

```dotenv
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=ZOMATO_WH
SNOWFLAKE_DATABASE=ZOMATO
SNOWFLAKE_SCHEMA=MARTS
SNOWFLAKE_ROLE=DBT_ROLE
GROQ_API_KEY=gsk_your_groq_api_key
```

### 4. Start Airflow

```bash
cd airflow
docker compose build
docker compose up -d
```

Wait for `airflow-init` to complete (approximately 60 seconds), then access the Airflow UI at [http://localhost:8080](http://localhost:8080) with credentials `admin / admin`.

In the Airflow UI:

1. Navigate to **Admin → Connections**.
2. Confirm that `snowflake_default` is present (it is auto-configured via the `AIRFLOW_CONN_SNOWFLAKE_DEFAULT` environment variable in `docker-compose.yaml`).
3. Enable and trigger the `zomato_batch` DAG.

### 5. Run the dbt Project Locally (Optional)

```bash
cd zomato
pip install dbt-snowflake==1.8.*

# Create profiles.yml (git-ignored) at zomato/profiles.yml
dbt debug           # verify connection
dbt build           # run all models and tests
```

### 6. Run the Streamlit Application Locally

```bash
cd ai
pip install -r requirements.txt
streamlit run app.py
```

The application will be available at [http://localhost:8501](http://localhost:8501).

---

## Streamlit Cloud Deployment

The application is deployable directly from GitHub at [share.streamlit.io](https://share.streamlit.io).

| Setting | Value |
|---|---|
| Repository | `Daksh1119/zomato-de-pipeline` |
| Branch | `main` |
| Main file path | `ai/app.py` |
| Python version | `3.11` |

In **Advanced Settings → Secrets**, add all keys from `ai/.env.example` in TOML format:

```toml
SNOWFLAKE_ACCOUNT = "your_account_identifier"
SNOWFLAKE_USER = "your_username"
SNOWFLAKE_PASSWORD = "your_password"
SNOWFLAKE_WAREHOUSE = "ZOMATO_WH"
SNOWFLAKE_DATABASE = "ZOMATO"
SNOWFLAKE_SCHEMA = "MARTS"
SNOWFLAKE_ROLE = "DBT_ROLE"
GROQ_API_KEY = "gsk_your_groq_api_key"
```

---

## Key Engineering Decisions

**Incremental loading over full refresh**: `FACT_ORDERS` uses a high-water mark on `order_timestamp` with merge deduplication. This reduces Snowflake compute credits and query time proportional to the day's delta rather than the full table size.

**dbt venv isolation**: Installing `dbt-snowflake` inside a virtualenv inside the Airflow container avoids dependency resolver conflicts between dbt's pinned packages and Airflow's provider ecosystem.

**Tag-based DAG splitting**: Tagging `mart_review_insights` with `tag:ai` and splitting `dbt_build_core` and `dbt_build_ai` into separate Airflow tasks ensures that the AI mart is never built before the enrichment step has written its data. It also allows the core dimensional model to be rebuilt independently.

**Parquet embedding cache**: Storing sentence-transformer embeddings in Parquet eliminates the model inference cost on every Streamlit reboot. The application starts in under one second on cache hit, versus 30+ seconds for a full re-embed from Snowflake.

**Groq as LLM provider**: Groq's inference engine delivers low-latency responses suitable for interactive applications. The OpenAI-compatible API surface means the client code requires no provider-specific SDK.

**`DIV0` and `COUNT_IF` for analytical safety**: Snowflake-native functions `DIV0` (null-safe division) and `COUNT_IF` (conditional aggregation without `CASE WHEN`) are used throughout the mart layer for correctness and readability.

---

## Security

- All credentials are managed via environment variables and are never committed to version control.
- `profiles.yml` is covered by `.gitignore`.
- `ai/.env` and `airflow/.env` are covered by `.gitignore`.
- Source data CSVs and generated parquet files are excluded from git via `data/` and `*.parquet` patterns.
- The Text-to-SQL application enforces a read-only guard: generated SQL is rejected if it does not begin with `SELECT` or `WITH`, or if it contains any DML or DDL keyword.
