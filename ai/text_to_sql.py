import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-20b"

FORBIDDEN_WORDS = ['drop', 'delete', 'truncate', 'alter', 'update', 'insert', 'create', 'replace', 'grant', 'revoke']

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the highest total order revenue?",
    "Average delivery time by city, longest first",
    "Order cancellation rate by payment method",
    "Average GMV and spend by customer age segment",
    "Peak order hours with the worst P90 delivery delays",
    "Top 5 restaurants with the highest customer ratings and order volume",
    "Weekend vs weekday order volume and average order value (AOV)",
    "Customer income band breakdown by top ordered cuisines",
    "Most common key issues mentioned in negative reviews"
]

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

SCHEMA = """
Snowflake Database: ZOMATO (Default Schema: MARTS)
Use bare table names for MARTS tables (e.g., FACT_ORDERS, DIM_RESTAURANTS). Use ZOMATO.AI.REVIEW_ENRICHED for review data.

TABLE DEFINITIONS:

1. FACT_ORDERS
   - Primary order transactions table.
   - Columns: order_id, order_timestamp, order_date, customer_id, restaurant_id, city, cuisine, payment_method, order_status, is_delivered (BOOLEAN), items_count, sales_qty, subtotal, discount, delivery_fee, gst, sales_amount, customer_rating, delivery_time_min

2. FACT_ORDER_ITEMS
   - Item-level breakdown of orders.
   - Columns: order_item_id, order_id, restaurant_id, f_id, order_ts, order_date, city, price, quantity, line_amount

3. DIM_RESTAURANTS
   - Restaurant dimension.
   - Columns: restaurant_id, restaurant_name, city, cuisine, rating, rating_count, cost_for_two

4. DIM_CUSTOMER
   - Customer demographic profiles.
   - Columns: customer_id, customer_name, email, age, age_segment (e.g. 18-24, 25-34), gender, marital_status, occupation, income_band, education, family_size

5. DIM_DATE
   - Calendar date dimension.
   - Columns: date_day, year, month, month_name, day_name, is_weekend (BOOLEAN)

6. MART_DAILY_CITY_REVENUE (Pre-aggregated daily city metrics)
   - Columns: order_date, city, orders, delivered_orders, cancel_rate, gmv, aov

7. MART_RESTAURANT_PERFORMANCE (Pre-aggregated restaurant metrics)
   - Columns: restaurant_id, restaurant_name, city, cuisine, orders, revenue, avg_customer_rating, avg_delivery_min

8. MART_DELIVERY_SLA (Pre-aggregated delivery SLA metrics)
   - Columns: city, order_hour (0-23), delivered_orders, p50 (median delivery min), p90 (90th percentile delivery min)

9. MART_REVIEW_INSIGHTS (Pre-aggregated review sentiment & topic metrics by city)
   - Columns: city, topic, sentiment_label, reviews, avg_sentiment_score, avg_star_rating, flagged_issues

10. ZOMATO.AI.REVIEW_ENRICHED (Per-review AI enriched sentiment classifications)
   - Columns: review_id, sentiment_label (positive/negative/neutral), sentiment_score (-1.0 to 1.0), topic (food quality, delivery, pricing, service, packaging, other), key_issue, model, enriched_at


BUSINESS LOGIC & METRIC DEFINITIONS:
- GMV (Gross Merchandise Value): Total delivered revenue. In FACT_ORDERS: SUM(sales_amount) WHERE is_delivered = TRUE. In MART_DAILY_CITY_REVENUE: gmv.
- AOV (Average Order Value): GMV / delivered_orders.
- Cancel Rate: Proportion of cancelled orders (cancel_rate in mart tables, or COUNT(CASE WHEN order_status = 'Cancelled' THEN 1 END)/COUNT(*) in FACT_ORDERS).
- Delivery Time: delivery_time_min in FACT_ORDERS, or p50/p90 in MART_DELIVERY_SLA.
- Weekend Analysis: Join with DIM_DATE on order_date = date_day and filter/group by is_weekend.
- Prefer MART_ tables when answering aggregated high-level questions for optimal performance.
"""

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert for Zomato. Write ONE clean, efficient SELECT query that accurately answers the user's question.

Rules:
- SELECT queries only, never modify data.
- Use bare table names for MARTS tables (FACT_ORDERS, DIM_CUSTOMER, etc.), use ZOMATO.AI.REVIEW_ENRICHED for reviews.
- Add an appropriate ORDER BY and LIMIT (default 100 or less, unless a single scalar total is asked).
- Use uppercase SQL keywords (SELECT, FROM, WHERE, GROUP BY, ORDER BY).
- Reply strictly as JSON in this exact format: {{"sql": "your query here"}}

{SCHEMA}
"""

@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="MARTS",
        role="DBT_ROLE"
    )

def generate_sql(question):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    sql = json.loads(answer)["sql"]

    sql = sql.replace("ZOMATO.MARTS.", "").replace("MARTS.", "")
    return sql.strip().rstrip(";")

def is_safe(sql):
    lowered = sql.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False

    for word in FORBIDDEN_WORDS:
        if word in lowered:
            return False

    return True

def run_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    return cursor.execute(sql).fetch_pandas_all()

st.title("Chat with your Zomato Data")
st.caption(f"Ask in English, {MODEL} writes the SQL, Snowflake runs it")

with st.sidebar:
    st.header("Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["text_sql_query"] = q

default_query = st.session_state.get("text_sql_query", "")
question = st.text_input(
    "Enter your question here", 
    value=default_query,
    placeholder="e.g. Top 10 restaurants by revenue in Bangalore"
)

if question:
    sql = generate_sql(question)
    st.code(sql, language="sql")

    if not is_safe(sql):
        st.error("The generated SQL is not safe to run. Please modify your question.")
    else:
        try:
            df = run_query(sql)
            st.success(f"{len(df)} rows returned")
            st.dataframe(df, hide_index=True, use_container_width=True)

            if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                st.bar_chart(df, x=df.columns[0], y=df.columns[1])

        except Exception as e:
            st.error(f"Error running query: {e}")