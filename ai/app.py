import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Zomato Reviews Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #E23744 0%, #FF645A 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(226, 55, 68, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
        color: white !important;
    }
    .main-header p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        opacity: 0.92;
        color: #fff;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 55, 68, 0.18);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(226, 55, 68, 0.12);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E23744;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }
    
    .answer-card {
        background: rgba(226, 55, 68, 0.04);
        border-left: 4px solid #E23744;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0 24px 0;
        font-size: 1.05rem;
        line-height: 1.65;
    }
    
    .review-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .badge-city {
        background-color: #E23744;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-score {
        background: rgba(40, 167, 69, 0.15);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.3);
        padding: 3px 9px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

CHAT_MODEL = "openai/gpt-oss-20b"
TOP_K = 5
TEMPERATURE = 0.2
_script_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_script_dir, "review_embeddings.parquet")
MATRIX_FILE = os.path.join(_script_dir, "review_matrix.npy")

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
1. FACT_ORDERS(order_id, order_timestamp, order_date, customer_id, restaurant_id, city, cuisine, payment_method, order_status, is_delivered, items_count, sales_qty, subtotal, discount, delivery_fee, gst, sales_amount, customer_rating, delivery_time_min)
2. FACT_ORDER_ITEMS(order_item_id, order_id, restaurant_id, f_id, order_ts, order_date, city, price, quantity, line_amount)
3. DIM_RESTAURANTS(restaurant_id, restaurant_name, city, cuisine, rating, rating_count, cost_for_two)
4. DIM_CUSTOMER(customer_id, customer_name, email, age, age_segment, gender, marital_status, occupation, income_band, education, family_size)
5. DIM_DATE(date_day, year, month, month_name, day_name, is_weekend)
6. MART_DAILY_CITY_REVENUE(order_date, city, orders, delivered_orders, cancel_rate, gmv, aov)
7. MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine, orders, revenue, avg_customer_rating, avg_delivery_min)
8. MART_DELIVERY_SLA(city, order_hour, delivered_orders, p50, p90)
9. MART_REVIEW_INSIGHTS(city, topic, sentiment_label, reviews, avg_sentiment_score, avg_star_rating, flagged_issues)
10. ZOMATO.AI.REVIEW_ENRICHED(review_id, sentiment_label, sentiment_score, topic, key_issue, model, enriched_at)

BUSINESS LOGIC & METRIC DEFINITIONS:
- GMV: Total delivered revenue (MART_DAILY_CITY_REVENUE.gmv or SUM(sales_amount) WHERE is_delivered = TRUE in FACT_ORDERS).
- AOV: GMV / delivered_orders.
- Cancel Rate: Proportion of cancelled orders.
- Delivery Time: delivery_time_min or p50/p90 in MART_DELIVERY_SLA.
- Weekend Analysis: Join with DIM_DATE on order_date = date_day and filter/group by is_weekend.
- Prefer MART_ tables when answering aggregated questions.
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
def get_snowflake_connection(schema="MARTS"):
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=schema,
        role=os.getenv("SNOWFLAKE_ROLE", "DBT_ROLE")
    )

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_reviews_from_snowflake():
    conn = get_snowflake_connection(schema="RAW")
    query = """
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()
    df.columns = [col.lower() for col in df.columns]
    return df

def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)

def embed(texts):
    return embed_texts(texts).tolist()

@st.cache_resource
def load_reviews():
    if os.path.exists(CACHE_FILE) and os.path.exists(MATRIX_FILE):
        df = pd.read_parquet(CACHE_FILE)
        matrix = np.load(MATRIX_FILE)
        return df, matrix

    df = read_reviews_from_snowflake()
    matrix = embed_texts(df['comment'].tolist())
    np.save(MATRIX_FILE, matrix)
    df_meta = df[['review_id', 'city', 'rating', 'comment']]
    df_meta.to_parquet(CACHE_FILE, index=False)
    return df_meta, matrix

def find_similar_reviews(question, df, matrix, top_k=TOP_K):
    question_vector = embed_texts([question])[0]
    scores = np.dot(matrix, question_vector)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    result = df.iloc[top_indices].copy()
    result['score'] = scores[top_indices]
    return result

def ask_rag_llm(question, top_reviews, temperature=TEMPERATURE):
    context = ""
    for _, row in top_reviews.iterrows():
        context += f"• [{row['city']} | {row['rating']}★]: {row['comment']}\n"

    system_prompt = (
        "You are an AI data analyst for Zomato. "
        "Answer the user's question accurately and ONLY based on the provided customer reviews. "
        "Be concise, clear, and structured. Use bullet points where appropriate. "
        "If the reviews don't contain enough information to answer, explicitly state that."
    )
    user_prompt = f"Question: {question}\n\nCustomer Reviews Context:\n{context}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def generate_sql(question):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
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

def is_safe_sql(sql):
    lowered = sql.lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False
    for word in FORBIDDEN_WORDS:
        if word in lowered:
            return False
    return True

def run_sql_query(sql):
    conn = get_snowflake_connection(schema="MARTS")
    cursor = conn.cursor()
    return cursor.execute(sql).fetch_pandas_all()

# -----------------------------------------------------------------------------
# Main Header & Tab Navigation
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Zomato Reviews Dashboard</h1>
    <p>Conversational review intelligence (RAG) and Text-to-SQL Snowflake analytics</p>
</div>
""", unsafe_allow_html=True)

nav_tabs = st.tabs(["Review Intelligence (RAG)", "SQL Data Analyst (Text-to-SQL)"])

# -----------------------------------------------------------------------------
# Tab 1: Review Intelligence (RAG)
# -----------------------------------------------------------------------------
with nav_tabs[0]:
    review_df, review_matrix = load_reviews()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(review_df):,}</div>
            <div class="metric-label">Analyzed Reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_rating = review_df['rating'].mean() if 'rating' in review_df.columns else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">★ {avg_rating:.2f}</div>
            <div class="metric-label">Avg Rating</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        num_cities = review_df['city'].nunique() if 'city' in review_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_cities}</div>
            <div class="metric-label">Cities Covered</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">Groq</div>
            <div class="metric-label">Ultra-Fast Inference</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    question_rag = st.text_input(
        "**Ask a question about customer reviews:**",
        placeholder="e.g. Top 5 cities for giving worst reviews, or what are the most common complaints about delivery?",
        key="rag_input"
    )

    if question_rag:
        with st.spinner("Searching semantic vectors and synthesizing insights with Groq..."):
            top_reviews = find_similar_reviews(question_rag, review_df, review_matrix, top_k=TOP_K)
            answer = ask_rag_llm(question_rag, top_reviews, temperature=TEMPERATURE)

        st.markdown("### **AI Insights**")
        st.markdown(f"""
        <div class="answer-card">
            {answer}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### **Top {TOP_K} Retrieved Context Reviews**")
        
        tab_cards, tab_table = st.tabs(["Visual Cards", "Data Table"])
        
        with tab_cards:
            for idx, row in top_reviews.iterrows():
                rating_stars = "★" * int(row.get('rating', 0)) + "☆" * (5 - int(row.get('rating', 0)))
                sim_score = row.get('score', 0.0) * 100
                
                st.markdown(f"""
                <div class="review-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span class="badge-city">{row.get('city', 'Unknown')}</span>
                            <span style="color: #f5a623; font-weight: 700; margin-left: 10px; font-size: 1.05rem;">{rating_stars}</span>
                            <span style="opacity: 0.7; font-size: 0.85rem; margin-left: 5px;">({row.get('rating')} stars)</span>
                        </div>
                        <span class="badge-score">Match: {sim_score:.1f}%</span>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.5; opacity: 0.92;">
                        "{row.get('comment', '')}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_table:
            display_df = top_reviews[['city', 'rating', 'comment', 'score']].copy()
            display_df['similarity'] = (display_df['score'] * 100).map("{:.1f}%".format)
            st.dataframe(
                display_df[['city', 'rating', 'similarity', 'comment']],
                column_config={
                    "city": st.column_config.TextColumn("City"),
                    "rating": st.column_config.NumberColumn("Rating ★", format="%d ★"),
                    "similarity": st.column_config.TextColumn("Similarity"),
                    "comment": st.column_config.TextColumn("Customer Review", width="large")
                },
                hide_index=True,
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# Tab 2: SQL Data Analyst (Text-to-SQL)
# -----------------------------------------------------------------------------
with nav_tabs[1]:
    st.markdown("### **Conversational Snowflake SQL Analytics**")
    st.caption("Ask questions in natural language. Groq LLM generates Snowflake SQL, runs queries, and displays data & charts.")

    st.markdown("#### **Example Queries**")
    ex_cols = st.columns(2)
    for i, q_text in enumerate(EXAMPLE_QUESTIONS[:6]):
        with ex_cols[i % 2]:
            if st.button(q_text, key=f"sql_btn_{i}", use_container_width=True):
                st.session_state["sql_question_val"] = q_text

    sql_default = st.session_state.get("sql_question_val", "")
    question_sql = st.text_input(
        "**Enter your business question:**",
        value=sql_default,
        placeholder="e.g. Top 10 cities by GMV, or Cancel rate by payment method",
        key="sql_text_input"
    )

    if question_sql:
        with st.spinner("Generating Snowflake SQL with Groq..."):
            sql = generate_sql(question_sql)
        
        st.markdown("##### **Generated SQL Query**")
        st.code(sql, language="sql")

        if not is_safe_sql(sql):
            st.error("The generated SQL is not safe to run (data modifications are prohibited).")
        else:
            try:
                with st.spinner("Running query on Snowflake..."):
                    df_sql = run_sql_query(sql)
                st.success(f"Returned {len(df_sql)} rows")
                st.dataframe(df_sql, hide_index=True, use_container_width=True)

                if len(df_sql.columns) == 2 and pd.api.types.is_numeric_dtype(df_sql.iloc[:, 1]):
                    st.bar_chart(df_sql, x=df_sql.columns[0], y=df_sql.columns[1])

            except Exception as e:
                st.error(f"Error running Snowflake query: {e}")
