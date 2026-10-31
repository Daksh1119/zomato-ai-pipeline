import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import duckdb
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

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
DATABASE_FILE = os.path.join(_script_dir, "zomato_analytics.duckdb")

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

@st.cache_resource
def get_llm_client():
    """Create the API client only when an AI question is submitted.

    This keeps a missing optional Groq key from preventing the dashboard shell
    from rendering on a fresh Streamlit deployment.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY (or OPENAI_API_KEY) in Streamlit secrets.")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

SCHEMA = """
DuckDB database containing Zomato's compact analytics-serving layer.
Use only the listed tables and exact column names. Raw event tables are intentionally not deployed.

TABLE DEFINITIONS:
1. MART_DAILY_CITY_REVENUE(order_date, city, orders, delivered_orders, cancel_rate, gmv, aov)
2. MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine, orders, revenue, avg_customer_rating, avg_delivery_min)
3. MART_DELIVERY_SLA(city, order_hour, delivered_orders, p50, p90)
4. MART_PAYMENT_CANCELLATION(payment_method, orders, cancelled_orders, cancel_rate)
5. MART_CUSTOMER_SEGMENT(age_segment, cuisine, orders, delivered_orders, gmv, aov)
6. MART_INCOME_CUISINE(income_band, cuisine, orders, gmv)
7. MART_WEEKDAY_PERFORMANCE(day_type, orders, delivered_orders, gmv, aov)
8. MART_REVIEW_INSIGHTS(city, topic, sentiment_label, reviews, avg_sentiment_score, flagged_issues)
9. DIM_RESTAURANTS(restaurant_id, restaurant_name, city, cuisine, rating, rating_count, cost_for_two)

BUSINESS LOGIC & METRIC DEFINITIONS:
- GMV: total delivered revenue; all gmv columns are already aggregated.
- AOV: GMV / delivered_orders.
- Cancel Rate: use MART_PAYMENT_CANCELLATION for payment-method analysis.
- Delivery Time: p50/p90 in MART_DELIVERY_SLA or avg_delivery_min in MART_RESTAURANT_PERFORMANCE.
- Weekend Analysis: MART_WEEKDAY_PERFORMANCE already provides Weekend and Weekday rows.
- To find cuisine revenue, SUM(revenue) from MART_RESTAURANT_PERFORMANCE grouped by cuisine.
"""

SYSTEM_PROMPT = f"""
You are the senior analytics engineer for Zomato's executive self-service analytics product. Convert each business question into ONE production-quality DuckDB SELECT query.

Your decision process:
1. Identify the business metric, requested grain, comparison, ranking, and time scope.
2. Select the narrowest serving-layer table that has the required metric at the requested grain. Do not invent raw tables, columns, joins, or data that are not in the schema.
3. Preserve metric semantics. GMV is delivered revenue, AOV is GMV divided by delivered orders, and cancellation rate is cancelled orders divided by all orders. When rolling up pre-aggregated rows, SUM additive metrics. Never average an already-aggregated AOV or cancellation rate when a weighted calculation is possible.
4. Use clear, business-friendly aliases such as total_gmv, delivered_orders, weighted_aov, or cancellation_rate. Aggregate before ranking and order descending for "top", "highest", "worst", or "longest" unless the user explicitly requests otherwise.
5. Return a concise, decision-useful result: include an ORDER BY and LIMIT 100 or less for multi-row answers. A scalar answer does not need a LIMIT.

DuckDB and safety rules:
- Produce exactly one read-only SELECT statement; a WITH clause is allowed.
- Use only the supplied tables and exact column names, with bare uppercase table names.
- Use DuckDB SQL syntax, never Snowflake-specific database or schema prefixes.
- Avoid SELECT *, data-modifying statements, comments, semicolons inside the query, unsupported functions, and unnecessary joins.
- If the requested analysis cannot be answered from the serving layer, return the closest valid query that exposes the available evidence; never fabricate a query against a nonexistent table.
- Use uppercase SQL keywords (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT).

Output contract:
- Reply with JSON only: {{"sql": "your query here"}}
- Do not include markdown, explanations, confidence notes, or extra JSON keys.

{SCHEMA}
"""

REVIEW_ANALYST_PROMPT = """
You are a senior Voice-of-Customer analyst for Zomato. Produce an evidence-grounded, executive-ready answer using only the retrieved customer reviews supplied in the user message.

Evidence standards:
- Treat the retrieved reviews as a small relevance-ranked sample, not as the complete customer population. Never claim platform-wide prevalence, causation, or statistical significance from this sample.
- Separate direct evidence from interpretation. Quote or precisely paraphrase only what the supplied reviews support, and do not invent restaurants, cities, incidents, counts, trends, or customer motives.
- Reconcile mixed evidence. If sentiment or themes conflict, state that nuance rather than forcing a single conclusion.
- If the evidence is insufficient, say exactly what is missing and give the most useful next question or data cut. Do not fill gaps with general knowledge.

Response quality:
- Lead with a one- or two-sentence executive takeaway that directly answers the question.
- Follow with 2–4 concise findings. For each finding, identify the theme, the supporting review signal, and its customer or operational implication.
- When appropriate, end with one clearly-labelled recommended action that is proportional to the evidence. Phrase it as a hypothesis to validate, not a proven fact.
- Use professional, neutral language and readable Markdown bullets. Keep the response focused; avoid repeating the retrieved reviews verbatim or exposing internal retrieval scores.
"""

def get_database_connection():
    if not os.path.exists(DATABASE_FILE):
        raise FileNotFoundError(
            "The local analytics database is missing. Run "
            "`python ai/build_analytics_db.py` before starting the app."
        )
    return duckdb.connect(DATABASE_FILE, read_only=True)

def read_reviews_from_duckdb():
    conn = get_database_connection()
    df = conn.execute("""
        SELECT review_id, city, rating, comment
        FROM reviews
        ORDER BY review_id
    """).fetchdf()
    conn.close()
    df.columns = [col.lower() for col in df.columns]
    return df, len(df)

@st.cache_resource
def load_reviews():
    df, total = read_reviews_from_duckdb()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df["comment"].fillna(""))
    return df, vectorizer, matrix, total

def find_similar_reviews(question, df, vectorizer, matrix, top_k=TOP_K):
    question_vector = vectorizer.transform([question])
    scores = (matrix @ question_vector.T).toarray().ravel()
    top_k = min(top_k, len(df))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    result = df.iloc[top_indices].copy()
    result['score'] = scores[top_indices]
    return result

def ask_rag_llm(question, top_reviews, temperature=TEMPERATURE):
    context = ""
    for _, row in top_reviews.iterrows():
        context += f"• [{row['city']} | {row['rating']}★]: {row['comment']}\n"

    user_prompt = f"Question: {question}\n\nCustomer Reviews Context:\n{context}"

    response = get_llm_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": REVIEW_ANALYST_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def generate_sql(question):
    response = get_llm_client().chat.completions.create(
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
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.ParserException:
        return False
    return len(statements) == 1 and str(statements[0].type) == "StatementType.SELECT"

def run_sql_query(sql):
    conn = get_database_connection()
    df = conn.execute(sql).fetchdf()
    conn.close()
    return df

# -----------------------------------------------------------------------------
# Main Header & Tab Navigation
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Zomato Reviews Dashboard</h1>
    <p>Conversational review intelligence (RAG) and Text-to-SQL DuckDB analytics</p>
</div>
""", unsafe_allow_html=True)

nav_tabs = st.tabs(["Review Intelligence (RAG)", "SQL Data Analyst (Text-to-SQL)"])

# -----------------------------------------------------------------------------
# Tab 1: Review Intelligence (RAG)
# -----------------------------------------------------------------------------
with nav_tabs[0]:
    if "reviews_ready" not in st.session_state:
        st.session_state.reviews_ready = False

    if not st.session_state.reviews_ready:
        try:
            with st.spinner("Preparing review intelligence…"):
                review_df, review_vectorizer, review_matrix, total_reviews = load_reviews()
            st.session_state.review_df = review_df
            st.session_state.review_vectorizer = review_vectorizer
            st.session_state.review_matrix = review_matrix
            st.session_state.total_reviews = total_reviews
            st.session_state.reviews_ready = True
        except Exception as exc:
            st.error(
                "Review data could not be loaded. Ensure the local DuckDB database "
                "is deployed with the app, then refresh the page."
            )
            st.exception(exc)

    if st.session_state.reviews_ready:
        review_df = st.session_state.review_df
        review_vectorizer = st.session_state.review_vectorizer
        review_matrix = st.session_state.review_matrix
        total_reviews = st.session_state.total_reviews

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(review_df):,}</div>
                <div class="metric-label">Reviews Indexed &nbsp;|&nbsp; Full Corpus</div>
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
            st.markdown("""
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

    if st.session_state.reviews_ready and question_rag:
        with st.spinner("Retrieving relevant reviews and synthesizing insights with Groq..."):
            top_reviews = find_similar_reviews(
                question_rag, review_df, review_vectorizer, review_matrix, top_k=TOP_K
            )
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
def select_example_query(question):
    """Populate the SQL input before Streamlit reruns after an example tap."""
    st.session_state["sql_text_input"] = question


with nav_tabs[1]:
    st.markdown("### **Conversational DuckDB SQL Analytics**")
    st.caption("Ask questions in natural language. Groq generates DuckDB SQL, runs it locally, and displays data and charts.")

    st.markdown("#### **Example Queries**")
    ex_cols = st.columns(2)
    for i, q_text in enumerate(EXAMPLE_QUESTIONS[:6]):
        with ex_cols[i % 2]:
            st.button(
                q_text,
                key=f"sql_btn_{i}",
                use_container_width=True,
                on_click=select_example_query,
                args=(q_text,)
            )

    question_sql = st.text_input(
        "**Enter your business question:**",
        placeholder="e.g. Top 10 cities by GMV, or Cancel rate by payment method",
        key="sql_text_input"
    )

    if question_sql:
        with st.spinner("Generating DuckDB SQL with Groq..."):
            sql = generate_sql(question_sql)
        
        st.markdown("##### **Generated SQL Query**")
        st.code(sql, language="sql")

        if not is_safe_sql(sql):
            st.error("The generated SQL is not safe to run (data modifications are prohibited).")
        else:
            try:
                with st.spinner("Running query on DuckDB..."):
                    df_sql = run_sql_query(sql)
                st.success(f"Returned {len(df_sql)} rows")
                st.dataframe(df_sql, hide_index=True, use_container_width=True)

                if len(df_sql.columns) == 2 and pd.api.types.is_numeric_dtype(df_sql.iloc[:, 1]):
                    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
                    st.markdown("#### Visual summary")
                    st.bar_chart(df_sql, x=df_sql.columns[0], y=df_sql.columns[1])

            except Exception as e:
                st.error(f"Error running DuckDB query: {e}")
