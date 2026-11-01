import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from ai.insight_format import (
        REVIEW_ANALYST_PROMPT,
        normalise_review_insight,
        render_review_insight,
    )
except ModuleNotFoundError:
    from insight_format import (
        REVIEW_ANALYST_PROMPT,
        normalise_review_insight,
        render_review_insight,
    )

st.set_page_config(
    page_title="Zomato Review Intelligence",
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
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(226, 55, 68, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
        color: white !important;
        display: flex;
        align-items: center;
        gap: 12px;
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
    .answer-summary { margin: 0; font-size: 1.06rem; font-weight: 500; }
    .insight-findings { display: grid; gap: 10px; margin-top: 16px; }
    .insight-finding { border-top: 1px solid rgba(128, 128, 128, 0.18); padding-top: 12px; }
    .insight-finding p, .insight-note, .insight-next { margin: 5px 0; }
    .finding-title { font-weight: 650; }
    .finding-label, .insight-note span, .insight-next span {
        display: inline-block; min-width: 112px; color: #7b2630; font-weight: 600;
    }
    .insight-note, .insight-next {
        border-top: 1px solid rgba(128, 128, 128, 0.18); padding-top: 12px; margin-top: 14px;
    }
    
    .review-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .review-card:hover {
        border-color: rgba(226, 55, 68, 0.4);
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
NEW_REVIEWS = 500
TOP_K = 5
TEMPERATURE = 0.2
CACHE_FILE = "review_embeddings.parquet"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df

def embed(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]

@st.cache_data()
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()
    df['embedding'] = embed(df['comment'].tolist())
    df.to_parquet(CACHE_FILE)
    return df

def find_similar_reviews(question, df, top_k=TOP_K):
    question_vector = np.array(embed([question])[0])
    matrix = np.vstack(df['embedding'].values)
    scores = np.dot(matrix, question_vector)

    df = df.copy()
    df['score'] = scores
    return df.nlargest(top_k, 'score')

def ask_llm(question, top_reviews, temperature=TEMPERATURE):
    context_parts = []
    for position, (_, row) in enumerate(top_reviews.iterrows(), start=1):
        context_parts.append(
            f"Review {position}\nCity: {row['city']}\nRating: {row['rating']} out of 5\n"
            f"Comment: {row['comment']}"
        )

    user_prompt = (
        f"Question: {question}\n\nRetrieved customer reviews begin\n"
        + "\n\n".join(context_parts)
        + "\nRetrieved customer reviews end"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REVIEW_ANALYST_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return normalise_review_insight(response.choices[0].message.content)

st.markdown("""
<div class="main-header">
    <h1>Zomato Review Intelligence</h1>
    <p>Semantic search and conversational analytics over customer reviews powered by Snowflake & Groq LLM</p>
</div>
""", unsafe_allow_html=True)

review_df = load_reviews()

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

question = st.text_input(
    "**Ask a question about customer reviews:**",
    placeholder="e.g. Top 5 cities for giving worst reviews, or what are the most common complaints about delivery?"
)

if question:
    with st.spinner("Searching semantic vectors and synthesizing insights with Groq..."):
        top_reviews = find_similar_reviews(question, review_df, top_k=TOP_K)
        insight = ask_llm(question, top_reviews, temperature=TEMPERATURE)

    st.markdown("### Insights")
    st.markdown(render_review_insight(insight), unsafe_allow_html=True)

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
