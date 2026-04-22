"""
🎬 Movie Recommender — Streamlit Frontend
Separate from submission files. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time
import json

from llm import get_recommendation, MOVIES_DF

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🍿",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid #0f3460;
    }
    .movie-title {
        color: #e94560;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .movie-meta {
        color: #a8a8b3;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .movie-desc {
        color: #eaeaea;
        font-size: 16px;
        line-height: 1.6;
    }
    .rating-badge {
        background: #e94560;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .genre-tag {
        background: #0f3460;
        color: #53d8fb;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    /* All buttons — default red style */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #c23152);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #c23152, #e94560);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# 🎬 Movie Recommender")
st.markdown("*Tell me what you're in the mood for, and I'll find your next favorite film.*")
st.markdown("---")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []
if "history_ids" not in st.session_state:
    st.session_state.history_ids = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ---------------------------------------------------------------------------
# Sidebar — Watch History
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📋 Watch History")
    st.markdown("*Movies you've already seen (won't be recommended)*")

    # Quick add from database
    all_titles = sorted(MOVIES_DF["title"].tolist())
    selected = st.multiselect(
        "Add movies you've watched:",
        options=all_titles,
        default=st.session_state.history,
        key="history_select",
    )

    # Update session state
    st.session_state.history = selected
    st.session_state.history_ids = []
    for title in selected:
        row = MOVIES_DF[MOVIES_DF["title"] == title]
        if not row.empty:
            st.session_state.history_ids.append(int(row.iloc[0]["tmdb_id"]))

    if st.session_state.history:
        st.markdown(f"**{len(st.session_state.history)}** movies in history")
        
    mood_cols = st.columns(2)
    moods = [
        ("😂 Funny", "I want something funny and lighthearted"),
        ("😱 Scary", "I want to be genuinely scared"),
        ("💕 Romance", "A beautiful love story"),
        ("🦸 Superhero", "Action movies with superheroes"),
        ("🧠 Mind-bending", "Something mind-bending that makes you think"),
        ("👨‍👩‍👧 Family", "Something fun to watch with kids"),
        ("🔪 Thriller", "An intense thriller with plot twists"),
        ("🎭 Drama", "A powerful emotional drama"),
    ]

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------

col1, col2 = st.columns([4, 1])
with col1:
    preferences = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g., 'A mind-bending sci-fi like Inception' or 'something funny for date night'",
        key="pref_input",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("# Go", use_container_width=True)

# Quick mood buttons
st.markdown("**Or pick a mood:**")
mood_cols = st.columns(4)
mood_clicked = None
for i, (label, prompt) in enumerate(moods):
    with mood_cols[i % 4]:
        if st.button(label, key=f"mood_{i}", use_container_width=True):
            mood_clicked = prompt

# Determine what to search for
search_query = mood_clicked or (preferences if recommend_btn else None)

# ---------------------------------------------------------------------------
# Get recommendation
# ---------------------------------------------------------------------------

if search_query:
    with st.spinner("🍿 Finding the perfect movie for you..."):
        start = time.perf_counter()
        try:
            result = get_recommendation(
                search_query,
                st.session_state.history,
                st.session_state.history_ids,
            )
            elapsed = time.perf_counter() - start

            # Look up full movie details
            movie = MOVIES_DF[MOVIES_DF["tmdb_id"] == result["tmdb_id"]]
            if not movie.empty:
                m = movie.iloc[0]

                st.markdown("---")
                st.markdown(f"### 🎯 Here's your pick for: *\"{search_query}\"*")

                # Movie card
                img_col, info_col = st.columns([1, 2])

                with img_col:
                    poster = m.get("poster_path", "")
                    if pd.notna(poster) and poster:
                        st.image(poster, use_container_width=True)

                with info_col:
                    st.markdown(f"## {m['title']} ({int(m['year'])})")

                    # Genre tags
                    genres = str(m.get("genres", "")).split(", ")
                    genre_html = " ".join(
                        f'<span class="genre-tag">{g}</span>' for g in genres if g
                    )
                    st.markdown(genre_html, unsafe_allow_html=True)

                    # Meta info
                    meta_parts = []
                    if pd.notna(m.get("director")) and m["director"]:
                        meta_parts.append(f"🎬 **{m['director']}**")
                    if pd.notna(m.get("runtime_min")) and m["runtime_min"]:
                        meta_parts.append(f"⏱️ {int(m['runtime_min'])} min")
                    if pd.notna(m.get("us_rating")) and m["us_rating"]:
                        meta_parts.append(f"📋 {m['us_rating']}")
                    if meta_parts:
                        st.markdown(" · ".join(meta_parts))

                    # Rating
                    if pd.notna(m.get("vote_average")) and m["vote_average"]:
                        stars = "⭐" * round(m["vote_average"] / 2)
                        st.markdown(
                            f'{stars} **{m["vote_average"]}/10** '
                            f'({int(m["vote_count"]):,} votes)'
                        )

                    # Cast
                    if pd.notna(m.get("top_cast")) and m["top_cast"]:
                        cast = ", ".join(c.strip() for c in m["top_cast"].split(",")[:4])
                        st.markdown(f"🎭 {cast}")

                # Description from the LLM
                st.markdown("---")
                st.markdown("### 💬 Why you'll love it")
                st.info(result["description"])

                # TMDB link
                tmdb_url = m.get("tmdb_url", "")
                if pd.notna(tmdb_url) and tmdb_url:
                    st.markdown(f"[🔗 View on TMDB]({tmdb_url})")

                st.caption(f"Found in {elapsed:.1f}s")

                # Add to history option
                if m["title"] not in st.session_state.history:
                    if st.button(f"✅ Mark '{m['title']}' as watched"):
                        st.session_state.history.append(m["title"])
                        st.session_state.history_ids.append(int(m["tmdb_id"]))
                        st.rerun()

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 13px;'>"
    "Built with ❤️ using Streamlit + Gemma 4 · "
    "Data from TMDB"
    "</div>",
    unsafe_allow_html=True,
)
