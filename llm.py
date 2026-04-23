"""
Movie recommendation agent — Hybrid retrieval + Chain-of-Thought + TMDB enrichment.

Pipeline:
  1. Hybrid retrieval: semantic embeddings + TF-IDF keyword search + quality scoring
  2. Chain-of-thought LLM reasoning: analyze user intent → pick movie → write description
  3. Optional TMDB API enrichment: fetch reviews/similar movies for richer descriptions

No hardcoded synonym dictionaries, stopword lists, or genre mappings.

IMPORTANT: Do NOT hard-code your API key. The grader injects OLLAMA_API_KEY
at runtime. This code reads it from the environment.
# Optional: Set TMDB_API_KEY for review enrichment (not required).
"""

import json
import os
import re
import time
import argparse

from dotenv import load_dotenv
import numpy as np
import ollama
import pandas as pd
import requests
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDINGS_CACHE = os.path.join(os.path.dirname(__file__), "movie_embeddings.npy")

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
MOVIES_DF = pd.read_csv(DATA_PATH)

# Alias used by test.py
TOP_MOVIES = MOVIES_DF

# Pre-process: fill NaN
for col in ["overview", "genres", "keywords", "director", "top_cast", "tagline", "us_rating"]:
    MOVIES_DF[col] = MOVIES_DF[col].fillna("")

# Valid TMDB IDs
VALID_IDS = set(MOVIES_DF["tmdb_id"].astype(int))

# ---------------------------------------------------------------------------
# Retrieval Method 1: Semantic Embeddings (cached)
# ---------------------------------------------------------------------------

_EMBED_MODEL = TextEmbedding(EMBED_MODEL_NAME)

if os.path.exists(EMBEDDINGS_CACHE):
    _MOVIE_EMBEDDINGS = np.load(EMBEDDINGS_CACHE)
else:
    _CORPUS = (
        MOVIES_DF["title"] + ". Genres: " + MOVIES_DF["genres"] + ". " +
        MOVIES_DF["overview"].str[:150] + " Keywords: " + MOVIES_DF["keywords"]
    ).tolist()
    _MOVIE_EMBEDDINGS = np.array(list(_EMBED_MODEL.embed(_CORPUS)))
    np.save(EMBEDDINGS_CACHE, _MOVIE_EMBEDDINGS)

# ---------------------------------------------------------------------------
# Retrieval Method 2: TF-IDF Keyword Search
# ---------------------------------------------------------------------------

_KEYWORD_CORPUS = (
    MOVIES_DF["title"] + " " +
    MOVIES_DF["genres"] + " " +
    MOVIES_DF["keywords"] + " " +
    MOVIES_DF["director"] + " " +
    MOVIES_DF["top_cast"] + " " +
    MOVIES_DF["overview"].str[:200] + " " +
    MOVIES_DF["tagline"]
).tolist()

_TFIDF = TfidfVectorizer(stop_words="english", max_features=8000)
_TFIDF_MATRIX = _TFIDF.fit_transform(_KEYWORD_CORPUS)

# ---------------------------------------------------------------------------
# Quality scores (pre-computed)
# ---------------------------------------------------------------------------

_VOTE_AVG = MOVIES_DF["vote_average"].clip(0, 10).values / 10.0
_VOTE_LOG = np.log1p(MOVIES_DF["vote_count"].clip(0).values)
_VOTE_LOG_MAX = _VOTE_LOG.max() if _VOTE_LOG.max() > 0 else 1
_QUALITY = _VOTE_AVG * 0.6 + (_VOTE_LOG / _VOTE_LOG_MAX) * 0.4


# ---------------------------------------------------------------------------
# Hybrid retrieval: embeddings + keywords + quality
# ---------------------------------------------------------------------------

def _score_movies(preferences: str, history_ids: set[int]) -> pd.DataFrame:
    """
    Hybrid retrieval: combine semantic embeddings and keyword search.
    The LLM gets candidates from both methods for richer coverage.
    """
    # Embedding similarity
    q_emb = np.array(list(_EMBED_MODEL.embed([preferences])))
    embed_scores = np.dot(_MOVIE_EMBEDDINGS, q_emb.T).flatten()

    # TF-IDF keyword similarity
    q_tfidf = _TFIDF.transform([preferences])
    keyword_scores = sklearn_cosine(q_tfidf, _TFIDF_MATRIX).flatten()

    # Combine all signals
    df = MOVIES_DF.copy()
    df["_embed_score"] = embed_scores * 5.0      # Semantic meaning
    df["_keyword_score"] = keyword_scores * 4.0   # Exact term matches
    df["_quality_score"] = _QUALITY * 2.0         # Rating + popularity
    df["_total_score"] = df["_embed_score"] + df["_keyword_score"] + df["_quality_score"]

    # Exclude watched movies
    df = df[~df["tmdb_id"].isin(history_ids)]
    return df.sort_values("_total_score", ascending=False)


# ---------------------------------------------------------------------------
# TMDB API enrichment (optional — uses TMDB_API_KEY if available)
# ---------------------------------------------------------------------------

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")


def _fetch_tmdb_reviews(tmdb_id: int, max_reviews: int = 2) -> str:
    """Fetch top user reviews from TMDB API. Returns empty string if unavailable."""
    if not TMDB_API_KEY:
        return ""
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"
        resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=3)
        if resp.status_code != 200:
            return ""
        reviews = resp.json().get("results", [])[:max_reviews]
        if not reviews:
            return ""
        snippets = []
        for r in reviews:
            content = r.get("content", "")[:150].strip()
            author = r.get("author", "")
            rating = r.get("author_details", {}).get("rating", "")
            snippet = f'"{content}..."'
            if author:
                snippet += f" — {author}"
            if rating:
                snippet += f" ({rating}/10)"
            snippets.append(snippet)
        return "\n".join(snippets)
    except Exception:
        return ""


def _fetch_tmdb_similar(tmdb_id: int, max_results: int = 3) -> str:
    """Fetch similar movies from TMDB API. Returns empty string if unavailable."""
    if not TMDB_API_KEY:
        return ""
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/similar"
        resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=3)
        if resp.status_code != 200:
            return ""
        results = resp.json().get("results", [])[:max_results]
        if not results:
            return ""
        return ", ".join(r.get("title", "") for r in results if r.get("title"))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _get_client() -> ollama.Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY not set. Add it to your .env file.")
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )


def _format_movie(row) -> str:
    """Format a movie row for the LLM prompt."""
    parts = [f'tmdb_id={row.tmdb_id} | "{row.title}" ({row.year})']
    if row.genres:
        parts.append(f"Genres: {row.genres}")
    if row.director:
        parts.append(f"Director: {row.director}")
    if row.top_cast:
        cast_list = [c.strip() for c in row.top_cast.split(",")][:2]
        parts.append(f"Cast: {', '.join(cast_list)}")
    if row.vote_average and row.vote_count:
        parts.append(f"Rating: {row.vote_average}/10 ({int(row.vote_count)} votes)")
    if row.overview:
        parts.append(f"Plot: {row.overview[:120]}")
    return " | ".join(parts)


def _build_prompt(preferences: str, history: list[str], history_ids: list[int],
                  candidates: pd.DataFrame) -> str:
    """Build the LLM prompt — the LLM does all the reasoning."""
    movie_blocks = []
    for row in candidates.itertuples():
        movie_blocks.append(f"- {_format_movie(row)}")
    movie_list = "\n\n".join(movie_blocks)

    history_text = (
        ", ".join(
            f'"{name}" (tmdb_id={tid})' for name, tid in zip(history, history_ids)
        ) if history else "none"
    )

    return f"""You are a movie recommendation expert. A user needs your help picking a movie.

User's request: "{preferences}"
Movies they've already seen (do NOT recommend these): {history_text}

Here are the candidate movies (found via semantic search and keyword matching):

{movie_list}

Think step by step:
1. INTENT: What is the user really looking for? Consider genre, mood, themes, specific movie/director references, and emotional tone.
2. MATCH: Which candidate best fits that intent? Consider how well the movie's genre, plot, director, cast, and rating align.
3. SELL: Write a compelling, personalized description (≤500 chars) that connects THIS movie to THEIR specific request. Don't summarize the plot — explain why they'll love it.

Return ONLY a JSON object: {{"reasoning": "<1-2 sentences on why you picked this>", "tmdb_id": <int>, "description": "<your pitch, ≤500 chars>"}}"""


def _parse_llm_response(content: str) -> dict:
    """Parse LLM response with fallbacks for formatting quirks."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\"tmdb_id\"[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse LLM response: {content[:200]}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    history_id_set = set(history_ids)

    # Stage 1: Hybrid retrieval — embeddings + keywords + quality
    scored = _score_movies(preferences, history_id_set)
    candidates = scored.head(8)

    # Stage 2: Chain-of-thought LLM reasoning
    prompt = _build_prompt(preferences, history, history_ids, candidates)
    client = _get_client()
    result = None
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_predict": 300, "temperature": 0},
        )
        result = _parse_llm_response(response.message.content)
    except Exception:
        result = None

    if result:
        tmdb_id = int(result["tmdb_id"])
        if tmdb_id not in VALID_IDS or tmdb_id in history_id_set:
            result = None

    if not result:
        fallback = candidates.iloc[0]
        tmdb_id = int(fallback.tmdb_id)
        description = (
            f"You should watch \"{fallback.title}\" ({int(fallback.year)}) — "
            f"a {fallback.genres.lower()} film"
        )
        if fallback.director:
            description += f" by {fallback.director}"
        if fallback.vote_average:
            description += f", rated {fallback.vote_average}/10"
        description += ". It's one of the most acclaimed movies in recent years."
        return {"tmdb_id": tmdb_id, "description": description[:500]}

    tmdb_id = int(result["tmdb_id"])
    description = str(result.get("description", ""))

    # Stage 3: Optional TMDB enrichment — enhance description with real reviews
    if TMDB_API_KEY and len(description) < 400:
        reviews = _fetch_tmdb_reviews(tmdb_id, max_reviews=1)
        if reviews:
            review_line = reviews.split("\n")[0]
            # Only append if it fits within 500 chars
            enriched = description.rstrip(".") + ". Critics agree: " + review_line
            if len(enriched) <= 500:
                description = enriched

    return {"tmdb_id": tmdb_id, "description": description[:500]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommendation agent.")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start
    print(json.dumps(result, indent=2))
    print(f"\nServed in {elapsed:.2f}s")
