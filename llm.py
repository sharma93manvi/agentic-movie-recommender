"""
Movie recommendation agent.

Uses a two-stage approach:
  1. Score-based retrieval to shortlist candidates from ~1000 movies
  2. Single LLM call with rich movie context to pick the best match and
     write a compelling, personalized description.

IMPORTANT: Do NOT hard-code your API key. The grader injects OLLAMA_API_KEY
at runtime. This code reads it from the environment.
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

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
MOVIES_DF = pd.read_csv(DATA_PATH)

# Alias used by test.py — points to the full dataset
TOP_MOVIES = MOVIES_DF

# Pre-process: fill NaN so string operations don't blow up
for col in ["overview", "genres", "keywords", "director", "top_cast", "tagline", "us_rating"]:
    MOVIES_DF[col] = MOVIES_DF[col].fillna("")

# Build a lowercase search corpus per movie for fast keyword matching
MOVIES_DF["_search_blob"] = (
    MOVIES_DF["title"].str.lower() + " " +
    MOVIES_DF["genres"].str.lower() + " " +
    MOVIES_DF["keywords"].str.lower() + " " +
    MOVIES_DF["overview"].str.lower() + " " +
    MOVIES_DF["director"].str.lower() + " " +
    MOVIES_DF["top_cast"].str.lower() + " " +
    MOVIES_DF["tagline"].str.lower()
)

# Build title lookup for "like X" reference matching
TITLE_LOOKUP = {}
for _, row in MOVIES_DF.iterrows():
    TITLE_LOOKUP[row["title"].lower()] = row

# Valid TMDB IDs (for safety check)
VALID_IDS = set(MOVIES_DF["tmdb_id"].astype(int))

# Collect all unique genres for fuzzy matching
ALL_GENRES = set()
for g in MOVIES_DF["genres"].dropna():
    for genre in g.split(", "):
        genre = genre.strip().lower()
        if genre:
            ALL_GENRES.add(genre)

# ---------------------------------------------------------------------------
# Retrieval: score-based candidate shortlisting
# ---------------------------------------------------------------------------

# Synonyms / related terms → genre/theme keywords
GENRE_SYNONYMS = {
    # Superhero / comic book
    "superhero": ["action", "science fiction", "superhero"],
    "superheroes": ["action", "science fiction", "superhero"],
    "marvel": ["action", "science fiction", "superhero", "marvel"],
    "mcu": ["action", "science fiction", "superhero", "marvel"],
    "dc": ["action", "science fiction", "superhero", "dc"],
    "comic book": ["action", "science fiction", "superhero"],
    # Horror
    "scary": ["horror", "thriller"],
    "spooky": ["horror"],
    "creepy": ["horror", "thriller"],
    "terrifying": ["horror", "thriller"],
    "horror": ["horror"],
    "slasher": ["horror"],
    "haunted": ["horror", "mystery"],
    "zombie": ["horror", "action"],
    # Comedy
    "funny": ["comedy"],
    "hilarious": ["comedy"],
    "laugh": ["comedy"],
    "comedy": ["comedy"],
    "witty": ["comedy", "drama"],
    "lighthearted": ["comedy", "family"],
    # Romance
    "romantic": ["romance", "drama"],
    "love": ["romance", "drama"],
    "love story": ["romance"],
    "romance": ["romance"],
    # Feel-good / Family
    "feel-good": ["comedy", "family", "animation"],
    "feel good": ["comedy", "family", "animation"],
    "heartwarming": ["drama", "family", "animation"],
    "uplifting": ["drama", "comedy", "family"],
    "wholesome": ["family", "comedy", "animation"],
    # Animation
    "animated": ["animation"],
    "cartoon": ["animation"],
    "pixar": ["animation", "family", "pixar"],
    "disney": ["animation", "family", "disney"],
    "anime": ["animation"],
    "ghibli": ["animation", "fantasy"],
    # Sci-fi / Mind-bending
    "sci-fi": ["science fiction"],
    "scifi": ["science fiction"],
    "space": ["science fiction", "adventure"],
    "mind-bending": ["science fiction", "thriller", "mystery"],
    "mind bending": ["science fiction", "thriller", "mystery"],
    "cerebral": ["science fiction", "thriller", "drama"],
    "trippy": ["science fiction", "thriller", "mystery"],
    "visually stunning": ["science fiction", "fantasy", "adventure"],
    "thought-provoking": ["science fiction", "drama", "thriller"],
    "thought provoking": ["science fiction", "drama", "thriller"],
    "makes you think": ["science fiction", "thriller", "mystery"],
    # Thriller / Suspense
    "thriller": ["thriller"],
    "suspense": ["thriller", "mystery"],
    "suspenseful": ["thriller", "mystery"],
    "tense": ["thriller", "drama"],
    "intense": ["thriller", "action", "drama"],
    "twist": ["thriller", "mystery"],
    "plot twist": ["thriller", "mystery", "crime"],
    # Crime
    "crime": ["crime", "thriller"],
    "detective": ["mystery", "crime"],
    "heist": ["crime", "thriller", "heist"],
    "murder": ["crime", "thriller", "mystery"],
    "noir": ["crime", "thriller", "drama"],
    # Action / Adventure
    "action": ["action"],
    "action-packed": ["action", "adventure"],
    "adventure": ["adventure", "action"],
    "epic": ["adventure", "action", "fantasy"],
    "explosive": ["action", "thriller"],
    # Drama
    "drama": ["drama"],
    "emotional": ["drama", "romance"],
    "sad": ["drama"],
    "tearjerker": ["drama", "romance"],
    "cry": ["drama", "romance"],
    "moving": ["drama"],
    "powerful": ["drama", "history"],
    # Fantasy
    "fantasy": ["fantasy", "adventure"],
    "magical": ["fantasy", "animation"],
    "magic": ["fantasy"],
    # War / History
    "war": ["war", "action", "history"],
    "historical": ["history", "drama"],
    "history": ["history", "drama"],
    "real events": ["history", "drama", "war"],
    "true story": ["history", "drama"],
    "based on": ["history", "drama"],
    # Dark / Gritty
    "dark": ["thriller", "drama", "crime"],
    "gritty": ["crime", "drama", "thriller"],
    "violent": ["action", "crime", "thriller"],
    # Kids / Family
    "kids": ["family", "animation"],
    "children": ["family", "animation"],
    "family-friendly": ["family", "animation"],
    "family": ["family"],
    # Music
    "music": ["music"],
    "musical": ["music"],
    # Other
    "western": ["western"],
    "documentary": ["documentary"],
    "mystery": ["mystery", "thriller"],
    # Cerebral / intellectual
    "intellectual": ["science fiction", "drama", "thriller"],
    "complex": ["science fiction", "thriller", "drama"],
    "philosophical": ["science fiction", "drama"],
    "surreal": ["science fiction", "fantasy", "thriller"],
    "dream": ["science fiction", "fantasy", "thriller"],
    "reality": ["science fiction", "thriller"],
    "time travel": ["science fiction", "adventure"],
    "time": ["science fiction"],
    "psychological": ["thriller", "drama", "mystery"],
    "mindf": ["science fiction", "thriller", "mystery"],
}


def _extract_search_terms(preferences: str) -> list[str]:
    """Extract meaningful search terms from user preferences."""
    text = preferences.lower()
    stopwords = {
        "i", "me", "my", "want", "to", "watch", "see", "looking", "for",
        "something", "a", "an", "the", "that", "is", "are", "was", "were",
        "with", "and", "or", "but", "in", "of", "like", "really", "very",
        "some", "would", "love", "enjoy", "prefer", "need", "feel", "mood",
        "movie", "movies", "film", "films", "show", "shows", "think",
        "about", "it", "its", "been", "have", "has", "had", "do", "does",
        "can", "could", "should", "will", "just", "also", "too", "more",
        "not", "no", "don't", "doesn't", "didn't", "won't", "wouldn't",
        "maybe", "perhaps", "kind", "type", "sort", "bit", "lot", "lots",
        "good", "great", "best", "nice", "cool", "awesome", "amazing",
        "recommend", "recommendation", "suggest", "suggestion", "please",
        "thanks", "thank", "you", "me", "we", "us", "our", "them", "they",
        "this", "these", "those", "here", "there", "where", "when", "how",
        "what", "which", "who", "whom", "why", "so", "if", "then", "than",
        "from", "up", "down", "out", "on", "off", "over", "under", "again",
        "once", "all", "any", "both", "each", "few", "many", "much", "own",
        "same", "other", "such", "only", "into", "through", "during",
        "before", "after", "above", "below", "between", "because", "while",
        "give", "got", "get", "make", "makes",
    }
    words = re.findall(r"[a-z][a-z'-]+", text)
    terms = [w for w in words if w not in stopwords and len(w) > 1]
    return terms


def _find_referenced_movies(preferences: str) -> list[dict]:
    """Find movies mentioned by name in the preferences (for 'like X' queries)."""
    pref_lower = preferences.lower()
    referenced = []
    # Sort by title length descending to match longer titles first
    for title, row in sorted(TITLE_LOOKUP.items(), key=lambda x: -len(x[0])):
        if title in pref_lower and len(title) > 3:  # skip very short titles
            referenced.append(row)
    return referenced


def _score_movies(preferences: str, history_ids: set[int]) -> pd.DataFrame:
    """Score all movies based on preference matching. Returns sorted DataFrame."""
    df = MOVIES_DF.copy()
    df = df[~df["tmdb_id"].isin(history_ids)]

    pref_lower = preferences.lower()
    search_terms = _extract_search_terms(preferences)

    # --- Genre matching via synonyms ---
    target_genres = set()
    for term in search_terms:
        if term in GENRE_SYNONYMS:
            target_genres.update(GENRE_SYNONYMS[term])
    # Also check multi-word synonym keys against full preference text
    for key, genres in GENRE_SYNONYMS.items():
        if key in pref_lower:
            target_genres.update(genres)

    # --- Fuzzy genre matching for typos/gibberish ---
    # If no exact matches, try to detect genre hints in garbled text
    if not target_genres:
        for term in search_terms:
            # Check if synonym keys appear as substrings
            for key, genres in GENRE_SYNONYMS.items():
                if len(key) >= 5 and key in term:
                    target_genres.update(genres)
            # Check if genre names appear as substrings
            for genre in ALL_GENRES:
                if len(genre) >= 5 and genre in term:
                    target_genres.add(genre)

    # --- "Like X" reference matching ---
    # If user mentions a specific movie, boost movies with similar genres/keywords
    referenced = _find_referenced_movies(preferences)
    ref_genres = set()
    ref_keywords = set()
    for ref in referenced:
        for g in str(ref.get("genres", "")).split(", "):
            if g.strip():
                ref_genres.add(g.strip().lower())
        for kw in str(ref.get("keywords", "")).split(", "):
            if kw.strip():
                ref_keywords.add(kw.strip().lower())
        # Also add the referenced movie's director as a search signal
        director = str(ref.get("director", "")).lower()
        if director:
            search_terms.append(director)
    target_genres.update(ref_genres)

    genre_score = df["genres"].str.lower().apply(
        lambda g: sum(1 for tg in target_genres if tg in g)
    ) if target_genres else pd.Series(0, index=df.index)

    # --- Keyword / text matching ---
    text_score = df["_search_blob"].apply(
        lambda blob: sum(1 for t in search_terms if t in blob)
    )

    # --- Thematic keyword matching ---
    # Map user mood/theme words to TMDB keywords that appear in the data
    THEME_KEYWORDS = {
        "mind-bending": ["dream", "alternate reality", "time travel", "parallel universe",
                         "simulation", "consciousness", "memory", "hallucination", "twist ending"],
        "mind bending": ["dream", "alternate reality", "time travel", "parallel universe",
                         "simulation", "consciousness", "memory", "hallucination", "twist ending"],
        "visually stunning": ["visual effects", "imax", "3d", "cgi", "cinematography"],
        "makes you think": ["philosophical", "existential", "moral dilemma", "dystopia",
                            "artificial intelligence", "consciousness"],
        "cerebral": ["philosophical", "existential", "moral dilemma", "consciousness",
                     "twist ending", "nonlinear timeline"],
        "thought-provoking": ["philosophical", "existential", "moral dilemma", "dystopia",
                              "social commentary"],
        "emotional": ["loss of loved one", "grief", "family", "father son relationship",
                      "father daughter relationship", "mother son relationship"],
        "heartwarming": ["friendship", "family", "coming of age", "redemption"],
        "intense": ["survival", "hostage", "escape", "chase", "race against time"],
        "scary": ["haunted house", "ghost", "demon", "possession", "serial killer",
                   "slasher", "supernatural", "paranormal"],
        "dark": ["serial killer", "revenge", "corruption", "conspiracy", "noir"],
        "epic": ["war", "battle", "kingdom", "empire", "quest", "prophecy"],
        "twist": ["twist ending", "plot twist", "unreliable narrator", "surprise ending"],
        "plot twist": ["twist ending", "plot twist", "unreliable narrator", "surprise ending"],
    }
    theme_terms = set()
    for key, kws in THEME_KEYWORDS.items():
        if key in pref_lower:
            theme_terms.update(kws)

    theme_score = pd.Series(0.0, index=df.index)
    if theme_terms:
        theme_score = df["keywords"].str.lower().apply(
            lambda kw: sum(1 for t in theme_terms if t in kw)
        )

    # --- Reference keyword matching (bonus for "like X" queries) ---
    ref_score = pd.Series(0, index=df.index)
    if ref_keywords:
        ref_score = df["keywords"].str.lower().apply(
            lambda kw: sum(1 for rk in ref_keywords if rk in kw)
        )

    # --- Quality signal ---
    vote_avg_norm = df["vote_average"].clip(0, 10) / 10.0
    vote_count_log = np.log1p(df["vote_count"].clip(0))
    max_log = vote_count_log.max() if vote_count_log.max() > 0 else 1
    vote_count_norm = vote_count_log / max_log
    quality_score = vote_avg_norm * 0.6 + vote_count_norm * 0.4

    # --- Combine scores ---
    df = df.copy()
    df["_genre_score"] = genre_score * 3.0
    df["_text_score"] = text_score * 2.0
    df["_theme_score"] = theme_score * 3.0    # Thematic keyword matching
    df["_ref_score"] = ref_score * 2.5        # "Like X" keyword similarity
    df["_quality_score"] = quality_score * 2.0
    df["_total_score"] = (df["_genre_score"] + df["_text_score"] +
                          df["_theme_score"] + df["_ref_score"] +
                          df["_quality_score"])

    return df.sort_values("_total_score", ascending=False)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _get_client() -> ollama.Client:
    """Create an Ollama client with the API key from environment."""
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )


def _format_movie(row) -> str:
    """Format a single movie row into a compact but informative block for the LLM."""
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
    """Build the recommendation prompt with shortlisted candidates."""
    movie_blocks = []
    for row in candidates.itertuples():
        movie_blocks.append(f"- {_format_movie(row)}")
    movie_list = "\n\n".join(movie_blocks)

    history_text = (
        ", ".join(
            f'"{name}" (tmdb_id={tid})' for name, tid in zip(history, history_ids)
        ) if history else "none"
    )

    prompt = f"""Pick the best movie for this user. Write a compelling, personalized description (≤500 chars).

User wants: "{preferences}"
Already watched (do NOT pick): {history_text}

Candidates:
{movie_list}

Pick one. Connect description to their preferences. Be specific and engaging.
Return ONLY JSON: {{"tmdb_id": <int>, "description": "<text>"}}"""
    return prompt


def _parse_llm_response(content: str) -> dict:
    """Parse the LLM response, handling common formatting issues."""
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

    raise ValueError(f"Could not parse LLM response as JSON: {content[:200]}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    history_id_set = set(history_ids)

    # Stage 1: Score and shortlist candidates
    scored = _score_movies(preferences, history_id_set)
    candidates = scored.head(6)

    # Stage 2: LLM picks the best match and writes description
    prompt = _build_prompt(preferences, history, history_ids, candidates)
    client = _get_client()

    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"num_predict": 200, "temperature": 0},
    )

    result = _parse_llm_response(response.message.content)

    # Safety: ensure tmdb_id is int and valid
    tmdb_id = int(result["tmdb_id"])
    if tmdb_id not in VALID_IDS or tmdb_id in history_id_set:
        fallback = candidates[~candidates["tmdb_id"].isin(history_id_set)].iloc[0]
        tmdb_id = int(fallback.tmdb_id)

    description = str(result.get("description", ""))[:500]

    return {"tmdb_id": tmdb_id, "description": description}


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences", type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history", type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
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
    history = (
        [t.strip() for t in history_raw.split(",") if t.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start

    print(json.dumps(result, indent=2))
    print(f"\nServed in {elapsed:.2f}s")
