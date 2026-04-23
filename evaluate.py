"""
LLM-as-a-Judge evaluation for the movie recommender.

Runs diverse test prompts through get_recommendation(), then uses the same
LLM to score each recommendation on three dimensions:
  1. Relevance  — Does the movie match what the user asked for?
  2. Description — Is the pitch compelling, personalized, and engaging?
  3. Persuasion  — Would this make someone actually want to watch the movie?

Each dimension is scored 1-5. We report per-prompt and aggregate scores.
"""

import json
import os
import time

from dotenv import load_dotenv
import ollama
import pandas as pd

load_dotenv()

from llm import get_recommendation, MOVIES_DF, MODEL

# ---------------------------------------------------------------------------
# Test cases — diverse preference styles that a real judge might use
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "superhero action",
        "preferences": "I love action movies with superheroes.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "funny feel-good",
        "preferences": "I want something funny and feel-good.",
        "history": ["The Dark Knight Rises"],
        "history_ids": [49026],
    },
    {
        "label": "mind-bending sci-fi",
        "preferences": "Something like Inception — mind-bending, visually stunning, makes you think.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "scary horror",
        "preferences": "I want to be genuinely scared. Give me the best horror movie.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "romantic drama",
        "preferences": "A beautiful love story that will make me cry.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "animated family",
        "preferences": "Something I can watch with my kids, animated and fun.",
        "history": ["Frozen", "Coco"],
        "history_ids": [109445, 354912],
    },
    {
        "label": "crime thriller",
        "preferences": "I love crime thrillers with plot twists, like Se7en or Gone Girl.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "vague/open",
        "preferences": "I'm bored, just recommend something really good.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "specific director",
        "preferences": "I love Christopher Nolan movies. What should I watch?",
        "history": ["Interstellar", "The Dark Knight"],
        "history_ids": [157336, 49026],
    },
    {
        "label": "war/historical",
        "preferences": "An intense war movie based on real events.",
        "history": [],
        "history_ids": [],
    },
    # --- Additional diverse tests ---
    {
        "label": "date night",
        "preferences": "Something fun for date night, not too heavy.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "specific actor",
        "preferences": "I love anything with Leonardo DiCaprio.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "foreign language",
        "preferences": "I want a great foreign language film, maybe Korean or Japanese.",
        "history": ["Parasite"],
        "history_ids": [496243],
    },
    {
        "label": "nostalgia/90s-2000s vibe",
        "preferences": "Something that gives me early 2010s nostalgia, like The Hunger Games era.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "underrated gem",
        "preferences": "Recommend me something underrated that most people haven't seen.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "heavy watch history",
        "preferences": "I want a great sci-fi movie.",
        "history": ["Interstellar", "The Martian", "Arrival", "Dune", "Blade Runner 2049"],
        "history_ids": [157336, 286217, 329865, 438631, 335984],
    },
    {
        "label": "mood: rainy day",
        "preferences": "It's a rainy Sunday, I want something cozy and slow-paced.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "documentary",
        "preferences": "I want a documentary that will blow my mind.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "franchise/sequel",
        "preferences": "I loved the first John Wick, what else should I watch?",
        "history": ["John Wick"],
        "history_ids": [245891],
    },
    {
        "label": "anti-preference",
        "preferences": "Anything but horror. I hate being scared.",
        "history": [],
        "history_ids": [],
    },
    # --- Gibberish & edge cases ---
    {
        "label": "pure gibberish",
        "preferences": "xyzabc123 qwerty asdf",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "gibberish with hint",
        "preferences": "sdvdsgcomdyj funnylol",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "single word",
        "preferences": "robots",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "emoji only",
        "preferences": "🎬🍿👻🔪",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "very long preference",
        "preferences": "I want a movie that has great acting, amazing cinematography, a plot that keeps me on the edge of my seat, preferably with some humor mixed in, maybe set in a big city, with a strong female lead, and a twist ending that I won't see coming.",
        "history": [],
        "history_ids": [],
    },
]


def _get_client() -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )


def _get_movie_info(tmdb_id: int) -> dict:
    """Look up movie details from the dataset."""
    row = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
    if row.empty:
        return {"title": "Unknown", "genres": "", "overview": ""}
    row = row.iloc[0]
    return {
        "title": row["title"],
        "year": row["year"],
        "genres": row["genres"],
        "director": row["director"],
        "overview": row["overview"][:200],
        "vote_average": row["vote_average"],
    }


def judge_recommendation(preferences: str, history: list[str],
                         tmdb_id: int, description: str) -> dict:
    """Use the LLM to score a recommendation on relevance, description quality, persuasion."""
    movie = _get_movie_info(tmdb_id)

    prompt = f"""You are a movie recommendation judge. Score this recommendation.

USER ASKED FOR: "{preferences}"
ALREADY WATCHED: {history if history else "nothing"}

RECOMMENDED MOVIE:
  Title: "{movie['title']}" ({movie['year']})
  Genres: {movie['genres']}
  Director: {movie['director']}
  Rating: {movie['vote_average']}/10
  Overview: {movie['overview']}

DESCRIPTION GIVEN TO USER:
"{description}"

Score each dimension from 1 (terrible) to 5 (excellent):

1. RELEVANCE: Does this movie match what the user asked for? (genre, mood, themes, specific requests)
   1=completely wrong genre/mood, 3=somewhat related, 5=perfect match

2. DESCRIPTION_QUALITY: Is the description compelling, personalized, and well-written?
   1=generic/boring, 3=decent but generic, 5=personalized, engaging, makes you want to watch

3. PERSUASION: Would this recommendation convince someone to actually watch this movie?
   1=not at all, 3=maybe, 5=absolutely

Return ONLY JSON: {{"relevance": <1-5>, "description_quality": <1-5>, "persuasion": <1-5>, "reasoning": "<brief explanation>"}}"""

    client = _get_client()
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"num_predict": 256},
    )

    content = response.message.content
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code blocks
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding any JSON object
    match = re.search(r"\{[^{}]*\"relevance\"[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"relevance": 0, "description_quality": 0, "persuasion": 0,
            "reasoning": f"Failed to parse: {content[:100]}"}


def run_evaluation():
    """Run all test cases and print a scorecard."""
    print("=" * 70)
    print("MOVIE RECOMMENDER EVALUATION")
    print("=" * 70)

    results = []

    for tc in TEST_CASES:
        label = tc["label"]
        print(f"\n--- {label} ---")
        print(f"  Preferences: {tc['preferences']}")

        # Get recommendation
        start = time.perf_counter()
        try:
            rec = get_recommendation(tc["preferences"], tc["history"], tc["history_ids"])
            elapsed = time.perf_counter() - start
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"label": label, "relevance": 0, "description_quality": 0, "persuasion": 0})
            continue

        movie = _get_movie_info(rec["tmdb_id"])
        print(f"  Recommended: \"{movie['title']}\" ({movie['year']}) [{movie['genres']}]")
        print(f"  Description: {rec['description'][:120]}...")
        print(f"  Time: {elapsed:.2f}s")

        # Judge it
        scores = judge_recommendation(
            tc["preferences"], tc["history"],
            rec["tmdb_id"], rec["description"]
        )
        print(f"  Scores: relevance={scores.get('relevance')}, "
              f"description={scores.get('description_quality')}, "
              f"persuasion={scores.get('persuasion')}")
        print(f"  Reasoning: {scores.get('reasoning', 'N/A')[:150]}")

        results.append({
            "label": label,
            "movie": movie["title"],
            "relevance": scores.get("relevance", 0),
            "description_quality": scores.get("description_quality", 0),
            "persuasion": scores.get("persuasion", 0),
            "time": elapsed,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    for col in ["relevance", "description_quality", "persuasion"]:
        avg = df[col].mean()
        print(f"  Avg {col}: {avg:.2f} / 5.00")

    overall = df[["relevance", "description_quality", "persuasion"]].mean().mean()
    print(f"\n  OVERALL SCORE: {overall:.2f} / 5.00")
    print(f"  Avg response time: {df['time'].mean():.2f}s")

    # Show weakest areas
    print("\n  Per-test breakdown:")
    for _, row in df.iterrows():
        total = row["relevance"] + row["description_quality"] + row["persuasion"]
        print(f"    {row['label']:25s} → {row.get('movie', 'N/A'):30s} "
              f"R={row['relevance']} D={row['description_quality']} P={row['persuasion']} "
              f"Total={total}/15")

    return df


if __name__ == "__main__":
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY is not set.")
        exit(1)
    run_evaluation()
