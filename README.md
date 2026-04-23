# Agentic Movie Recommender

Recommends movies from a database of ~1000 popular films and writes a personalized pitch to convince you to watch it.

## How It Works

Three stages, no hardcoded rules.

### 1. Hybrid Retrieval

Two methods find candidates in parallel:

- **Semantic embeddings** (BAAI/bge-small-en-v1.5) — understands meaning. "Mind-bending" finds sci-fi thrillers, "feel-good" finds comedies. Pre-computed and cached for fast loading.
- **TF-IDF keyword search** — catches exact terms like director names, actor names, and movie references.
- **Quality signal** — rating + vote count so well-known films rank higher.

Top 8 candidates go to the LLM.

### 2. Chain-of-Thought LLM Reasoning

The LLM (`gemma4:31b-cloud`) reasons step by step:

1. **INTENT** — What does the user actually want?
2. **MATCH** — Which candidate fits best?
3. **SELL** — Write a personalized pitch (≤500 chars)

### 3. TMDB API Enrichment (Optional)

If `TMDB_API_KEY` is set, fetches real user reviews from TMDB and appends a critic quote to the description. Works without it too.

## Evaluation

LLM-as-a-judge pipeline (`evaluate.py`) — 25 test cases scored on relevance, description quality, and persuasion.

| Example query | Recommended | Score |
|---|---|---|
| "I love action movies with superheroes" | Spider-Man: Into the Spider-Verse | 14/15 |
| "Something like Inception — mind-bending" | Interstellar | 15/15 |
| "I love crime thrillers like Se7en" | The Girl with the Dragon Tattoo | 15/15 |
| "I love Christopher Nolan movies" | Oppenheimer | 13/15 |
| "robots" (single word) | The Wild Robot | 14/15 |
| "sdvdsgcomdyj funnylol" (gibberish) | We're the Millers | 13/15 |

Handles gibberish, emojis, single words, and long inputs gracefully.

## How We Got Here

1. Started with top 5 movies by vote count — same recommendations every time.
2. Added hardcoded synonym mappings — good results but too rule-based.
3. Switched to cached semantic embeddings — no hardcoding, fast startup.
4. Added TF-IDF keyword search alongside embeddings for better coverage.
5. Added chain-of-thought prompting for smarter LLM reasoning.
6. Integrated TMDB API for real critic reviews.

## Running

```bash
pip install -r requirements.txt
export OLLAMA_API_KEY=your_key_here        # required
export TMDB_API_KEY=your_tmdb_key_here     # optional

python test.py                              # grader tests
python evaluate.py                          # 25-case evaluation
python llm.py --preferences "sci-fi"        # interactive
streamlit run app.py                        # web UI
```

## Files

| File | Purpose |
|------|---------|
| `llm.py` | Main agent |
| `movie_embeddings.npy` | Cached embeddings |
| `evaluate.py` | Evaluation pipeline (25 tests) |
| `test.py` | Grader test suite |
| `tmdb_top1000_movies.csv` | Movie database |
| `app.py` | Streamlit frontend |
