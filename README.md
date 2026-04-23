# Agentic Movie Recommender

AI-powered movie recommendation agent. Picks the best movie from ~1000 films and writes a personalized pitch to convince you to watch it.

## Pipeline

Three stages, no hardcoded rules — the models do all the thinking.

### 1. Hybrid Retrieval

Two retrieval methods run in parallel to find candidates:

- **Semantic embeddings** (BAAI/bge-small-en-v1.5): Understands meaning — "mind-bending" finds sci-fi thrillers, "feel-good" finds comedies. Pre-computed and cached to `.npy` for fast loading (~1ms).
- **TF-IDF keyword search**: Catches exact terms embeddings miss — director names, actor names, movie title references.
- **Quality signal**: Rating + vote count blend so acclaimed films rank above obscure matches.

Scores are combined. Top 8 candidates go to the LLM.

### 2. Chain-of-Thought LLM Reasoning

The LLM (`gemma4:31b-cloud`) follows a structured reasoning process:

1. **INTENT** — What is the user really looking for? (genre, mood, themes, references)
2. **MATCH** — Which candidate best fits that intent?
3. **SELL** — Write a personalized description (≤500 chars) that connects the movie to their request

This produces better picks than a simple "choose one" prompt because the model explicitly reasons before deciding.

### 3. TMDB API Enrichment (Optional)

After the LLM picks a movie, we fetch real user reviews from the TMDB API and append a critic quote to the description. Adds credibility. Works without the key too.

## Evaluation

Built an LLM-as-a-judge pipeline (`evaluate.py`) — 25 test cases scored on relevance, description quality, and persuasion (1-5 each).

**Standard preferences (15 tests):** Avg 14.1/15 — strong picks for superhero, comedy, horror, romance, thriller, war, date night, specific actors/directors, foreign language, etc.

**Edge cases (10 tests):** Handles gibberish, emojis, single words, very long inputs. Graceful fallback — never crashes, always under 20s.

| Highlight picks | |
|---|---|
| "I love action movies with superheroes" | Spider-Man: Into the Spider-Verse (14/15) |
| "Something like Inception — mind-bending" | Interstellar (15/15) |
| "I love crime thrillers like Se7en" | The Girl with the Dragon Tattoo (15/15) |
| "I love Christopher Nolan movies" | Oppenheimer (13/15) |
| "robots" (single word) | The Wild Robot (14/15) |
| "sdvdsgcomdyj funnylol" (gibberish) | We're the Millers (13/15) |

## Iteration Journey

1. **Baseline** — Top 5 by vote count. Same movies every time.
2. **Hardcoded synonyms** — 80 manual genre mappings. 5.00/5.00 but heavily rule-based.
3. **Cached embeddings** — Replaced all rules. 0.1s startup, no hardcoding.
4. **Hybrid retrieval** — Added TF-IDF for exact term matching alongside embeddings.
5. **Chain-of-thought** — Structured LLM reasoning (INTENT → MATCH → SELL).
6. **TMDB API** — Real user reviews to enrich descriptions.

## Running

```bash
pip install -r requirements.txt
export OLLAMA_API_KEY=your_key_here        # required
export TMDB_API_KEY=your_tmdb_key_here     # optional

python test.py                              # run tests
python evaluate.py                          # run 25-case evaluation
python llm.py --preferences "sci-fi"        # interactive
streamlit run app.py                        # web UI
```

## Files

| File | Purpose |
|------|---------|
| `llm.py` | Main agent — hybrid retrieval, chain-of-thought LLM, TMDB enrichment |
| `movie_embeddings.npy` | Cached movie embedding vectors |
| `evaluate.py` | LLM-as-a-judge evaluation (25 tests) |
| `test.py` | Grader test suite |
| `tmdb_top1000_movies.csv` | Movie database |
| `app.py` | Streamlit web frontend |
