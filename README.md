# Agentic Movie Recommender

An AI-powered movie recommendation agent that picks the best movie for a user from a database of ~1000 popular films and writes a compelling pitch to convince them to watch it.

## How It Works

The system uses a **hybrid retrieval** pipeline followed by **LLM reasoning**. There are no hardcoded synonym dictionaries, stopword lists, or genre mappings — the models do all the thinking.

### Step 1 — Hybrid Retrieval (Embeddings + Keyword Search)

We use two retrieval methods in parallel to find candidate movies:

- **Semantic embeddings** (`BAAI/bge-small-en-v1.5` via fastembed): Encodes the user's query and all movies into vectors, finds closest matches by cosine similarity. Understands meaning — "mind-bending" matches sci-fi thrillers, "feel-good" matches comedies.
- **TF-IDF keyword search** (scikit-learn): Matches exact terms like director names, actor names, and specific movie references that embeddings might miss.
- **Quality signal**: Blends in normalized rating + log-scaled vote count so acclaimed films rank above obscure matches.

Movie embeddings are **pre-computed and cached** to a `.npy` file (~1ms load vs ~15s compute). The two retrieval scores are combined, and the top 8 candidates move to Step 2.

### Step 2 — LLM Does the Thinking

The 6 shortlisted movies (with genres, director, cast, rating, and plot summary) are sent to `gemma4:31b-cloud` via Ollama Cloud. The LLM is given the user's full request and instructed to:

1. Analyze what the user is really looking for — genre, mood, themes, specific references
2. Pick the ONE best match from the candidates
3. Write a compelling, personalized description (≤500 chars) that sells the movie

The LLM handles all the nuance — understanding that "something like Inception" means mind-bending sci-fi, that "make me cry" means emotional drama, that "Nolan fan" means a specific directorial style. No rules needed.

### Why This Design

- **No hardcoding:** Zero synonym dictionaries, stopword lists, or genre mappings. Embeddings understand semantics, TF-IDF catches exact terms, and the LLM reasons about intent.
- **Hybrid retrieval:** Embeddings alone miss exact matches (director names); keywords alone miss meaning ("feel-good"). Together they cover both.
- **Fast:** Cached embeddings load in ~1ms. TF-IDF fits at import (~40ms). Query time is ~5ms. Total response is 3–5s.

## Approaches Explored

We tried three retrieval strategies during development:

| Approach | Startup | Per Query | Quality | Method |
|----------|---------|-----------|---------|--------|
| 1. Hardcoded synonyms | 0ms | 2–5s | 5.00/5.00 | ~80 manual synonym entries + stopwords + theme maps |
| 2. Embeddings only | ~0.1s | 3–5s | 4.77/5.00 | Semantic similarity + quality |
| 3. Hybrid (current) | ~0.1s | 3–5s | TBD | Embeddings + TF-IDF keywords + quality |

We chose **Approach 3** because it combines the strengths of both retrieval methods — embeddings understand meaning while TF-IDF catches exact terms like director names and actor references — all without any hardcoded rules. The LLM does all the reasoning about which movie best fits the user's intent.

## Evaluation Strategy

### LLM-as-a-Judge Evaluation (`evaluate.py`)

We built an automated evaluation pipeline that uses the same LLM to score recommendations on three dimensions (1-5 each):

1. **Relevance** — Does the movie match what the user asked for?
2. **Description Quality** — Is the pitch compelling, personalized, and well-written?
3. **Persuasion** — Would this convince someone to actually watch the movie?

Results across 10 diverse test cases:

| Test Case | Recommended Movie | R | D | P | Total |
|-----------|------------------|---|---|---|-------|
| Superhero action | Avengers: Infinity War | 5 | 5 | 5 | 15/15 |
| Funny feel-good | The Intern | 5 | 5 | 5 | 15/15 |
| Mind-bending sci-fi (like Inception) | Interstellar | 5 | 5 | 5 | 15/15 |
| Scary horror | A Quiet Place | 4 | 5 | 4 | 13/15 |
| Romantic drama (make me cry) | Me Before You | 5 | 4 | 4 | 13/15 |
| Animated family (with kids) | The Lego Movie | 5 | 4 | 4 | 13/15 |
| Crime thriller (like Se7en) | Girl with the Dragon Tattoo | 5 | 5 | 5 | 15/15 |
| Vague/open (I'm bored) | Parasite | 5 | 5 | 5 | 15/15 |
| Specific director (Nolan) | Dune: Part Two | 4 | 5 | 5 | 14/15 |
| War/historical | Hacksaw Ridge | 5 | 5 | 5 | 15/15 |

**Overall Score: 4.77 / 5.00** | Avg response time: 4.03s

### Iterative Improvement Process

1. **Baseline:** Top-5 movies by vote count only. Always recommended the same blockbusters.
2. **Hardcoded synonyms:** Added genre synonym dictionary + text matching. Scored 5.00/5.00 but heavily rule-based.
3. **Live embeddings:** Replaced all rules with semantic embeddings. No hardcoding but 15s startup.
4. **Cached embeddings (final):** Pre-computed embeddings to `.npy` file. 0.1s startup, no rules, 4.77/5.00 quality.

## Project Structure

| File | What It Does |
|------|-------------|
| `llm.py` | Main implementation — embedding retrieval, LLM call, JSON parsing |
| `movie_embeddings.npy` | Pre-computed movie embedding vectors (cached) |
| `evaluate.py` | LLM-as-a-judge evaluation pipeline |
| `test.py` | Automated test suite (provided by instructor) |
| `tmdb_top1000_movies.csv` | Movie database (~1000 films with metadata) |
| `requirements.txt` | Python dependencies |
| `app.py` | Streamlit web frontend (not part of submission) |
| `llm_hardcoded_backup.py` | Backup of the hardcoded synonym approach |

### Key Functions in `llm.py`

- `get_recommendation()` — Main entry point. Orchestrates retrieval → LLM pipeline.
- `_score_movies()` — Hybrid retrieval: embedding similarity + TF-IDF keywords + quality scoring.
- `_build_prompt()` — Constructs the LLM prompt. The LLM analyzes intent and picks the movie.
- `_format_movie()` — Formats movie metadata for the prompt.
- `_parse_llm_response()` — Robust JSON extraction with multiple fallback strategies.

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key (or use a .env file)
export OLLAMA_API_KEY=your_key_here

# Run tests
python test.py

# Run evaluation
python evaluate.py

# Interactive mode
python llm.py --preferences "I love sci-fi thrillers"

# Web UI
streamlit run app.py
```

### Re-generating Cached Embeddings

If you modify the movie corpus or embedding model, regenerate the cache:

```python
python -c "
from fastembed import TextEmbedding
import pandas as pd, numpy as np
df = pd.read_csv('tmdb_top1000_movies.csv')
for c in ['overview','genres','keywords']: df[c] = df[c].fillna('')
corpus = (df['title']+'. Genres: '+df['genres']+'. '+df['overview'].str[:150]+' Keywords: '+df['keywords']).tolist()
emb = np.array(list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(corpus)))
np.save('movie_embeddings.npy', emb)
print(f'Saved {emb.shape}')
"
```
