# Agentic Movie Recommender

An AI-powered movie recommendation agent that picks the best movie for a user from a database of ~1000 popular films and writes a compelling pitch to convince them to watch it.

## How It Works

The system uses a two-stage pipeline: **semantic retrieval** → **LLM selection + description writing**. There are no hardcoded synonym dictionaries, stopword lists, or genre mappings — the models do all the thinking.

### Step 1 — Semantic Embedding Retrieval (No Hardcoded Rules)

We use a pre-trained sentence embedding model (`BAAI/bge-small-en-v1.5` via fastembed) to understand what the user is looking for. Each movie in the database is represented as a vector encoding its title, genres, overview, and keywords. The user's query is encoded the same way, and we find the closest matches by cosine similarity.

To avoid recommending obscure films that happen to match keywords, we blend the semantic similarity score with a quality signal (normalized rating + log-scaled vote count). This ensures well-regarded movies rise above generic matches.

Movie embeddings are **pre-computed and cached** to a `.npy` file, so startup is ~0.1s instead of ~15s. The embedding model only needs to encode the user's short query at runtime (~4ms).

Movies the user has already watched are excluded. The top 6 candidates move to Step 2.

### Step 2 — LLM Does the Thinking

The 6 shortlisted movies (with genres, director, cast, rating, and plot summary) are sent to `gemma4:31b-cloud` via Ollama Cloud. The LLM is given the user's full request and instructed to:

1. Analyze what the user is really looking for — genre, mood, themes, specific references
2. Pick the ONE best match from the candidates
3. Write a compelling, personalized description (≤500 chars) that sells the movie

The LLM handles all the nuance — understanding that "something like Inception" means mind-bending sci-fi, that "make me cry" means emotional drama, that "Nolan fan" means a specific directorial style. No rules needed.

### Why This Design

- **No hardcoding:** Zero synonym dictionaries, stopword lists, or genre mappings. The embedding model understands semantics, and the LLM reasons about intent.
- **Fast:** Cached embeddings load in ~1ms. Query embedding takes ~4ms. Total response time is 3–5s, well within the 20s limit.
- **Robust:** 15s timeout on the LLM call with a graceful fallback. If the API is slow, we return the top-scored candidate with a pre-built description.

## Approaches Explored

We tried three retrieval strategies during development:

| Approach | Startup | Per Query | Quality | Hardcoded Rules |
|----------|---------|-----------|---------|-----------------|
| 1. Hardcoded synonyms | 0ms | 2–5s | 5.00/5.00 | ~80 synonym entries, stopwords, theme maps |
| 2. Live embeddings | ~15s | 3–5s | 4.77/5.00 | None |
| 3. Cached embeddings (current) | ~0.1s | 3–5s | 4.77/5.00 | None |

We chose **Approach 3** because it eliminates all hardcoded rules while keeping startup fast. The slight quality drop (5.00 → 4.77) comes from edge cases where embeddings pick a good-but-not-perfect movie (e.g., "A Quiet Place" instead of "Hereditary" for horror). The LLM compensates by writing strong descriptions regardless.

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
- `_score_movies()` — Embedding cosine similarity + quality scoring. No hardcoded rules.
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
