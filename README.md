# Agentic Movie Recommender

An AI-powered movie recommendation agent that picks the best movie for a user from a database of ~1000 popular films and writes a compelling pitch to convince them to watch it.

## How It Works

The system uses a two-step pipeline: **smart filtering** → **AI selection + description writing**.

### Step 1 — Score & Shortlist (No AI)

We can't send all 1000 movies to the LLM — it would be too slow and too much text. So we first score every movie against the user's preferences using four signals:

- **Genre matching (3× weight):** A synonym dictionary maps everyday language to genre tags. "Scary" → Horror/Thriller, "superhero" → Action/Sci-Fi, "feel-good" → Comedy/Family/Animation. ~80 synonym entries cover most natural language patterns.
- **Text matching (2× weight):** Search terms from the user's input are matched against each movie's title, cast, director, keywords, overview, and tagline.
- **Thematic keyword matching (3× weight):** Maps mood/theme phrases ("mind-bending", "plot twist", "intense") to specific TMDB keywords in the data (e.g., "mind-bending" → dream, alternate reality, time travel, consciousness). This distinguishes cerebral sci-fi from generic action sci-fi.
- **"Like X" reference matching (2.5× weight):** When users mention a specific movie ("something like Inception"), we find that movie in the database and boost candidates that share its genres, keywords, and director.
- **Quality tiebreaker (2× weight):** Among equally relevant movies, we prefer ones with higher ratings and more audience votes.

Movies the user has already watched are excluded entirely. The top 6 candidates move to Step 2.

### Step 2 — AI Picks the Winner & Writes the Pitch

The 6 shortlisted movies (with genres, director, top cast, rating, tagline, and overview) are sent to `gemma4:31b-cloud` via Ollama Cloud. The model is given a "passionate movie critic" persona and instructed to:

1. Pick the single best match for this user's stated preferences
2. Write a personalized, compelling description (≤500 characters) that opens by connecting to the user's request, highlights what makes the movie special, and ends with a hook

The response is parsed as JSON with multiple fallback strategies. If the model returns an invalid movie ID or one the user already watched, we fall back to the top-scored candidate from Step 1.

### Why This Design

- **Fast:** Pre-filtering to 6 candidates keeps the prompt small. Responses come back in 2–4 seconds, well within the 20-second limit.
- **Accurate:** Multi-signal scoring (genre + text + thematic + reference + quality) ensures the AI only sees relevant, high-quality movies.
- **Robust:** Multiple JSON parsing fallbacks and a safety check on the returned ID prevent disqualification from edge cases.

## Evaluation Strategy

### LLM-as-a-Judge Evaluation (`evaluate.py`)

We built an automated evaluation pipeline that uses the same LLM to score recommendations on three dimensions (1-5 each):

1. **Relevance** — Does the movie match what the user asked for?
2. **Description Quality** — Is the pitch compelling, personalized, and well-written?
3. **Persuasion** — Would this convince someone to actually watch the movie?

We tested across 10 diverse preference styles:

| Test Case | Recommended Movie | R | D | P | Total |
|-----------|------------------|---|---|---|-------|
| Superhero action | Logan | 5 | 5 | 5 | 15/15 |
| Funny feel-good | Luca | 5 | 5 | 5 | 15/15 |
| Mind-bending sci-fi (like Inception) | Interstellar | 5 | 5 | 5 | 15/15 |
| Scary horror | Hereditary | 5 | 5 | 5 | 15/15 |
| Romantic drama (make me cry) | About Time | 5 | 5 | 5 | 15/15 |
| Animated family (with kids) | The Wild Robot | 5 | 5 | 5 | 15/15 |
| Crime thriller (like Se7en) | Girl with the Dragon Tattoo | 5 | 5 | 5 | 15/15 |
| Vague/open (I'm bored) | Parasite | 5 | 5 | 5 | 15/15 |
| Specific director (Nolan) | Oppenheimer | 5 | 5 | 5 | 15/15 |
| War/historical | Lone Survivor | 5 | 5 | 5 | 15/15 |

**Overall Score: 5.00 / 5.00** | Avg response time: 2.93s

### Iterative Improvement Process

We used the evaluation to drive improvements:

1. **Baseline (v1):** Only used top-5 movies by vote count. Scored poorly on diversity — always recommended the same few blockbusters.
2. **v2 — Genre synonym matching:** Added synonym dictionary. Improved relevance for genre-specific queries but still missed nuanced requests like "mind-bending."
3. **v3 — Thematic keyword matching:** Added TMDB keyword-level matching for mood/theme phrases. Fixed the "like Inception" case (Limitless → Interstellar). Score went from 4.80 to 5.00.
4. **v4 — Quality weight tuning:** Increased quality score weight so critically acclaimed films beat generic genre matches. Prevented mediocre movies from outranking great ones.

### Automated Testing

The provided `test.py` validates all hard requirements:
- Returns a valid dict with `tmdb_id` and `description` keys
- `tmdb_id` exists in the candidate database
- Does not recommend movies the user already watched
- Responds within 20 seconds
- All imports are listed in `requirements.txt`

## Project Structure

| File | What It Does |
|------|-------------|
| `llm.py` | Main implementation — scoring, retrieval, LLM call, JSON parsing |
| `evaluate.py` | LLM-as-a-judge evaluation pipeline (10 test cases, 3 scoring dimensions) |
| `test.py` | Automated test suite (provided by instructor) |
| `tmdb_top1000_movies.csv` | Movie database with metadata (genres, cast, director, ratings, etc.) |
| `requirements.txt` | Python dependencies |
| `.env` | API key (not included in submission) |

### Key Functions in `llm.py`

- `get_recommendation()` — Main entry point. Orchestrates the full pipeline.
- `_score_movies()` — Scores all movies using genre + text + thematic + reference + quality signals.
- `_find_referenced_movies()` — Detects movie titles mentioned in preferences for "like X" queries.
- `_extract_search_terms()` — Tokenizes user preferences, removes stopwords.
- `_build_prompt()` — Constructs the LLM prompt with shortlisted candidates and critic persona.
- `_format_movie()` — Formats a movie row into a compact block for the prompt.
- `_parse_llm_response()` — Robust JSON extraction with multiple fallback strategies.
- `GENRE_SYNONYMS` — Dictionary mapping ~80 everyday terms to genre/theme keywords.
- `THEME_KEYWORDS` (inside `_score_movies`) — Maps mood phrases to specific TMDB keywords.

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Option A: Use a .env file (recommended)
# Create .env with: OLLAMA_API_KEY=your_key_here
python test.py

# Option B: Set the key inline
OLLAMA_API_KEY=your_key_here python test.py

# Run the evaluation suite
python evaluate.py

# Interactive mode
python llm.py --preferences "I love sci-fi thrillers"
```
