# 🎬 Movie Recommender System

A content-based movie recommendation web app built with **Python, scikit-learn and Streamlit**.
Select a movie and the app suggests similar ones based on plot, genres, keywords, cast and director.

**Live demo:** https://moive-recommender-system.streamlit.app/

## How it works
1. Merge the TMDB movies and credits datasets on title.
2. Build a `tags` text for each movie: overview + genres + keywords + top 3 cast + director.
3. Remove spaces inside names (`Sam Worthington` → `SamWorthington`) so each name stays one token.
4. Apply Porter stemming (`loved`, `loving` → `love`).
5. Convert tags to vectors using `CountVectorizer` (top 5000 words, English stop words removed).
6. Compute cosine similarity between the selected movie and all others, and return the top N.

## Tech stack
- Python, pandas, NumPy
- scikit-learn (`CountVectorizer`, `cosine_similarity`)
- NLTK (`PorterStemmer`)
- Streamlit (UI)
- Deployed on Render

## Dataset
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle), about 4,800 movies.

## Run locally
```bash
git clone <your-repo-url>
cd movie-recommender
uv venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```
Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside the `data/` folder.

## Project structure
```
├── app.py              # model building + recommendation + Streamlit UI
├── requirements.txt
├── data/               # TMDB CSV files
└── README.md
```

## Limitations and future improvements
- Uses raw word counts. TF-IDF or sentence embeddings could capture meaning better.
- Ignores ratings and user behavior (no collaborative filtering).
- Could add posters via the TMDB API and filters by genre or year.
