# Movie Recommender - Streamlit app
# Same logic as my notebook: build "tags" for each movie, turn them into vectors,
# then recommend the movies whose vectors are most similar (cosine similarity).

import ast
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Step 1: helper functions ----------
def get_names(text, limit=None):
    """Convert a string like "[{'name': 'Action'}, ...]" into ['Action', ...]"""
    names = []
    for item in ast.literal_eval(text):
        if limit is not None and len(names) == limit:
            break
        names.append(item["name"])
    return names


def get_director(text):
    """Pick only the person whose job is Director from the crew column."""
    return [item["name"] for item in ast.literal_eval(text) if item["job"] == "Director"]


def remove_spaces(words):
    """'Sam Worthington' -> 'SamWorthington' so a name stays one token."""
    return [w.replace(" ", "") for w in words]


# ---------- Step 2: load data and build the model (runs once, then cached) ----------
@st.cache_resource
def build_model():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[["title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.dropna().drop_duplicates(subset="title").reset_index(drop=True)

    # tags = overview + genres + keywords + top 3 cast + director
    tags = (
        movies["overview"].apply(lambda x: x.split())
        + movies["genres"].apply(get_names).apply(remove_spaces)
        + movies["keywords"].apply(get_names).apply(remove_spaces)
        + movies["cast"].apply(lambda x: get_names(x, limit=3)).apply(remove_spaces)
        + movies["crew"].apply(get_director).apply(remove_spaces)
    )
    tags = tags.apply(lambda words: " ".join(words).lower())

    # stemming: "loved", "loving" -> "love" (done BEFORE vectorizing)
    stemmer = PorterStemmer()
    tags = tags.apply(lambda text: " ".join(stemmer.stem(w) for w in text.split()))

    # bag of words: each movie becomes a vector of counts of the 5000 most common words
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(tags)

    return movies[["title", "overview"]], vectors


# ---------- Step 3: recommendation function ----------
def recommend(title, how_many=5):
    movies, vectors = build_model()
    index = movies[movies["title"] == title].index[0]

    # similarity of the chosen movie with every movie
    scores = cosine_similarity(vectors[index], vectors)[0]

    # sort high -> low, skip position 0 (the movie itself)
    best = scores.argsort()[::-1][1 : how_many + 1]
    return movies.iloc[best]


# ---------- Step 4: the web page ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.write("Pick a movie and get similar ones (content-based filtering on TMDB 5000).")

movies, _ = build_model()
choice = st.selectbox("Movie", movies["title"].sort_values(), index=None,
                      placeholder="Type to search...")
count = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend") and choice:
    for _, row in recommend(choice, count).iterrows():
        st.subheader(row["title"])
        st.write(row["overview"])