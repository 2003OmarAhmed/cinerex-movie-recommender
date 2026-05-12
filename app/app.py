import streamlit as st
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="CineRex",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------
# LOAD DATASET
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "movies_v2.csv")

df = pd.read_csv(file_path)

# ---------------------------------
# CLEAN DATA
# ---------------------------------

df['combined_features'] = df['combined_features'].fillna('')

# ---------------------------------
# TF-IDF
# ---------------------------------

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ---------------------------------
# SIMILARITY MATRIX
# ---------------------------------

similarity = cosine_similarity(tfidf_matrix)

# ---------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------

def recommend(movie_title):

    index = df[df['title'] == movie_title].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ratings = []
    recommended_links = []

    for movie in movie_list:

        recommended_movies.append(df.iloc[movie[0]].title)

        poster_path = df.iloc[movie[0]].poster_path
        full_poster_path = "https://image.tmdb.org/t/p/w500/" + str(poster_path)
        recommended_posters.append(full_poster_path)

        rating = df.iloc[movie[0]].rating
        recommended_ratings.append(rating)

        link = df.iloc[movie[0]].imdb_link
        if pd.isna(link):
            link = "#"
        recommended_links.append(link)

    return recommended_movies, recommended_posters, recommended_ratings, recommended_links

# ---------------------------------
# UI HEADER
# ---------------------------------

st.title("🎬 CineRex AI")
st.subheader("Find your next favorite movie instantly with AI-powered recommendations")
st.markdown("---")

# ---------------------------------
# GLOBAL CSS (HOVER EFFECT)
# ---------------------------------

st.markdown("""
<style>
.movie-card {
    background-color:#1e1e1e;
    padding:10px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 4px 10px rgba(0,0,0,0.5);
    transition: all 0.3s ease;
}

.movie-card:hover {
    transform: scale(1.05);
    box-shadow:0px 10px 25px rgba(255,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# MOVIE SELECTOR
# ---------------------------------

movie_titles = df['title'].values

selected_movie = st.selectbox(
    "Choose a movie",
    movie_titles
)

# ---------------------------------
# BUTTON
# ---------------------------------

if st.button("Get Recommendations"):

    recommendations, posters, ratings, links = recommend(selected_movie)

    st.markdown("## 🔥 Top Picks for You")

    cols = st.columns(5)

    for col, movie, poster, rating, link in zip(cols, recommendations, posters, ratings, links):

        rating_color = (
            "green" if rating >= 8
            else "orange" if rating >= 6
            else "red"
        )

        with col:

            # ✅ CLICKABLE POSTER (SAFE STREAMLIT VERSION)
            st.markdown(
                f"""
                <a href="{link}" target="_blank">
                    <img src="{poster}" style="width:100%; border-radius:10px;">
                </a>
                """,
                unsafe_allow_html=True
            )

            # Movie title
            st.markdown(f"**{movie}**")

            # Rating badge
            st.markdown(
                f"<span style='color:{rating_color}; font-weight:bold;'>⭐ {float(rating):.1f}/10</span>",
                unsafe_allow_html=True
            )