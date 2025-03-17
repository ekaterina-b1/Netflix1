from flask import Flask, render_template, request
import pandas as pd
import numpy as np
# Import libraries for content-based filtering
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use relative paths instead of absolute paths
movies_path = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
credits_path = os.path.join(BASE_DIR, "tmdb_5000_credits.csv")
ratings_path = os.path.join(BASE_DIR, "ratings_small.csv")

# Load CSV files
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Clean and preprocess movie data
movies = movies[['id', 'title', 'overview']].dropna()


# Content-based filtering: Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of movie titles to DataFrame indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


df1 = pd.read_csv(credits_path)

# Merging two datasets on a common column id
df1.columns = ['id','tittle','cast','crew']
df = movies.merge(df1,on='id')

# Trending Movies Function
def get_trending_movies():
    c = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.95)
    popular_movies = df.copy().loc[df['vote_count'] >= m]

    def weighted_rating(x):
        v = x['vote_count']
        r = x['vote_average']
        return (v / (v + m) * r) + (m / (v + m) * c)

    popular_movies['score'] = popular_movies.apply(weighted_rating, axis=1)
    popular_movies = popular_movies.sort_values('score', ascending=False)
    return popular_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


def get_recommendations(title,cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']

for i in features:
    df[i] = df[i].apply(literal_eval)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(n):
    for i in n:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

    # Function for getting top 3 elements from the list
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

        # Return empty list in case of missing/malformed data
    return []

    # Define new director, cast, genres and keywords features that are in a suitable form.
df['director'] = df['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for i in features:
    df[i] = df[i].apply(get_list)

    # Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

    # Applying clean data function

features = ['cast', 'keywords', 'genres', 'director']

for i in features:
    df[i] = df[i].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

df['soup'] = df.apply(create_soup, axis=1)

cv_matrix = count.fit_transform(df['soup'])

cosine_sim2 = cosine_similarity(cv_matrix, cv_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/trending')
def trending():
    trending_movies = get_trending_movies()
    return render_template('trending.html', trending_movies=trending_movies, title="Trending Movies")



@app.route('/content_recommend', methods=['POST'])
def content_recommend():
    movie_title = request.form['movie_title']
    recommendations = get_recommendations(movie_title)
    return render_template('content_recommendations.html', recommendations=recommendations)

@app.route('/metadata_recommend', methods=['POST'])
def metadata_recommend():
    movie_title = request.form['movie_title'].strip()  # Trim spaces
    recommendations = get_recommendations(movie_title, cosine_sim2)
    return render_template('metadata_recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from Render environment
    app.run(host='0.0.0.0', port=port, debug=False)
