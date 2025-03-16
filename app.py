from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as tts
# Import libraries for content-based filtering
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

import os

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
movies['id'] = movies['id'].astype(str)  # Ensure movie IDs are strings

# Prepare collaborative filtering dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = tts(data, test_size=0.2)

# Train the collaborative filtering model (SVD)
svd = SVD()
svd.fit(trainset)

# Build full trainset for predictions
full_trainset = data.build_full_trainset()
svd.fit(full_trainset)

# Content-based filtering: Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of movie titles to DataFrame indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Hybrid Recommendation Function
def hybrid_recommend(user_id, movie_title, top_n=10):
    """
    Recommend movies based on both content-based similarity and collaborative filtering.

    :param user_id: ID of the user for personalized recommendations
    :param movie_title: Input movie title for similarity-based recommendations
    :param top_n: Number of recommendations to return
    :return: List of recommended movie titles
    """
    # 1. Get movie index
    if movie_title not in indices:
        return "Movie not found in database."

    idx = indices[movie_title]

    # 2. Get content-based similar movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the input movie
    movie_indices = [i[0] for i in sim_scores]

    # 3. Get movie IDs of similar movies
    similar_movies = movies.iloc[movie_indices][['id', 'title']]

    # 4. Predict user ratings using SVD for the similar movies
    similar_movies['predicted_rating'] = similar_movies['id'].apply(lambda x: svd.predict(user_id, int(x)).est)

    # 5. Sort by highest predicted rating
    recommended_movies = similar_movies.sort_values(by='predicted_rating', ascending=False).head(top_n)

    return recommended_movies[['title', 'predicted_rating']]

df1 = pd.read_csv(credits_path)
df2 = pd.read_csv(movies_path)

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Merging two datasets on a common column id
df1.columns = ['id','tittle','cast','crew']
df = df2.merge(df1,on='id')

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

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])
# Import linear_kernel from sklearn
from sklearn.metrics.pairwise import linear_kernel

# Compute cosine similarity using tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

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

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
cv_matrix = count.fit_transform(df['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(cv_matrix, cv_matrix)

indices = pd.Series(df.index, index=df['title'])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/trending')
def trending():
    trending_movies = get_trending_movies()
    return render_template('trending.html', trending_movies=trending_movies, title="Trending Movies")

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    movie_title = request.form['movie_title']
    recommendations = hybrid_recommend(user_id, movie_title)
    return render_template('recommendations.html', recommendations=recommendations)

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
    app.run(debug=True)
