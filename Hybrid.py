import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Import libraries for content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


movies = pd.read_csv("/Users/katebukina/PycharmProjects/MQUEA/Project/tmdb_5000_movies.csv")
ratings = pd.read_csv("/Users/katebukina/PycharmProjects/MQUEA/Project/ratings_small.csv")


# Clean and preprocess movie data
movies = movies[['id', 'title', 'overview']].dropna()
movies['id'] = movies['id'].astype(str)  # Ensure movie IDs are strings

# Prepare collaborative filtering dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

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

# Example Usage:
user_id = 1  # Change this based on the user
movie_title = "The Dark Knight"  # Change this to any movie in the dataset

# Get hybrid recommendations
print(hybrid_recommend(1, 'Avatar'))
print(hybrid_recommend(2, 'Avatar'))
print(hybrid_recommend(3, 'Avatar'))