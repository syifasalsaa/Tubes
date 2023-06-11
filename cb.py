import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset movie.csv
def load_movie_data():
    movie_data = pd.read_csv('movie.csv')
    return movie_data

# Load dataset movie_credit.csv
def load_movie_credit_data():
    movie_credit_data = pd.read_csv('movie_credit.csv')
    return movie_credit_data

# Get top rated movies
def get_top_rated_movies(movie_data):
    movie_data['release_year'] = movie_data['release_date'].str[:4].fillna(0).astype(int)
    top_rated_movies = movie_data.loc[movie_data['release_year'] > 2000]
    top_rated_movies = top_rated_movies.sort_values('vote_average', ascending=False)
    top_rated_movies = top_rated_movies.head(30)  # Hanya ambil 20 film teratas
    top_rated_movies = top_rated_movies[['title', 'overview']]  # Hanya ambil kolom title dan overview
    return top_rated_movies

# Get popular movies from specific year
def get_popular_movies_by_year(movie_data, year):
    popular_movies = movie_data.loc[(movie_data['release_date'].str[:4] == str(year)) & (movie_data['vote_average'] >= 7.0)]
    popular_movies = popular_movies.sort_values('vote_average', ascending=False)
    popular_movies = popular_movies[['title', 'overview']]  # Hanya ambil kolom title dan overview
    return popular_movies

# Preprocess dataset and split into training and testing data
def preprocess_data(movie_data):
    movie_data['release_year'] = movie_data['release_date'].str[:4].fillna(0).astype(int)
    movie_data = movie_data.loc[movie_data['release_year'] > 2000]
    movie_data = movie_data[['overview', 'vote_average']]  # Select relevant features

    # Split data into training and testing sets
    X = movie_data['overview'].fillna('')
    y = movie_data['vote_average'] >= 7.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Create TF-IDF matrix
def create_tfidf_matrix(X_train, X_test):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix_train = tfidf.fit_transform(X_train)
    tfidf_matrix_test = tfidf.transform(X_test)
    return tfidf_matrix_train, tfidf_matrix_test

# Compute cosine similarity matrix
def compute_cosine_similarity(tfidf_matrix_train, tfidf_matrix_test):
    cosine_sim = linear_kernel(tfidf_matrix_train, tfidf_matrix_test)
    return cosine_sim

# Get recommendations based on selected movie
def get_recommendations(title, movie_data, tfidf_matrix, cosine_sim):
    # Get index of selected movie
    idx = movie_data[movie_data['title'] == title].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 recommendations
    top_movies = sim_scores[1:6]
    movie_indices = [i[0] for i in top_movies]

# Return recommended movies (title and overview)
    recommended_movies = movie_data[['title', 'overview','vote_average']].iloc[movie_indices]
    recommended_movies = recommended_movies.rename(columns={'vote_average': 'Rating'})
    return recommended_movies

# Page for top rated movies
def page_top_rated_movies(movie_data):
    st.title("CINEMATIX")
    st.header("Top Rated Movies")
    st.subheader("Film-Film Pilihan dengan Peringkat Teratas!")
    top_rated_movies = get_top_rated_movies(movie_data)
    st.dataframe(top_rated_movies)

# Page for popular movies
def page_popular_movies(movie_data):
    st.title("CINEMATIX")
    st.header("Popular Movies")
    st.subheader("Jelajahi Film-Film Terpopuler dari Berbagai Tahun di Sini")
    selected_year = st.selectbox("Select a year", list(range(1940, 2021)))
    popular_movies = get_popular_movies_by_year(movie_data, selected_year)
    st.dataframe(popular_movies)

    if selected_year:
    # Display Model Training and Evaluation
        st.header("Tingkat Akurasi")
        st.subheader("Kecocokan film dengan pilihan Mu")
        X_train, X_test, y_train, y_test = preprocess_data(movie_data)
        tfidf_matrix_train, tfidf_matrix_test = create_tfidf_matrix(X_train, X_test)
        model = train_model(tfidf_matrix_train, y_train)
        accuracy = evaluate_model(model, tfidf_matrix_test, y_test)
        accuracy = accuracy * 100
        st.write("Accuracy:", f"{accuracy:.2f}%")

def page_movie_recommendations(movie_data, tfidf_matrix_train, cosine_sim):
    st.title("CINEMATIX")
    st.header("Movie Recommendations")
    st.subheader("Temukan Pengalaman Film Terbaik berdasarkan Pilihan Mu!")
    selected_movie = st.selectbox("Select a movie", movie_data['title'])
    if st.button("Rekomendasi"):
        recommended_movies = get_recommendations(selected_movie, movie_data, tfidf_matrix_train, cosine_sim)
        st.dataframe(recommended_movies)
    # Display Model Training and Evaluation
        st.header("Tingkat Akurasi")
        st.subheader("Kecocokan film dengan pilihan Mu")
        X_train, X_test, y_train, y_test = preprocess_data(movie_data)
        tfidf_matrix_train, tfidf_matrix_test = create_tfidf_matrix(X_train, X_test)
        model = train_model(tfidf_matrix_train, y_train)
        accuracy = evaluate_model(model, tfidf_matrix_test, y_test)
        accuracy = accuracy * 100
        st.write("Accuracy:", f"{accuracy:.2f}%")

# Train a logistic regression model
def train_model(tfidf_matrix_train, y_train):
    model = LogisticRegression()
    model.fit(tfidf_matrix_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, tfidf_matrix_test, y_test):
    y_pred = model.predict(tfidf_matrix_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Main function
def main():
    # Load data
    movie_data = load_movie_data()

    # Set page configuration
    st.set_page_config(layout="wide")

    # Sidebar menu
    st.sidebar.title("CINEMATIX")
    menu = ["Top Rated Movies", "Popular Movies", "Movie Recommendations"]
    choice = st.sidebar.selectbox("Nikmati Beragam Pilihan Menu",menu)
    

    if choice == "Top Rated Movies":
        page_top_rated_movies(movie_data)
    elif choice == "Popular Movies":
        page_popular_movies(movie_data)
    elif choice == "Movie Recommendations":
        X_train, X_test, _, _ = preprocess_data(movie_data)
        tfidf_matrix_train, tfidf_matrix_test = create_tfidf_matrix(X_train, X_test)
        cosine_sim = compute_cosine_similarity(tfidf_matrix_train, tfidf_matrix_test)
        page_movie_recommendations(movie_data, tfidf_matrix_train, cosine_sim)


# Run the app
if __name__ == '__main__':
    main()

    
                     
