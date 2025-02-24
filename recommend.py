"""
Simple Content-Based Recommendation System

This script recommends movies based on a user’s input description by comparing
the description with movie plot summaries in a fixed dataset (dataset.csv).

Instructions:
- Place your dataset file named "dataset.csv" in the same directory as this script.
- The CSV file must contain at least the following columns:
    - "Title": The movie title.
    - "Plot": The plot summary or description.
- Run the script with: python recommend.py
- When prompted, type in your movie preference description.
- The script prints out the top recommendation (highest similarity score)
  along with a few other recommendations.

The recommendation is based on TF-IDF vectorization and cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fixed location of the dataset file
DATASET_PATH = "data/movie.csv"

def load_dataset():
    """
    Load the dataset from a fixed CSV file location.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception as e:
        print(f"Error loading dataset from {DATASET_PATH}: {e}")
        exit(1)

def preprocess_text(df, text_column='Plot'):
    """
    Preprocess the text data by filling in missing values.
    
    Args:
        df (pd.DataFrame): The dataset.
        text_column (str): The column containing the textual data (default 'Plot').
        
    Returns:
        pd.Series: Preprocessed text data.
    """
    # Replace missing plot descriptions with an empty string
    df[text_column] = df[text_column].fillna("")
    return df[text_column]

def vectorize_texts(texts):
    """
    Vectorize the text data using TF-IDF.
    
    Args:
        texts (pd.Series): Series of text data.
        
    Returns:
        tuple: A tuple containing the fitted TfidfVectorizer and the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def get_recommendations(query, vectorizer, tfidf_matrix, df, top_n=5):
    """
    Given a user query, compute the cosine similarity between the query
    and each movie plot, and return the top_n recommendations.
    
    Args:
        query (str): The user input description.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        tfidf_matrix: The TF-IDF matrix for the dataset.
        df (pd.DataFrame): The original dataset.
        top_n (int): Number of recommendations to return.
        
    Returns:
        pd.DataFrame: DataFrame of recommended items with similarity scores.
    """
    # Transform the user query into the same TF-IDF vector space
    query_vec = vectorizer.transform([query])
    # Compute cosine similarity between the query vector and each movie plot vector
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get the indices of the top_n most similar movies
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations['Similarity'] = cosine_sim[top_indices]
    return recommendations

def main():
    print("Simple Content-Based Recommendation System")
    print("==========================================\n")
    
    # Load dataset from fixed location
    df = load_dataset()
    
    # Preprocess the text data from the 'Plot' column
    texts = preprocess_text(df, text_column='Plot')
    
    # Vectorize the plot texts into TF-IDF features
    vectorizer, tfidf_matrix = vectorize_texts(texts)
    
    # Prompt the user to enter a text description of their movie preferences
    query = input("Enter your movie preference description (e.g., 'I love thrilling action movies set in space, with a comedic twist.'): ")
    
    # Compute recommendations based on the user's query
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, df, top_n=5)
    
    # Always select the top recommendation (the one with the highest similarity score)
    top_rec = recommendations.iloc[0]
    
    print("\nTop Recommendation:")
    print(f"Title: {top_rec['Title']} | Similarity Score: {top_rec['Similarity']:.4f}")
    
    print("\nOther Recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"Title: {row['Title']} | Similarity Score: {row['Similarity']:.4f}")

if __name__ == "__main__":
    main()
