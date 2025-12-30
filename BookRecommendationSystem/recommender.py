import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

class Recommender:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.books_path = os.path.join(data_dir, 'books.csv')
        self.ratings_path = os.path.join(data_dir, 'ratings.csv')
        self.books = None
        self.ratings = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
        self.load_data()

    def load_data(self):
        if os.path.exists(self.books_path):
            try:
                self.books = pd.read_csv(self.books_path, on_bad_lines='skip')
                # Clean and preprocess books data
                self.books['Name'] = self.books['Name'].fillna('')
                self.books['Authors'] = self.books['Authors'].fillna('')
                self.books['Publisher'] = self.books['Publisher'].fillna('')
                self.books['Rating'] = pd.to_numeric(self.books['Rating'], errors='coerce').fillna(0)
                
                # Create a soup of metadata for content-based filtering
                self.books['soup'] = self.books['Name'] + ' ' + self.books['Authors'] + ' ' + self.books['Publisher']
                
                # Compute TF-IDF matrix
                tfidf = TfidfVectorizer(stop_words='english')
                self.tfidf_matrix = tfidf.fit_transform(self.books['soup'])
                
                # Compute Cosine Similarity matrix (might be slow for very large datasets, but 1000 rows is fine)
                # If dataset is huge, we compute on demand or use approximate nearest neighbors
                if len(self.books) < 5000:
                    self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
                    self.indices = pd.Series(self.books.index, index=self.books['Name']).drop_duplicates()
                
            except Exception as e:
                print(f"Error loading books.csv: {e}")
                self.books = pd.DataFrame()
        else:
            print("books.csv not found.")
            self.books = pd.DataFrame()

        if os.path.exists(self.ratings_path):
            try:
                self.ratings = pd.read_csv(self.ratings_path, on_bad_lines='skip')
                # Map text ratings to numbers
                rating_map = {
                    'it was amazing': 5,
                    'really liked it': 4,
                    'liked it': 3,
                    'it was ok': 2,
                    'did not like it': 1
                }
                self.ratings['RatingNum'] = self.ratings['Rating'].map(rating_map)
            except Exception as e:
                print(f"Error loading ratings.csv: {e}")
                self.ratings = pd.DataFrame()
        else:
            print("ratings.csv not found.")
            self.ratings = pd.DataFrame()

    def get_top_books(self, n=12):
        if self.books.empty:
            return []
        # Sort by Rating and CountsOfReview (if available) to get popular high-rated books
        # Assuming CountsOfReview is numeric
        if 'CountsOfReview' in self.books.columns:
            self.books['CountsOfReview'] = pd.to_numeric(self.books['CountsOfReview'], errors='coerce').fillna(0)
            # Weighted rating could be better, but simple sort is fine for now
            top_books = self.books.sort_values(by=['Rating', 'CountsOfReview'], ascending=False).head(n)
        else:
            top_books = self.books.sort_values(by='Rating', ascending=False).head(n)
        
        return top_books.to_dict('records')

    def search_books(self, query):
        if self.books.empty:
            return []
        # Simple case-insensitive search
        results = self.books[self.books['Name'].str.contains(query, case=False, na=False)]
        return results.head(20).to_dict('records')

    def get_recommendations(self, title, n=6):
        if self.books.empty or self.cosine_sim is None:
            return []
        
        try:
            # Get the index of the book that matches the title
            # Handle potential multiple matches or exact match issues
            if title not in self.indices:
                # Try partial match
                matches = self.books[self.books['Name'].str.contains(title, case=False, regex=False)]
                if not matches.empty:
                    title = matches.iloc[0]['Name']
                else:
                    return []
            
            idx = self.indices[title]
            
            # If multiple books have the same name, take the first one
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

            # Get the pairwsie similarity scores of all books with that book
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Sort the books based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the n most similar books
            sim_scores = sim_scores[1:n+1]

            # Get the book indices
            book_indices = [i[0] for i in sim_scores]

            # Return the top most similar books
            return self.books.iloc[book_indices].to_dict('records')
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def get_recommendations_by_id(self, book_id, n=6):
        if self.books.empty or self.cosine_sim is None:
            return []
        
        try:
            # Find the index of the book with the given Id
            # Check if Id column exists
            if 'Id' not in self.books.columns:
                return []

            # Filter to find the row
            # We need the dataframe index (0 to len-1), not the 'Id' column value
            book_idx_series = self.books.index[self.books['Id'] == book_id]
            
            if book_idx_series.empty:
                # Try converting to int if it was passed as string
                try:
                    book_id_int = int(book_id)
                    book_idx_series = self.books.index[self.books['Id'] == book_id_int]
                except:
                    pass
            
            if book_idx_series.empty:
                return []

            idx = book_idx_series[0]

            # Get the pairwsie similarity scores of all books with that book
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Sort the books based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the n most similar books
            sim_scores = sim_scores[1:n+1]

            # Get the book indices
            book_indices = [i[0] for i in sim_scores]

            # Return the top most similar books
            return self.books.iloc[book_indices].to_dict('records')
        except Exception as e:
            print(f"Error getting recommendations by ID: {e}")
            return []

    def get_book_details(self, book_id):
        if self.books.empty:
            return None
        book = self.books[self.books['Id'] == book_id]
        if not book.empty:
            return book.iloc[0].to_dict()
        return None

    def get_all_books(self):
        if self.books.empty:
            return []
        # Return relevant columns
        return self.books[['Id', 'Name', 'Authors', 'Rating']].to_dict('records')

    def get_reviews(self, book_name, limit=3):
        if self.ratings.empty:
            return []
        # Filter ratings by book name
        # Assuming 'Name' in ratings.csv corresponds to 'Name' in books.csv
        book_reviews = self.ratings[self.ratings['Name'] == book_name]
        if book_reviews.empty:
            return []
        # Return the text ratings, limited to 'limit' entries
        return book_reviews['Rating'].head(limit).tolist()
