import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from services.data_loader import YaleCourseData
from sklearn.metrics.pairwise import cosine_similarity

# NEW IMPORTS FOR OOM FIX
import pickle
import os

# --- Constants ---
TOP_N = 10
SBERT_WEIGHT = 0.70
TFIDF_WEIGHT = 0.30
RAW_TITLE_KEY = 'courseTitle'
RAW_DESC_KEY = 'courseDescription'

# NEW CONSTANTS FOR OOM FIX: File paths for saved matrices
SBERT_MATRIX_PATH = 'data/sbert_matrix.pkl'
TFIDF_VECTORIZER_PATH = 'data/tfidf_vectorizer.pkl'
TFIDF_MATRIX_PATH = 'data/tfidf_matrix.pkl'


class CourseRecommender:
    """
    A hybrid recommendation system combining Sentence-BERT (semantic) and 
    TF-IDF (keyword) similarity for course recommendations.
    """
    
    def __init__(self, data_file: str = 'data/yale_courses_202503.json'):
        # 1. Load Data
        self.df = YaleCourseData(data_file).get_course_dataframe()
        self.course_data = YaleCourseData(data_file)
        
        # --- OOM FIX: Check for pre-calculated files ---
        # If the matrices exist (meaning we ran the script locally once), load them.
        if os.path.exists(SBERT_MATRIX_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            print("Loading pre-calculated matrices from disk to conserve memory...")
            self.load_matrices_from_disk()
            self.is_ready = True
            return
            
        # If files don't exist (e.g., first run on a new machine), calculate and save.
        print("Calculating and saving matrices for the first time...")
        
        # 2. Initialize Models (Heavy computation)
        self.initialize_sbert()
        self.initialize_tfidf()
        
        # 3. Save Matrices (For subsequent memory-efficient loads)
        self.save_matrices_to_disk()
        self.is_ready = True

    # --- Initialization Methods (Called only if files don't exist) ---
    def initialize_sbert(self):
        print("Loading Sentence-BERT model: all-MiniLM-L6-v2...")
        # Note: This loads a very large language model into memory temporarily
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Generating S-BERT embeddings (This may take a moment)...")
        # Generate and store embeddings for every course description
        self.course_embeddings = self.sbert_model.encode(self.df['combined_text'].tolist(), show_progress_bar=False)
        print(f"S-BERT Encoding complete. Matrix shape: {self.course_embeddings.shape}")
        # Clean up the large model from memory after use (optional but helpful)
        del self.sbert_model 

    def initialize_tfidf(self):
        print("Initializing TF-IDF Vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])
        print(f"TF-IDF Vectorization complete. Matrix shape: {self.tfidf_matrix.shape}")

    # --- NEW: Save and Load Methods for OOM Fix ---
    def save_matrices_to_disk(self):
        # Save S-BERT Matrix
        with open(SBERT_MATRIX_PATH, 'wb') as f:
            pickle.dump(self.course_embeddings, f)
        
        # Save TF-IDF Vectorizer and Matrix
        with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(TFIDF_MATRIX_PATH, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        print("Matrices successfully saved to disk.")

    def load_matrices_from_disk(self):
        # Load S-BERT Matrix
        with open(SBERT_MATRIX_PATH, 'rb') as f:
            self.course_embeddings = pickle.load(f)
        
        # Load TF-IDF Vectorizer and Matrix
        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        print("Matrices loaded from disk.")
    # ----------------------------------------------

    # --- Core Recommendation Logic ---
    def recommend(self, job_title: str, current_major: str):
        if not self.is_ready:
            raise RuntimeError("Recommender model is not fully initialized.")

        # 1. Combine Job Title and Major into a Query String
        query = f"{job_title} {current_major}"

        # 2. Calculate S-BERT Similarity Score
        # NOTE: S-BERT model is NOT needed here if matrices are pre-calculated.
        # We temporarily re-load the model just to encode the single query string.
        sbert_query_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = sbert_query_model.encode([query])
        sbert_scores = cosine_similarity(query_embedding, self.course_embeddings).flatten()
        del sbert_query_model # Clean up after use

        # 3. Calculate TF-IDF Similarity Score
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        # 4. Hybrid Scoring
        hybrid_scores = (sbert_scores * SBERT_WEIGHT) + (tfidf_scores * TFIDF_WEIGHT)

        # 5. Apply Major Constraint Layer (Score Boost)
        recommendation_df = self.df.copy()
        recommendation_df['score'] = hybrid_scores
        
        major_boost_factor = 0.15 
        major_code = current_major.upper() 
        is_major_course = recommendation_df['subjectCode'].str.contains(major_code, na=False)
        recommendation_df.loc[is_major_course, 'score'] *= (1 + major_boost_factor)
        
        # 6. DEDUPLICATION STEP: Drop duplicates based on course content
        recommended_courses = recommendation_df.sort_values(
            by='score', ascending=False
        ).drop_duplicates(
            subset=[RAW_TITLE_KEY, RAW_DESC_KEY], 
            keep='first' 
        )

        # 7. Get Top N Recommendations
        recommended_courses = recommended_courses.head(TOP_N)
        
        # 8. Format Output
        results = recommended_courses[[
            'subjectCode', RAW_TITLE_KEY, RAW_DESC_KEY, 'score' 
        ]].rename(columns={'subjectCode': 'course_code', 
                           RAW_TITLE_KEY: 'title', 
                           RAW_DESC_KEY: 'description'}).to_dict('records')
        
        return results