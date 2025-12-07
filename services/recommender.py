import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from services.data_loader import YaleCourseData
from services.data_loader import RAW_TITLE_KEY, RAW_DESC_KEY # IMPORT CONSTANTS

# --- GLOBAL CONFIGURATION ---
DATA_PATH = "data/yale_courses_202503.json" 
TOP_N = 10
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2' 
SBERT_WEIGHT = 0.70 
TFIDF_WEIGHT = 0.30 

class CourseRecommender:
    """
    Core hybrid NLP model using TF-IDF and S-BERT.
    """
    def __init__(self):
        # 1. Load Data
        self.course_data = YaleCourseData(DATA_PATH)
        self.df = self.course_data.get_course_dataframe()
        self.texts = self.df['combined_text'].fillna('')
        
        # 2. Initialize Models
        self._load_nlp_tools()
        
        # 3. Create Vectors
        self.tfidf_matrix = self._vectorize_tfidf()
        self.sbert_matrix = self._vectorize_sbert()

    def _load_nlp_tools(self):
        """Loads spaCy and Sentence-BERT models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("FATAL ERROR: spaCy model 'en_core_web_sm' not found.")
            self.nlp = None
        
        try:
            print(f"Loading Sentence-BERT model: {SBERT_MODEL_NAME}...")
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as e:
            print(f"FATAL ERROR loading S-BERT model: {e}")
            self.sbert_model = None

    def _clean_job_input(self, text):
        """Cleans input text using spaCy for TF-IDF."""
        if not self.nlp:
            return text
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(tokens)
    
    def _vectorize_tfidf(self):
        """Vectorizes the combined course text using TF-IDF."""
        if self.df.empty:
            print("Warning: Cannot vectorize TF-IDF; DataFrame is empty.")
            return np.array([])

        print("Initializing TF-IDF Vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.85)
        cleaned_texts = [self._clean_job_input(text) for text in self.texts]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        print(f"TF-IDF Vectorization complete. Matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix

    def _vectorize_sbert(self):
        """Generates S-BERT embeddings for course text."""
        if self.df.empty or not self.sbert_model:
            print("Warning: Cannot vectorize S-BERT; Model not loaded or DataFrame is empty.")
            return np.array([])
            
        print("Generating S-BERT embeddings (This may take a moment)...")
        sbert_matrix = self.sbert_model.encode(self.texts, convert_to_tensor=False)
        print(f"S-BERT Encoding complete. Matrix shape: {sbert_matrix.shape}")
        return sbert_matrix

    def recommend(self, job_title: str, current_major: str):
        """
        Calculates hybrid course recommendations based on both S-BERT and TF-IDF similarity.
        """
        if self.tfidf_matrix.size == 0 or self.sbert_matrix.size == 0:
            return []

        # 1. Prepare and Vectorize Job Title (TF-IDF & S-BERT)
        cleaned_job_tfidf = self._clean_job_input(job_title)
        job_vector_tfidf = self.tfidf_vectorizer.transform([cleaned_job_tfidf])
        job_vector_sbert = self.sbert_model.encode([job_title], convert_to_tensor=False)

        # 2. Calculate Similarity Scores
        tfidf_scores = cosine_similarity(job_vector_tfidf, self.tfidf_matrix).flatten()
        sbert_scores = cosine_similarity(job_vector_sbert, self.sbert_matrix).flatten()
        
        # 3. Normalize Scores
        tfidf_max = tfidf_scores.max() if tfidf_scores.max() > 0 else 1
        sbert_max = sbert_scores.max() if sbert_scores.max() > 0 else 1
        
        normalized_tfidf = tfidf_scores / tfidf_max
        normalized_sbert = sbert_scores / sbert_max
        
        # 4. Calculate Hybrid Score
        hybrid_scores = (normalized_sbert * SBERT_WEIGHT) + (normalized_tfidf * TFIDF_WEIGHT)
        
        # 5. Apply Major Constraint Layer (Score Boost)
        recommendation_df = self.df.copy()
        recommendation_df['score'] = hybrid_scores
        
        major_boost_factor = 0.15 
        major_code = current_major.upper() 
        is_major_course = recommendation_df['subjectCode'].str.contains(major_code, na=False)
        recommendation_df.loc[is_major_course, 'score'] *= (1 + major_boost_factor)

        # 6. Get Top Recommendations
        recommended_courses = recommendation_df.sort_values(
            by='score', ascending=False
        ).head(TOP_N)
        
        # 7. Format Output (Uses the imported RAW_KEY constants!)
        results = recommended_courses[[
            'subjectCode', RAW_TITLE_KEY, RAW_DESC_KEY, 'score' 
        ]].rename(columns={'subjectCode': 'course_code', 
                           RAW_TITLE_KEY: 'title', 
                           RAW_DESC_KEY: 'description'}).to_dict('records')
        
        return results

# Example Usage (Test the recommender):
if __name__ == "__main__":
    recommender = CourseRecommender()
    
    test_job = "Financial modeling and risk management"
    test_major = "ECON" 
    
    print(f"\n--- Hybrid Recommendations for Job: {test_job}, Major: {test_major} ---")
    recommendations = recommender.recommend(test_job, test_major)
    
    for rank, course in enumerate(recommendations, 1):
        score = f"{course['score']:.4f}"
        print(f"Rank {rank}: {course['course_code']} - {course['title']} (Score: {score})")