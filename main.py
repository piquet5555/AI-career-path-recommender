import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # FIX 1: Import CORS
from pydantic import BaseModel
from services.recommender import CourseRecommender

# --- 1. INITIALIZATION ---

# Initialize the FastAPI application
app = FastAPI(
    title="Yale Career Path Recommender API",
    description="Provides course recommendations based on job title and current major.",
    version="1.0.0"
)

# FIX 2: Define origins list for CORS before usage
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null", # Allows requests from files opened directly in the browser (file:///)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommender model globally
# This ensures the model (TF-IDF/S-BERT matrices) loads only once when the server starts
try:
    recommender = CourseRecommender()
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize CourseRecommender: {e}")
    recommender = None


# --- 2. DATA SCHEMA (Pydantic) ---

# Define the expected format for user input data (the request body)
class RecommendationRequest(BaseModel):
    job_title: str
    current_major: str


# --- 3. API ENDPOINT ---

@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest):
    """
    Receives a job title and major, and returns a list of top N courses 
    ranked by similarity score.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model service is unavailable.")
    
    try:
        # Check if the input major is valid based on the loaded data
        valid_majors = recommender.course_data.get_majors_list()
        major_input = request.current_major.upper()

        if major_input not in valid_majors:
            print(f"Warning: Major '{major_input}' not found. Using generic recommendation.")


        # Call the core recommendation logic
        recommendations = recommender.recommend(
            job_title=request.job_title,
            current_major=major_input
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No relevant courses found for the given criteria.")

        return {
            "query": request.job_title,
            "major": major_input,
            "results": recommendations
        }

    except Exception as e:
        print(f"Unhandled error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during processing.")


# --- 4. SERVER RUNNER (For Local Development) ---

if __name__ == "__main__":
    print("Run the application using: uvicorn main:app --reload")