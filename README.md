🎓 Yale Career Path Recommender (Hybrid NLP)Project OverviewThis is an end-to-end AI application designed to recommend relevant Yale courses based on a user's desired job title and current major. The core of the system is a hybrid NLP model that calculates course relevance using both semantic (deep learning) and keyword-based (statistical) approaches, delivered via a high-performance FastAPI web service.Problem: Helping students bridge the gap between academic majors and specific career requirements.Solution: Matching user career input against course descriptions to provide ranked, personalized recommendations.✨ FeaturesHybrid Recommendation Model: Combines S-BERT for deep semantic understanding and TF-IDF for statistical keyword matching.Major Constraint Layer: Applies a scoring boost to courses relevant to the user's declared major (e.g., boosting 'ECON' courses if the user is an Economics major).FastAPI Backend: Provides a high-performance, asynchronous REST API endpoint for real-time recommendations.Robust Environment: Utilizes a dedicated Conda environment to ensure dependency stability across different Python versions.🏗️ ArchitectureThe project is structured into three main layers:Data Layer: Fetches raw course data from the Yale API and performs initial cleaning.Service Layer (/services): Houses the core business logic, including the CourseRecommender model.Application Layer (main.py): Runs the FastAPI web server to expose the recommendation logic.⚙️ Local Setup and InstallationPrerequisitesMiniconda or Anaconda (essential for creating the stable Python environment).Python 3.11 (The project is stabilized on Python 3.11 to support scientific libraries like scikit-learn).Step 1: Clone the RepositoryBashgit clone https://github.com/piquet5555/AI-career-path-recommender.git
cd AI-career-path-recommender
Step 2: Create and Activate the EnvironmentYou must use a Conda environment to avoid conflicts with scientific libraries. Run these commands in your Anaconda Prompt:Bash# Create the environment with Python 3.11
conda create -n yale-env python=3.11 -y

# Activate the environment
conda activate yale-env
Step 3: Install DependenciesWith the environment active, install all necessary libraries:Bash(yale-env) $ pip install -r requirements.txt
Step 4: Run the FastAPI ServerLaunch the web API using Uvicorn. This will also trigger the initial loading and vectorization of the AI model.Bash(yale-env) $ uvicorn main:app --reload
The API server will now be running at http://127.0.0.1:8000.🚀 UsageThe API is accessed via a single endpoint.1. Test via Interactive Documentation (Swagger UI)Open your web browser and navigate to:http://127.0.0.1:8000/docsUse the POST /recommendations/ endpoint to input the JSON body and test the model.2. Request Body ExampleThe API requires a job title and a major code.JSON{
  "job_title": "Quantitative risk manager with focus on machine learning",
  "current_major": "MATH"
}
3. Response Body ExampleThe response returns a ranked list of courses:JSON{
  "query": "...",
  "major": "MATH",
  "results": [
    {
      "course_code": "STAT 301",
      "title": "Introduction to Data Science",
      "description": "...",
      "score": 0.8521
    },
    // ... top 10 courses
  ]
}
🎯 Model DetailsComponentTechnologyRoleSemantic ModelSentence-BERT (S-BERT)Generates high-quality semantic embeddings for deep conceptual matching.Keyword ModelTF-IDFIdentifies important keywords, crucial for matching specific technical jargon.Hybrid ScoreWeighted AverageHybrid Score = (S-BERT Score * 0.70) + (TF-IDF Score * 0.30)Data Processingpandas, spacyCleaning, deduplication, and efficient data handling.📁 File Structure.
├── data/
│   └── yale_courses_202503.json # Course data
├── services/
│   ├── data_loader.py         # Handles data fetching and cleaning
│   └── recommender.py         # Core AI model logic (Hybrid NLP, Scoring)
├── frontend/                  # Simple HTML/JS client to test API
│   └── index.html
├── main.py                    # FastAPI entry point
└── requirements.txt