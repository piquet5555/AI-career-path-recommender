# 🏛️ Yale Career Path Recommender (Hybrid NLP/Project Overview)

This is an end-to-end AI application designed to recommend relevant Yale courses based on a user's desired job title and current major. The core of the system is a **hybrid NLP model** that calculates course relevance using both semantic (deep learning) and keyword-based (statistical) approaches, delivered via a high-performance **FastAPI web service**.

**Problem:** Helping students bridge the gap between academic majors and specific career requirements.

**Solution:** Matching user career input against course descriptions to provide ranked, personalized recommendations.

---

## ✨ Features

* **Hybrid Recommendation Model:** Combines **S-BERT** for deep semantic understanding and **TF-IDF** for statistical keyword matching.
* **Major Constraint Layer:** Applies a scoring boost to courses relevant to the user's declared major (e.g., boosting ECON courses if the user is an economics major).
* **FastAPI Backend:** Provides a high-performance, asynchronous **REST API** endpoint for real-time recommendations.
* **Robust Environment:** Utilizes a dedicated **Conda environment** to ensure dependency stability across different Python versions.

---

## 🏗️ Architecture (3 Main Layers)

The architecture is structured into three main layers:

1.  **Data Layer:** Fetches raw course data from the Yale API and performs initial cleaning. (Files: `services/data_loader.py`)
2.  **Service Layer (/services):** Houses the core business logic, including the `CourseRecommender` model.
3.  **Application Layer (`main.py`):** Runs the FastAPI web server to expose the recommendation logic.

---

## ⚙️ Local Setup and Installation

### Prerequisites

You must have **Miniconda** or **Anaconda** installed (essential for creating the stable Python environment). Python 3.11 is the project's stable base.

### Step 1: Clone the Repository

Clone the project to your local machine using Git Bash or your terminal:

```bash
git clone [https://github.com/piquet5555/AI-career-path-recommender.git](https://github.com/piquet5555/AI-career-path-recommender.git)
cd AI-career-path-recommender