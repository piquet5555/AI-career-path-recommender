import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

# --- CONFIGURATION ---
YALE_API_ENDPOINT = os.getenv("YALE_API_ENDPOINT")
YALE_API_KEY = os.getenv("YALE_API_KEY")

# List of common Yale College subject codes (expand this list for full coverage)
# You need a comprehensive list of all subjects offered at Yale College (YC)
# Example of a partial list:
SUBJECT_CODES = [
    "AFAM", "AMST", "ANTH", "ARBC", "ARCG", "ASTR", "BENG", "BIOL",
    "CPSC", "ECON", "ENGL", "HIST", "MATH", "MUSI", "PHIL", "PLSC",
    "PSYC", "SOCI", "STAT", "YALE" # YALE includes courses not specific to a dept.
]
TERM_CODE = "202603"  # Example: Fall 2026. Adjust to current/target term.
SCHOOL_CODE = "YC"    # Yale College

def fetch_yale_course_data():
    """
    Fetches course data from the Yale API by looping through subject codes
    and saves the aggregated data to a local JSON file.
    """
    if not YALE_API_ENDPOINT or not YALE_API_KEY:
        print("ERROR: YALE_API_ENDPOINT or YALE_API_KEY not found in environment variables.")
        return False

    all_courses = []
    print(f"Starting data fetch for term {TERM_CODE}...")

    for subject in SUBJECT_CODES:
        params = {
            'apikey': YALE_API_KEY,
            'subjectCode': subject,
            'termCode': TERM_CODE,
            'school': SCHOOL_CODE,
            'mode': 'json'
        }

        try:
            # Send GET request to the Yale API
            response = requests.get(YALE_API_ENDPOINT, params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()

            # The API response structure may require adjustment here.
            # Assuming the course list is under a 'course' key in the response:
            course_list = data.get('course', []) 
            
            if course_list:
                all_courses.extend(course_list)
                print(f"  > Fetched {len(course_list)} courses for {subject}")
            else:
                print(f"  > No courses found for {subject}")

        except requests.exceptions.RequestException as e:
            print(f"ERROR fetching data for {subject}: {e}")
            continue

    # --- Data Processing and Saving ---
    if not all_courses:
        print("Completed fetch, but no courses were retrieved.")
        return False

    # Convert to DataFrame for easy cleaning and inspection
    df = pd.DataFrame(all_courses)
    
    # Save the raw data to the data/ directory
    output_path = f"data/yale_courses_{TERM_CODE}.json"
    df.to_json(output_path, orient='records', indent=4)

    print(f"\nSUCCESS: Total {len(df)} courses aggregated and saved to {output_path}")
    return True

if __name__ == "__main__":
    # Execute the fetch function when running this script directly
    fetch_yale_course_data()