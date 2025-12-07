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

# Set timeout to 60 seconds to prevent read timeout errors
REQUEST_TIMEOUT = 60 

# List of Yale College subject codes (expand this list for full coverage)
SUBJECT_CODES = [
    "ACCT","AFAM","AFST","AKKD","AMST","AMTH","ANTH","APHY","ARBC","ARCG","ARCH",
    "ARMN","ART","ASL","ASTR","BENG","BIOL","BRST","BURM","CENG","CGSC","CHEM",
    "CHER","CHLD","CHNS","CLCV","CLSS","CPAR","CPLT","CPSC","CSEC","CSLI","CZEC",
    "DEVN","DRST","DUTC","EALL","EAST","EEB","EGYP","ENAS","ENGL","ENRG","ENVE",
    "EPS","ER&M","EVST","FILM","FNSH","FREN","GLBL","GMAN","GREK","HEBR","HELN",
    "HGRN","HIST","HLTH","HMRT","HNDI","HSAR","HSHM","HUMS","INDN","ITAL","JAPN",
    "JDST","KHMR","KREN","LAST","LATN","LING","MATH","MB&B","MCDB","MENG","MMES",
    "MTBT","MUSI","NAVY","NELC","NSCI","OTTM","PERS","PHIL","PHYS","PLSC","PLSH",
    "PNJB","PORT","PSYC","RLST","ROMN","RSEE","RUSS","SBCR","SKRT","SLAV","SNHL",
    "SOCY","SAST","SWAH","SPAN","SPEC","TKSH","TAML","TBTN","TDPS","UKRN","URBN",
    "USAF","VIET","WLOF","WGSS","YORU","ZULU"

]

# Changed to a known recent term (Fall 2025) for testing purposes
TERM_CODE = "202503" 
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
            # Send GET request with the increased timeout
            response = requests.get(YALE_API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            
            # --- API Key/Status Check ---
            if response.status_code == 401:
                print(f"FATAL ERROR: 401 Unauthorized for {subject}. Check your YALE_API_KEY in the .env file.")
                return False # Stop further processing if key is rejected

            # Raise HTTPError for other bad responses (4xx or 5xx)
            response.raise_for_status() 

            data = response.json()

            # Check if the response is already a list (which is the list of courses itself)
            if isinstance(data, list):
                course_list = data
            # Otherwise, assume the course list is nested under a 'course' key (our original assumption)
            else:
                course_list = data.get('course', [])
            
            if course_list:
                all_courses.extend(course_list)
                print(f"  > Fetched {len(course_list)} courses for {subject}")
            else:
                print(f"  > No courses found for {subject}")

        except requests.exceptions.RequestException as e:
            # Catch network errors, connection failures, and final timeouts
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