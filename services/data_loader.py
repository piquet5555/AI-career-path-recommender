import pandas as pd
import os
import json      # NEW: For loading the config file
import requests  # NEW: For making API calls

# --- GLOBAL CONSTANTS ---
RAW_TITLE_KEY = 'courseTitle'
RAW_DESC_KEY = 'description'
CONFIG_PATH = 'data/fetch_config.json' # NEW: Path to the configuration file
# ------------------------

class YaleCourseData:
    """
    Handles loading the Yale course data. If a specific data file is provided, it loads that. 
    Otherwise, it fetches data iteratively from the API based on the configuration.
    """
    
    def __init__(self, file_path: str = None):
        # We will use the config file for fetching, ignoring the path parameter if it exists
        self.file_path = file_path 
        self.df = None
        self._load_data()

    def _load_data(self):
        """Loads course data iteratively from the API based on config, merges, and deduplicates."""
        
        if not os.path.exists(CONFIG_PATH):
            print(f"FATAL ERROR: Configuration file not found at {CONFIG_PATH}")
            return

        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

            all_courses = []
            
            # --- Iterate through all combinations of terms and schools ---
            for term in config['terms']:
                for school in config['schools']:
                    print(f"Fetching courses for Term: {term}, School: {school}...")
                    
                    # Construct API call parameters
                    api_params = {
                        'page': 'fose',         
                        'route': 'search',      
                        'col': school,          # Variable: School code from config
                        'termCode': term,       # Variable: Academic term code from config
                        'onlyListing': 'true'   
                    }
                    
                    # Make the API request
                    response = requests.get(config['base_api_url'], params=api_params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # The API returns a large JSON structure; courses are usually nested under 'results'
                        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
                             all_courses.extend(data['results'])
                        elif isinstance(data, list):
                             all_courses.extend(data)
                        else:
                             # Handle cases where the API returns a success code but no recognizable course array
                             print(f"Warning: No course results found in API response for {term}/{school}.")
                    else:
                        print(f"Warning: API call failed for {term}/{school} with status code {response.status_code}")

            # Convert list of all courses to DataFrame
            self.df = pd.DataFrame(all_courses)

            if not self.df.empty:
                # Drop duplicates based on title and description (essential for multi-term data)
                self.df.drop_duplicates(subset=[RAW_TITLE_KEY, RAW_DESC_KEY], keep='first', inplace=True)
                
                # Apply Text Combination Logic
                if RAW_TITLE_KEY in self.df.columns and RAW_DESC_KEY in self.df.columns:
                    self.df['combined_text'] = self.df[RAW_TITLE_KEY].astype(str) + " " + self.df[RAW_DESC_KEY].astype(str)
                    print(f"Data Loaded: Successfully merged and loaded {len(self.df)} unique courses.")
                else:
                    print("FATAL ERROR: Could not find required columns in API response after merging.")
                    self.df = pd.DataFrame()
            else:
                print("Data Loaded: No course data was successfully fetched across all terms/schools.")

        except Exception as e:
            print(f"FATAL ERROR: Failed to process data: {e}")
            self.df = pd.DataFrame()

    # --- Getter Methods ---
    def get_course_dataframe(self):
        """Returns the prepared DataFrame."""
        return self.df

    def get_majors_list(self):
        """Returns a list of all unique major codes (Subject Codes) for the UI input."""
        if self.df is not None and not self.df.empty and 'subjectCode' in self.df.columns:
            return sorted(self.df['subjectCode'].unique().tolist())
        return []

# Example Usage (for testing the module):
if __name__ == "__main__":
    # Note: When running this locally, you must first create fetch_config.json!
    course_data = YaleCourseData()
    
    if not course_data.get_course_dataframe().empty:
        print("\n--- Sample Course Data ---")
        print(course_data.get_course_dataframe()[['subjectCode', RAW_TITLE_KEY, 'combined_text']].head())