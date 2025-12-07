import pandas as pd
import os

# --- GLOBAL COLUMN NAME CONSTANTS (PERMANENT FIX) ---
# These must match the exact key names in your JSON file (yale_courses_202503.json)
RAW_TITLE_KEY = 'courseTitle'    
RAW_DESC_KEY = 'description'
# ----------------------------------------------------

class YaleCourseData:
    """
    Handles loading the Yale course data and prepares it for the recommender.
    Uses globally defined constants for robust column access.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self._load_data()

    def _load_data(self):
        """Loads the JSON course data into a Pandas DataFrame and prepares text columns."""
        if not os.path.exists(self.file_path):
            print(f"FATAL ERROR: Course data file not found at {self.file_path}")
            return

        try:
            # Load the JSON data
            self.df = pd.read_json(self.file_path)
            
            # Use the globally defined constants to create the combined text column
            if RAW_TITLE_KEY in self.df.columns and RAW_DESC_KEY in self.df.columns:
                 # Ensure columns are treated as strings and combine them
                 self.df['combined_text'] = self.df[RAW_TITLE_KEY].astype(str) + " " + self.df[RAW_DESC_KEY].astype(str)
                 print(f"Data Loaded: Successfully loaded {len(self.df)} courses.")
            else:
                 # Critical failure if the required columns aren't found
                 print(f"FATAL ERROR: Could not find required columns in JSON: '{RAW_TITLE_KEY}' or '{RAW_DESC_KEY}'.")
                 self.df = pd.DataFrame() 

        except Exception as e:
            print(f"FATAL ERROR: Failed to load data from JSON: {e}")
            self.df = pd.DataFrame() 

    def get_course_dataframe(self):
        """Returns the prepared DataFrame."""
        return self.df

    def get_majors_list(self):
        """Returns a list of all unique major codes (Subject Codes) for the UI input."""
        if self.df is not None and not self.df.empty:
            # Assumes the subject code is in a column named 'subjectCode'
            return sorted(self.df['subjectCode'].unique().tolist())
        return []

# Example Usage (for testing the module):
if __name__ == "__main__":
    DATA_PATH = "data/yale_courses_202503.json" 
    course_data = YaleCourseData(DATA_PATH)
    
    if not course_data.get_course_dataframe().empty:
        print("\n--- Sample Course Data ---")
        # Use the constants for display verification
        print(course_data.get_course_dataframe()[['subjectCode', RAW_TITLE_KEY, 'combined_text']].head())