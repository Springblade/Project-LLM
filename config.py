import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.3
MAX_TOKENS = 2048

# Dataset Configuration
DATASET_PATH = "Path/TestDataset.json"
# RESULTS_OUTPUT_PATH = "results/evaluation_results.csv"
