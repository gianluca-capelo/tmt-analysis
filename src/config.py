# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_FOLDER = os.path.join(BASE_DIR, "data", "patients_data_pyxations")
HAND_ANALYSIS_FOLDER = os.path.join(BASE_DIR, "data", "hand_analysis")
HAND_ANALYSIS_CSV = os.path.join(BASE_DIR, "data", "hand_analysis", "metrics.csv")

METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")
DATA_FOLDER = os.path.join(BASE_DIR, "data", "patients_data_pyxations_derivatives")

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

# TMT parameters
CORRECT_THRESHOLD = None
CONSECUTIVE_POINTS = 5
