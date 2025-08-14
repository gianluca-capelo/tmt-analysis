import os
import shutil

from src import config
from src.data_processing.filter_data_with_last_date import filter_data_with_last_date
from src.data_processing.mock_subjects import mock_subjects
from src.data_processing.process import process_data
from src.data_processing.remove_placeholders import remove_placeholders
from src.hand_analysis.loader.analysis_loader import run_analysis
from src.model.datasetbuilder.build_datasets import build_datasets

# Remove all placeholders
no_placeholders_dir = config.RAW_WITH_NO_PLACEHOLDERS_DIR

remove_placeholders(output_path=no_placeholders_dir)

# Mock subjects in order to pyxations to work with the data
mock_subjects(source_path=no_placeholders_dir)

# Filter data to keep only the last date
filtered_by_date_dir = config.FILTERED_DATA_DIR

filter_data_with_last_date(no_placeholders_dir, filtered_by_date_dir)

# Delete the placeholders directory after filtering
if os.path.exists(no_placeholders_dir):
    shutil.rmtree(no_placeholders_dir)

# Process the data to convert it to BIDS format and compute derivatives
process_data(source_path=filtered_by_date_dir)

# Delete the filtered_by_date_dir after processing
if os.path.exists(filtered_by_date_dir):
    shutil.rmtree(filtered_by_date_dir)

run_analysis()

build_datasets()
