import pyxations as pyx

from src import config
from src.data_processing.filter_data_with_last_date import filter_data_with_last_date

data_dir = config.RAW_DATA_DIR
filtered_data_dir = config.FILTERED_DATA_DIR

filter_data_with_last_date(data_dir, filtered_data_dir)

pyx.dataset_to_bids(
    target_folder_path=data_dir,
    files_folder_path=filtered_data_dir,
    dataset_name=config.PYXATIONS_PATIENTS_DATASET_NAME,
)

pyx.compute_derivatives_for_dataset(
    bids_dataset_folder=config.PYXATIONS_PATIENTS_DATA_DIR,
    msg_keywords=["beginning_of_trial", "end_of_trial"],
    detection_algorithm="eyelink",
    start_msgs={"trial": ["beginning_of_trial"]},
    end_msgs={"trial": ["end_of_trial"]},
    overwrite=True,
    dataset_format="eyelink"
)
