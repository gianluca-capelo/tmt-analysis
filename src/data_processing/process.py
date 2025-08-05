import pyxations as pyx

from src import config
from src.data_processing.filter_data_with_last_date import filter_data_with_last_date

def process_data(source_path):

    pyx.dataset_to_bids(
        target_folder_path=config.DATA_DIR,
        files_folder_path=source_path,
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


if __name__ == "__main__":
    process_data()
    print("Data processing completed successfully.")