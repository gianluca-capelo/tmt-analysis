import pandas as pd

from src import config
from src.metadata.metadata_process import convert_subject_id


def filter_invalid_subjects(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid subjects from the list of subjects.
    """
    valid_subjects = remove_manual_rejected_subjects(metrics_df)

    return valid_subjects


def remove_manual_rejected_subjects(metrics_df):
    rejected_subjects = config.MANUAL_REJECTED_SUBJECTS
    rejected_subjects = [convert_subject_id(subject_id) for subject_id in rejected_subjects]
    valid_subjects = metrics_df[~metrics_df['subject_id'].isin(rejected_subjects)]
    return valid_subjects