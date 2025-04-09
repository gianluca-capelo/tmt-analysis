import pandas as pd

from src import config


def filter_invalid_subjects(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid subjects from the list of subjects.
    """
    rejected_subjects = config.MANUAL_REJECTED_SUBJECTS
    valid_subjects = metrics_df[~metrics_df['subject_id'].isin(rejected_subjects)]

    return valid_subjects