import logging
from typing import List

import pandas as pd
import pyxations as pyx

from src import config


def gather_validation_data(experiment: pyx.Experiment) -> pd.DataFrame:
    """
    Retrieves validation data from the first session of each subject in the `experiment`.
    It loads Eyelink calibration data, filters lines that contain 'VALIDATION', and
    extracts POOR/GOOD/FAIR validation quality.

    Parameters
    ----------
    experiment : object
        An object that contains the list `experiment.subjects`. Each subject has
        an attribute `sessions` (list of sessions).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns such as 'line', 'validation_quality', 'subject_id',
        and 'group' for each subject (only the first session).
        If no data is found, returns an empty DataFrame with predefined columns.
    """
    records = []

    for subject in experiment.subjects:
        # Take only the first session available
        if not subject.sessions:
            continue  # No sessions for this subject, skip it
        session = subject.sessions[0]

        # Load calibration data from 'eyelink'
        session.load_data("eyelink")

        # Copy the calibration DataFrame to avoid mutating it directly
        df_calib = session.calib.copy()

        # Extract validation quality (POOR, GOOD, FAIR) from the 'line' column
        df_calib["validation_quality"] = df_calib["line"].str.extract(
            r"\b(POOR|GOOD|FAIR)\b",
            expand=False
        )

        # Keep only rows containing the word 'VALIDATION'
        df_calib = df_calib[df_calib["line"].str.contains(r"\bVALIDATION\b", regex=True)]

        # Add subject and group information
        df_calib["subject_id"] = subject.subject_id
        df_calib["group"] = config.SUBJECT_GROUP.get(int(subject.subject_id), "Unknown")

        # Drop rows without validation quality
        df_calib = df_calib.dropna(subset=["validation_quality"])

        # If the resulting DataFrame is not empty, store it
        if not df_calib.empty:
            records.append(df_calib)

    # If no valid records were found, return an empty DataFrame with columns defined
    if not records:
        return pd.DataFrame(columns=["line", "validation_quality", "subject_id", "group"])

    # Concatenate all partial DataFrames
    return pd.concat(records, ignore_index=True)


def gather_valid_subjects(experiment: pyx.Experiment, drop_subjects: bool = True) -> List[str]:
    """
        Retrieves the list of valid subjects based on specific calibration criteria:
          - Calib_index == 2
          - Average error (avg) < 1

        Parameters
        ----------
        experiment : object
            An object that contains `experiment.subjects`.

        Returns
        -------
        List[str]
            A list of subject_ids (as strings) that meet the validation criteria.
        """

    # 1) Get the validation DataFrame
    df_validation = gather_validation_data(experiment)
    df_filtered = df_validation.copy()
    if drop_subjects:
        # 2) Filter those with Calib_index == 2
        df_filtered = df_validation[df_validation["Calib_index"] == 2].copy()

        # 3) Drop duplicates by 'subject_id', keeping the last occurrence
        df_filtered = df_filtered.drop_duplicates(subset=["subject_id"], keep="last")

        # 4) Extract 'avg' from the 'line' column
        #    Pattern: "ERROR <number> avg."
        df_filtered["avg"] = df_filtered["line"].str.extract(
            r"ERROR\s+([\d.]+)\s+avg\.",
            expand=False
        ).astype(float, errors="ignore")

        # 5) Keep only rows where avg < 1
        df_filtered = df_filtered[df_filtered["avg"] < 1]

    logging.warning(
        f"different ids: {set(df_validation['subject_id'].unique()) - set(df_filtered['subject_id'].unique())}")

    # 6) Return the list of subject_ids
    return df_filtered["subject_id"].astype(str).tolist()

