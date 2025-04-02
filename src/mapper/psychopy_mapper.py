import logging
from typing import Tuple, Dict

import pandas as pd
import pyxations as pyx
from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.metrics import calculate_distance
from neurotask.tmt.model.tmt_model import *

from src import config


def mock_personal_info():
    return SubjectPersonalInformation(
        birthdate=datetime(1990, 1, 1),
        gender="M",
        education_level="University",
        nationality="Argentine",
        residence_country="Argentina",
        residence_region="Buenos Aires"
    )


class PsychopyTMTMapper(TMTMapper):

    def map(self, data_path: str, metadata_path: Optional[str] = None) -> TMTExperiment:

        # TODO GIAN: Pendiente mapear metadata

        experiment = self._load_pyxations_experiment(data_path)

        valid_subjects = self.gather_valid_subjects(experiment, drop_subjects=True)

        # Convert subject_ids to strings and fill with zeros for BIDS standard
        subject_ids = [str(s).zfill(4) for s in valid_subjects]

        experiment = self.map_experiment(experiment, subject_ids)

        return experiment

    def _load_pyxations_experiment(self, data_path):
        return pyx.Experiment(dataset_path=data_path)

    def gather_valid_subjects(self, experiment: pyx.Experiment, drop_subjects: bool = True) -> List[str]:
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
        df_validation = self.gather_validation_data(experiment)
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

        logging.warning(f"different ids: {set(df_validation['subject_id'].unique()) - set(df_filtered['subject_id'].unique())}")


        # 6) Return the list of subject_ids
        return df_filtered["subject_id"].astype(str).tolist()

    def gather_validation_data(self, experiment: pyx.Experiment) -> pd.DataFrame:
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

    def map_experiment(self, experiment: pyx.Experiment, subject_ids: List[str]) -> TMTExperiment:
        # TODO GUS: excluirlos
        # bad_mouse_subjects = {25}
        # too_young_subject = {32}
        # bad_data_visual_inspection = {75}

        tmt_subjects: Dict[str, TMTSubject] = {}

        for subject_id in subject_ids:

            subject_id_int = int(subject_id)
            if subject_id_int not in config.SUBJECT_GROUP:
                logging.warning(f"Subject {subject_id} not found in the configuration file.")
                continue

            group = config.SUBJECT_GROUP[subject_id_int]

            logging.info(f"Processing subject: {subject_id}, group: {group}")

            # Load the first session data
            experiment_index = subject_id_int - 1
            session = experiment[experiment_index].sessions[0]
            session.load_data("eyelink")

            rejected_trials = self.get_rejected_trials()
            session = self.filter_fixations(session, rejected_trials=rejected_trials)

            # Extract behavior data
            df_psychopy = session.behavior_data.iloc[1:-1].reset_index(drop=True)
            # Map (posArrayX, posArrayY) to a trial_id using config.trial_id_map
            df_psychopy["trial_id"] = (
                df_psychopy
                .set_index(["posArrayX", "posArrayY"])
                .index.map(config.trial_id_map)
                .astype("Int64")
            )

            subject_trials = self.map_to_tmt_trials(df_psychopy)

            # Discard trials whose `order_of_appearance` is in rejected_trials
            valid_subject_trials = [
                trial for trial in subject_trials
                if trial.order_of_appearance not in rejected_trials
            ]

            training_trials = [trial for trial in valid_subject_trials if
                               trial.order_of_appearance in self.get_training_trials_order()]

            # Dice que gus que los mapeemos
            tmt_subjects[subject_id] = TMTSubject(
                training_trials=training_trials,
                testing_trials=valid_subject_trials,
                target_radius=config.RADIUS_HEIGHT,
                canvas_size=None,
                personal_info=mock_personal_info(),
                session_context=None
            )

        return TMTExperiment(subjects=tmt_subjects)

    def get_rejected_trials(self):
        training_trials = self.get_training_trials_order()
        non_trial = [-1]
        return training_trials + non_trial

    def get_training_trials_order(self):
        return [0, 1]

    def map_to_tmt_trials(self, data: pd.DataFrame) -> List[TMTTrial]:
        trials = []
        for index, row in data.iterrows():
            if self.is_valid_row(row):
                stimuli, trial_type = self.map_to_stimuli_and_trial_type(row)

                cursor_trial = self.map_to_cursor_trail(row)
                first_hit = calculate_first_target_hit(cursor_trial, stimuli, config.RADIUS_HEIGHT)
                trial = TMTTrial(
                    order_of_appearance=row["trial_id"],
                    id=row["trial_id"],
                    rt=self.calculate_rt(row),
                    stimuli=stimuli,
                    cursor_trail=cursor_trial,
                    trial_type=trial_type,
                    with_custom_start=True,
                    start=first_hit
                )

                trials.append(trial)

            else:
                logging.warning(f"Invalid row: {row}")

        return trials

    def is_valid_row(self, row: pd.Series) -> bool:
        return (
                pd.notnull(row["trialMouse.x"])
                and pd.notnull(row["trialMouse.y"])
                and pd.notnull(row["trialMouse.time"])
                and pd.notnull(row["RT en en el trial"])
                and pd.notnull(row["trial_id"])
        )

    def calculate_rt(self, row):
        # seconds to milliseconds
        return int(row["RT en en el trial"] * 1000)

    def map_to_cursor_trail(self, row: pd.Series) -> List[CursorInfo]:

        mouse_x = [
            float(x.strip("[]")) for x in row["trialMouse.x"].split(",")
        ]
        mouse_y = [
            float(y.strip("[]")) for y in row["trialMouse.y"].split(",")
        ]
        mouse_time = [
            float(t.strip("[]")) for t in row["trialMouse.time"].split(",")
        ]

        return [
            CursorInfo(position=Coordinate(x, y), time=t)
            for x, y, t in zip(mouse_x, mouse_y, mouse_time)
        ]

    def map_to_stimuli_and_trial_type(self, row: pd.Series) -> Tuple[List[TMTTarget], TrialType]:

        pos_x = row["posArrayX"].split(",") if pd.notnull(row["posArrayX"]) else []
        pos_y = row["posArrayY"].split(",") if pd.notnull(row["posArrayY"]) else []
        targets = row["Targets"].split(",") if pd.notnull(row["Targets"]) else []

        stimuli = []
        for x, y, target in zip(pos_x, pos_y, targets):
            try:
                position = Coordinate(float(x.strip("[]")), float(y.strip("[]")))
                stimuli.append(TMTTarget(content=target, position=position))
            except ValueError:
                logging.exception(f"Error processing stimuli coordinates: {x}, {y}")

        return stimuli, self.determine_trial_type(targets)

    def determine_trial_type(self, targets: List[str]) -> TrialType:
        return TrialType.PART_B if "A" in targets else TrialType.PART_A

    def filter_fixations(self, session, min_fix_dur=50, max_fix_dur=1000, rejected_trials=None):
        """Filter fixation data by duration and bad data exclusion.

        Args:
            session: The session object.
            min_fix_dur: The minimum fixation duration (in ms).
            max_fix_dur: The maximum fixation duration (in ms).
            rejected_trials: A list of trial numbers to exclude from the fixation data.
        """
        # TODO GUS: chequear si es necesario este filtrado

        fixations = session.fix

        if rejected_trials:
            fixations = fixations[~fixations["trial_number"].isin(rejected_trials)]

        filtered_fixations = fixations.loc[
            (fixations["duration"] > min_fix_dur)
            & (fixations["duration"] < max_fix_dur)
            & (fixations["bad"] == False)
            ]

        session.fix = filtered_fixations

        return session


def calculate_first_target_hit(cursor_trail: List[CursorInfo], stimuli: List[TMTTarget], target_radius: float) -> \
        Optional[CursorInfo]:
    """
    Calculates the index of the first target hit in the cursor trail.

    Parameters:
    - cursor_trail: List of CursorInfo objects.
    - stimuli: List of TMTTarget objects.
    - target_radius: float, the radius of the targets.

    Returns:
    - index: int, the index of the first target hit in the cursor trail.
    """
    first_target = stimuli[0]

    for cursor_info in cursor_trail:
        distance = calculate_distance(cursor_info.position, first_target.position)
        is_touched = distance <= target_radius
        if is_touched:
            return cursor_info

    logging.warning("No target was hit in the cursor trail.")

    return None
