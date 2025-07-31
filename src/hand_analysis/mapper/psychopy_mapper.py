import logging
from typing import Tuple, Dict

import pandas as pd
import pyxations as pyx
from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.metrics.distance_calculation import calculate_distance
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
        print("data_path:", data_path, "\n\n")
        experiment = pyx.Experiment(dataset_path=data_path)
        print("experiment.subjects:", experiment.subjects.keys())

        subject_ids_list = [str(subject_id) for subject_id in experiment.subjects.keys()]

        # Convert subject_ids to strings and fill with zeros for BIDS standard
        subject_ids = [str(s).zfill(4) for s in subject_ids_list]

        experiment = self.map_experiment(experiment, subject_ids)

        return experiment

    def map_experiment(self, experiment: pyx.Experiment, subject_ids: List[str]) -> TMTExperiment:
        # TODO GUS: excluirlos
        # bad_mouse_subjects = {25}
        # too_young_subject = {32}
        # bad_data_visual_inspection = {75}

        tmt_subjects: Dict[str, TMTSubject] = {}

        for subject_id, subject in experiment.subjects.items():

            subject_id_int = int(subject_id)
            if subject_id_int not in config.SUBJECT_GROUP:
                logging.warning(f"Subject {subject_id} not found in the configuration file.")
                continue

            group = config.SUBJECT_GROUP[subject_id_int]

            logging.info(f"Processing subject: {subject_id}, group: {group}")
            try:
                # Load the first session data
                print(subject.sessions.keys())
                session = subject.sessions['experimento']
                session.load_data("eyelink")

                # Extract behavior data
                df_psychopy = session.load_behavior_data().iloc[1:-1].reset_index(drop=True)
                # Map (posArrayX, posArrayY) to a trial_id using config.trial_id_map
                df_psychopy["trial_id"] = (
                    df_psychopy
                    .set_index(["posArrayX", "posArrayY"])
                    .index.map(config.trial_id_map)
                    .astype("Int64")
                )

                subject_trials = self.map_to_tmt_trials(df_psychopy)

                # Discard trials whose `order_of_appearance` is in rejected_trials
                rejected_trials = self.get_rejected_trials()
                valid_subject_trials = [
                    trial for trial in subject_trials
                    if trial.order_of_appearance not in rejected_trials
                ]

                training_trials = [trial for trial in valid_subject_trials if
                                trial.order_of_appearance in self.get_training_trials_order()]

                # Dice que gus que los mapeemos
                tmt_subjects[subject.subject_id] = TMTSubject(
                    training_trials=training_trials,
                    testing_trials=valid_subject_trials,
                    target_radius=config.RADIUS_HEIGHT,
                    canvas_size=None,
                    personal_info=mock_personal_info(),
                    session_context=None
                )
                print(f"suj {subject.subject_id} mapped")
            except Exception as e:
                logging.error(f"{subject.subject_id}: {e}")
                raise e
                break
                #print(e.format_exc())

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
                    order_of_appearance=index,
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
