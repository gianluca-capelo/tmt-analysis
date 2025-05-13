import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config as config_file
from src.config import RANDOM_STATE
from src.hand_analysis.dataset_split.eval_train_split import split_subjectwise_evaluation_set_stratified
from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.loader.load_last_split import get_run_configuration
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_configuration_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis(random_state: int,
                  eval_size: float,
                  split: bool,
                  old_split_config_date: str | None
                  ) -> pd.DataFrame:
    """
    Run the analysis pipeline, save results and metadata, and optionally split the dataset.
    """
    if split is True and eval_size is None:
        raise ValueError("eval_size must be set when split is True.")

    if old_split_config_date is not None and split is True:
        raise ValueError("Cannot use old_split_config_date when split is True.")

    # 1) Generate a timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = create_run_folder(timestamp)
    logging.info("Created run directory %s", run_dir)

    # 2) Execute analysis
    analysis = run_analysis_with_configuration_parameters(run_dir)
    metrics_df = analysis.get_metrics_dataframe()
    metrics_df = add_metadata_to_metrics(metrics_df)
    valid_metrics_df = filter_invalid_subjects(metrics_df)

    # 3) Split the dataset if required
    train_subject_ids, eval_subject_ids = get_split_ids(eval_size, old_split_config_date, random_state, split,
                                                        valid_metrics_df)

    # 4) Save results and metadata
    save_results(valid_metrics_df, run_dir, timestamp, random_state, eval_size, train_subject_ids, eval_subject_ids,
                 split,
                 old_split_config_date)

    return valid_metrics_df


def get_split_ids(eval_size, old_split_config_date, random_state, split, valid_metrics_df):
    if split:
        train_subject_ids, eval_subject_ids = split_subjectwise_evaluation_set_stratified(
            valid_metrics_df,
            eval_size=eval_size,
            random_state=random_state
        )
        logging.info(
            "Completed data split with test_size=%s, random_state=%s",
            eval_size, random_state
        )
    elif old_split_config_date:
        logging.info(
            "Loading old split configuration from %s",
            old_split_config_date
        )
        old_split_config_dir = Path(config_file.HAND_ANALYSIS_FOLDER) / old_split_config_date
        old_configuration = get_run_configuration(old_split_config_dir)
        train_subject_ids = old_configuration.get("train_subject_ids")
        eval_subject_ids = old_configuration.get("eval_subject_ids")
    else:
        logging.info("No split performed, using all subjects for training.")
        # Use all subjects for training
        train_subject_ids = valid_metrics_df['subject_id'].unique().tolist()
        eval_subject_ids = []
    return train_subject_ids, eval_subject_ids,


def create_run_folder(timestamp: str) -> Path:
    """
    Create a timestamped directory under config_file.HAND_ANALYSIS_FOLDER.
    """
    root_dir = Path(config_file.HAND_ANALYSIS_FOLDER)
    run_dir = root_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_info() -> dict:
    """
    Retrieve current Git branch, commit SHA, message, and dirty state.
    """
    meta: dict = {}
    try:
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_msg"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_dirty"] = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"]
        ) != 0

    except subprocess.CalledProcessError as e:
        logging.warning("Failed to retrieve Git metadata: %s", e)
        meta["git_error"] = str(e)
    return meta


def save_results(
        df: pd.DataFrame,
        run_dir: Path,
        timestamp: str,
        random_state: int,
        test_size: float,
        train_subject_ids,
        eval_subject_ids,
        split: bool,
        old_split_config_date: str
) -> None:
    """
    Save the metrics DataFrame and configuration metadata to the run directory.
    """
    run_config = {
        "timestamp": timestamp,
        "correct_targets_minimum": config_file.CORRECT_THRESHOLD,
        "cut_criteria": config_file.CUT_CRITERIA,
        "consecutive_points": config_file.CONSECUTIVE_POINTS,
        "calculate_crosses": config_file.CALCULATE_CROSSES,
        "random_state": random_state,
        "test_size": test_size,
        "raw_data_dir": config_file.RAW_DATA_DIR,
        "metadata_path": config_file.METADATA_CSV,
        "train_subject_ids": train_subject_ids,
        "eval_subject_ids": eval_subject_ids,
        "is_new_split": split,
        "old_split_config_date": old_split_config_date,
    }
    run_config.update(git_info())

    config_path = run_dir / "configuration.json"
    try:
        with config_path.open("w") as f:
            json.dump(run_config, f, indent=2)
        logging.info("Saved configuration to %s", config_path)
    except IOError as e:
        logging.error("Failed to write configuration file: %s", e)

    # Write results CSV
    data_path = run_dir / "analysis.csv"
    try:
        df.to_csv(data_path, index=False)
        logging.info("Saved analysis results to %s", data_path)
    except IOError as e:
        logging.error("Failed to write analysis CSV: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run hand analysis.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Perform a split of the dataset into train and evaluation sets."
    )
    args = parser.parse_args()
    split = args.split
    eval_size = None
    if split:
        eval_size = float(input("Enter evaluation test size (e.g., 0.2): "))

    #get old split config date
    old_split_config_date = None
    parser.add_argument(
        "--old_split_config_date",
        type=str,
        help="Date of the old split configuration (YYYY-MM-DD_HH-MM-SS)."
    )
    args = parser.parse_args()
    old_split_config_date = args.old_split_config_date




    load_analysis(RANDOM_STATE, eval_size, split, old_split_config_date)


if __name__ == "__main__":
    main()
