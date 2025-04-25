import json
import subprocess
from datetime import datetime
from pathlib import Path
from venv import logger

import pandas as pd

from src import config as config_file
from src.config import RANDOM_STATE
from src.hand_analysis.dataset_split.eval_train_split import split_subjectwise_evaluation_set_stratified
from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_configuration_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis(random_state: int,
                  test_size: float,
                  split: bool = False) -> pd.DataFrame:
    """
    Run the analysis pipeline, save results and metadata, and optionally split the dataset.
    """
    # 1) Generate a timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = create_run_folder(timestamp)
    logger.info("Created run directory %s", run_dir)

    # 2) Execute analysis
    analysis = run_analysis_with_configuration_parameters(run_dir)
    metrics_df = analysis.get_metrics_dataframe()
    metrics_df = add_metadata_to_metrics(metrics_df)
    valid_metrics_df = filter_invalid_subjects(metrics_df)

    # 3) Save results and metadata
    save_results(valid_metrics_df, run_dir, timestamp, random_state, test_size)

    # 4) Optional stratified split
    if split:
        split_subjectwise_evaluation_set_stratified(
            valid_metrics_df,
            test_size=test_size,
            random_state=random_state
        )
        logger.info(
            "Completed data split with test_size=%s, random_state=%s",
            test_size, random_state
        )

    return valid_metrics_df


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
        logger.warning("Failed to retrieve Git metadata: %s", e)
        meta["git_error"] = str(e)
    return meta


def save_results(
        df: pd.DataFrame,
        run_dir: Path,
        timestamp: str,
        random_state: int,
        test_size: float
) -> None:
    """
    Save the metrics DataFrame and configuration metadata to the run directory.
    """
    run_config = {
        "timestamp": timestamp,
        "correct_targets_minimum": config_file.CORRECT_THRESHOLD,
        "cut_criteria": config_file.CUT_CRITERIA,
        "consecutive_points": config_file.CONSECUTIVE_POINTS,
        "random_state": random_state,
        "test_size": test_size,
        "raw_data_dir": config_file.RAW_DATA_DIR,
        "metadata_path": config_file.METADATA_CSV,
    }
    run_config.update(git_info())

    config_path = run_dir / "configuration.json"
    try:
        with config_path.open("w") as f:
            json.dump(run_config, f, indent=2)
        logger.info("Saved configuration to %s", config_path)
    except IOError as e:
        logger.error("Failed to write configuration file: %s", e)

    # Write results CSV
    data_path = run_dir / "analysis.csv"
    try:
        df.to_csv(data_path, index=False)
        logger.info("Saved analysis results to %s", data_path)
    except IOError as e:
        logger.error("Failed to write analysis CSV: %s", e)


def main():
    test_size = 0.2
    metrics_df = load_analysis(RANDOM_STATE, test_size)
    print("Metrics DataFrame:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
