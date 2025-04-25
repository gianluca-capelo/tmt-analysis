import json
import os
import subprocess
from datetime import datetime

import pandas as pd

from src import config
from src.config import RANDOM_STATE
from src.hand_analysis.dataset_split.eval_train_split import split_subjectwise_evaluation_set_stratified
from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_configuration_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis(random_state, test_size, split=False) -> pd.DataFrame:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = create_save_results_folder(timestamp)

    analysis = run_analysis_with_configuration_parameters(run_dir)

    metrics_df = analysis.get_metrics_dataframe()

    metrics_df_with_metadata = add_metadata_to_metrics(metrics_df)

    valid_metrics_df = filter_invalid_subjects(metrics_df_with_metadata)

    save_results(valid_metrics_df, timestamp, run_dir)

    if split:
        split_subjectwise_evaluation_set_stratified(valid_metrics_df, test_size=test_size, random_state=random_state)

    return valid_metrics_df


def create_save_results_folder(timestamp):
    root_dir = config.HAND_ANALYSIS_FOLDER
    run_dir = os.path.join(root_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def git_info():
    # current branch
    branch = (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
    # last commit hash
    sha = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
    # optionally: commit message
    msg = (
        subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
    # detect local uncommitted changes
    dirty = (
            subprocess.call(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
            )
            != 0
    )
    return {
        "git_branch": branch,
        "git_commit": sha,
        "git_msg": msg,
        "git_dirty": dirty
    }


def save_results(valid_metrics_df, timestamp, run_dir):
    threshold = config.CORRECT_THRESHOLD
    cut_criteria = config.CUT_CRITERIA
    points = config.CONSECUTIVE_POINTS

    configuration = {
        "correct_targets_minimum": threshold,
        "cut_criteria": cut_criteria,
        "consecutive_points": points,
        "timestamp": timestamp,
    }

    configuration.update(git_info())

    with open(os.path.join(run_dir, "configuration.json"), "w") as f:
        json.dump(config, f, indent=2)

    valid_metrics_df.to_csv(os.path.join(run_dir, "analysis.csv"), index=False)


def main():
    test_size = 0.2
    metrics_df = load_analysis(RANDOM_STATE, test_size)
    print("Metrics DataFrame:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
