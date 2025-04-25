import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config as config_file


def get_configuration(
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and evaluation DataFrames from the most recent  run directory by reading from configuration.json.

    Returns:
    - df_train: DataFrame with training data.
    - df_eval: DataFrame with evaluation data.
    """
    run_dir = get_last_run_directory()

    run_config = get_run_configuration(run_dir)

    train_ids = run_config.get("train_subject_ids")
    eval_ids = run_config.get("eval_subject_ids")
    if train_ids is None or eval_ids is None:
        raise KeyError("`train_subject_ids` or `eval_subject_ids` not found in configuration.json")

    # Load full results
    data_path = run_dir / "analysis.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Analysis CSV not found in {run_dir}")
    df = pd.read_csv(data_path)

    # Filter by IDs
    df_train = df[df['subject_id'].isin(train_ids)].reset_index(drop=True)
    df_eval = df[df['subject_id'].isin(eval_ids)].reset_index(drop=True)

    return df_train, df_eval


def get_run_configuration(run_dir):
    config_path = run_dir / "configuration.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found in {run_dir}")
    with config_path.open() as f:
        run_config = json.load(f)
    return run_config


def get_last_run_directory():
    base_dir = Path(config_file.HAND_ANALYSIS_FOLDER)
    candidates = [d for d in base_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    run_dir = max(candidates)
    logging.info("Loading split from %s", run_dir)
    return run_dir
