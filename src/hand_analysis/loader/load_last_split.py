import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

#import config as config_file
from src import config as config_file


def load_analysis_by_run_dir(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a run_dir (…/hand_analysis/YYYY-MM-DD_HH-MM-SS), load train/eval DataFrames.
    """
    # read config
    run_config = get_run_configuration(run_dir)
    train_ids = run_config.get("train_subject_ids")
    eval_ids = run_config.get("eval_subject_ids")
    if train_ids is None or eval_ids is None:
        raise KeyError("`train_subject_ids` or `eval_subject_ids` not found in configuration.json")
    train_ids = [int(i) for i in train_ids]
    eval_ids = [int(i) for i in eval_ids]

    # read full CSV
    data_path = run_dir / "analysis.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Analysis CSV not found in {run_dir}")
    df = pd.read_csv(data_path)

    # split out train/eval
    df_train = df[df['subject_id'].isin(train_ids)].reset_index(drop=True)
    df_eval = df[df['subject_id'].isin(eval_ids)].reset_index(drop=True)

    return df_train, df_eval


def load_analysis_by_date(
        date: Union[str, "datetime.date"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Look under HAND_ANALYSIS_FOLDER for subdirs whose names start with `YYYY-MM-DD`
    (or the given string), pick the latest one, and load it.
    """
    # normalize date to string
    if not isinstance(date, str):
        date = date.strftime("%Y-%m-%d")

    base_dir = Path(config_file.HAND_ANALYSIS_FOLDER)
    # find all runs matching the prefix
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith(date)
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found for date {date!r} in {base_dir}")

    # because folders are named YYYY-MM-DD_HH-MM-SS, lexicographic sort works
    run_dir = max(candidates, key=lambda d: d.name)
    return load_analysis_by_run_dir(run_dir)


def load_last_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the train/eval DataFrames from the most recent run directory.
    """
    run_dir = get_last_run_directory()
    return load_analysis_by_run_dir(run_dir)


def get_run_configuration(run_dir) -> dict:
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
