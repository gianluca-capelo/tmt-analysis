import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config as config_file


def load_last_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and evaluation DataFrames from the most recent (or specified) run directory.

    Parameters:
    - run_folder: path to a specific run directory. If None, auto-selects the latest by timestamp.

    Returns:
    - df_train: DataFrame with training data.
    - df_eval: DataFrame with evaluation data.
    """
    base_dir = Path(config_file.HAND_ANALYSIS_FOLDER)

    candidates = [d for d in base_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {base_dir}")

    run_dir = max(candidates)
    logging.info("Loading split from %s", run_dir)

    train_ids_path = run_dir / 'train_subject_ids.json'
    eval_ids_path = run_dir / 'eval_subject_ids.json'
    if not train_ids_path.exists() or not eval_ids_path.exists():
        raise FileNotFoundError("Train/eval ID files not found in " + str(run_dir))

    with train_ids_path.open() as f:
        train_ids = json.load(f)
    with eval_ids_path.open() as f:
        eval_ids = json.load(f)

    data_path = run_dir / 'analysis.csv'
    if not data_path.exists():
        raise FileNotFoundError("Analysis CSV not found in " + str(run_dir))
    df = pd.read_csv(data_path)

    df_train = df[df['subject_id'].isin(train_ids)].reset_index(drop=True)
    df_eval = df[df['subject_id'].isin(eval_ids)].reset_index(drop=True)

    return df_train, df_eval
