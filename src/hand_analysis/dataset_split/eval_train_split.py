import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAIN_SET_PATH, EVAL_SET_PATH


def split_subjectwise_evaluation_set_stratified(df: pd.DataFrame, test_size, random_state):
    """
    Splits a dataset into training and evaluation sets, stratifying by group at the subject level.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - test_size: proportion of subjects to include in the evaluation set.
    - random_state: seed for reproducibility.

    Returns:
    - df_train: DataFrame with training data.
    - df_eval: DataFrame with evaluation data.
    """

    if random_state is None:
        raise ValueError("Random state must be present")

    subject_col = 'subject_id'
    group_col = 'group'
    # Get unique subjects and their group labels
    subject_group_df = df[[subject_col, group_col]].drop_duplicates()

    # Stratified split based on group label
    train_subjects, eval_subjects = train_test_split(
        subject_group_df[subject_col],
        test_size=test_size,
        stratify=subject_group_df[group_col],
        random_state=random_state
    )

    # Select rows from the original dataframe
    df_train = df[df[subject_col].isin(train_subjects)].copy()
    df_eval = df[df[subject_col].isin(eval_subjects)].copy()

    df_train.to_csv(TRAIN_SET_PATH)
    df_eval.to_csv(EVAL_SET_PATH)

    return df_train, df_eval
