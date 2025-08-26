import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

from src.config import CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR


def _select_metric(metric: str):
    """
    Devuelve (metric_fn, higher_is_better) según el nombre de la métrica.
    """
    name = str(metric).lower()
    if name == "auc":
        return roc_auc_score, True  # mayor es mejor
    if name == "r2":
        return r2_score, True
    if name == "mse":
        return mean_squared_error, False  # menor es mejor
    if name == "mae":
        return mean_absolute_error, False

    raise ValueError("Unknown metric: {}".format(metric))


def permutation_test_auc(y_true, y_pred_proba, n_permutations=1000, seed=42, metric='auc'):
    rng = np.random.RandomState(seed)

    scoring_fn, higher_score_is_better = _select_metric(metric)

    true_score = scoring_fn(y_true, y_pred_proba)

    permuted_scores = []
    attempts = 0
    max_attempts = n_permutations * 10  # Prevent infinite loop

    while len(permuted_scores) < n_permutations and attempts < max_attempts:
        y_permuted = rng.permutation(y_true)
        try:
            permuted_auc_score = scoring_fn(y_permuted, y_pred_proba)
            permuted_scores.append(permuted_auc_score)
        except ValueError:
            pass  # Skip failed permutations
        attempts += 1

    if len(permuted_scores) < n_permutations:
        print(f"Warning: Only {len(permuted_scores)} valid permutations out of {n_permutations}")

    if higher_score_is_better:
        permuted_better_or_equal = np.array(permuted_scores) >= true_score
    else:
        permuted_better_or_equal = np.array(permuted_scores) <= true_score

    n_better_or_equal = np.sum(permuted_better_or_equal)
    p_value = (n_better_or_equal + 1) / (len(permuted_scores) + 1)

    return true_score, p_value


def compute_permutation_tests(date_folder: str, task: str, datasets_filter: list = None, metric:str = "auc"):
    task_results_dir = CLASSIFICATION_RESULTS_DIR if task == "classification" else REGRESSION_RESULTS_DIR
    results_dir = Path(os.path.join(task_results_dir, date_folder))

    y_pred_column = 'y_pred_proba' if task == "classification" else 'y_pred'

    summary_paths = [f for f in results_dir.rglob("summary.csv")
                     if (datasets_filter is None) or (f.parent.name in datasets_filter)]

    all_summary_dfs = []

    for path in summary_paths:
        df = pd.read_csv(path)
        dataset = path.parent.name
        df['dataset'] = dataset
        df['parent_date'] = path.parents[1].name  # e.g. 2025-08-19_1224
        all_summary_dfs.append(df)

    all_summaries = pd.concat(all_summary_dfs, ignore_index=True)

    best_df = pd.DataFrame(all_summaries)

    for idx, row in best_df.iterrows():
        y_true = eval(row['y_true']) if isinstance(row['y_true'], str) else row['y_true']
        y_pred = eval(row[y_pred_column]) if isinstance(row[y_pred_column], str) else row[y_pred_column]
        score, p_value = permutation_test_auc(y_true,
                                              y_pred,
                                              n_permutations=1000,
                                              metric=metric
                                              )

        print(f"Dataset: {row['dataset']}, Model: {row['model']}, {metric.upper()}: {score:.4f}, p-value: {p_value:.4f}")
        print("---" * 100)


if __name__ == "__main__":
    compute_permutation_tests("2025-08-25_1821", task="regression", metric="mae")
