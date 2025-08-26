import numpy as np

from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error


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


def permutation_test_auc(y_true, y_pred_proba, n_permutations=1000, seed=42):
    rng = np.random.RandomState(seed)

    scoring_fn, higher_score_is_better = _select_metric("auc")

    true_auc = scoring_fn(y_true, y_pred_proba)

    permuted_auc_scores = []
    attempts = 0
    max_attempts = n_permutations * 10  # Prevent infinite loop

    while len(permuted_auc_scores) < n_permutations and attempts < max_attempts:
        y_permuted = rng.permutation(y_true)
        try:
            permuted_auc_score = scoring_fn(y_permuted, y_pred_proba)
            permuted_auc_scores.append(permuted_auc_score)
        except ValueError:
            pass  # Skip failed permutations
        attempts += 1

    if len(permuted_auc_scores) < n_permutations:
        print(f"Warning: Only {len(permuted_auc_scores)} valid permutations out of {n_permutations}")

    if higher_score_is_better:
        permuted_better_or_equal = np.array(permuted_auc_scores) >= true_auc
    else:
        permuted_better_or_equal = np.array(permuted_auc_scores) <= true_auc

    n_better_or_equal = np.sum(permuted_better_or_equal)
    p_value = (n_better_or_equal + 1) / (len(permuted_auc_scores) + 1)

    return true_auc, p_value
