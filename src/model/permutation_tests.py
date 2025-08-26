import numpy as np

from sklearn.metrics import roc_curve, auc, roc_auc_score


def permutation_test_auc(y_true, y_pred_proba, n_permutations=1000, seed=42):
    rng = np.random.RandomState(seed)
    true_auc = roc_auc_score(y_true, y_pred_proba)

    permuted_auc_scores = []
    attempts = 0
    max_attempts = n_permutations * 10  # Prevent infinite loop

    while len(permuted_auc_scores) < n_permutations and attempts < max_attempts:
        y_permuted = rng.permutation(y_true)
        try:
            permuted_auc_score = roc_auc_score(y_permuted, y_pred_proba)
            permuted_auc_scores.append(permuted_auc_score)
        except ValueError:
            pass  # Skip failed permutations
        attempts += 1

    if len(permuted_auc_scores) < n_permutations:
        print(f"Warning: Only {len(permuted_auc_scores)} valid permutations out of {n_permutations}")

    permuted_better_or_equal = np.array(permuted_auc_scores) >= true_auc
    n_better_or_equal = np.sum(permuted_better_or_equal)
    p_value = (n_better_or_equal + 1) / (len(permuted_auc_scores) + 1)

    return true_auc, p_value