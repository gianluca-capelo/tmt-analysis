import logging

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.model.classification.classification import save_results, calculate_feature_importance, perform





def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    global_seed = 42
    inner_cv_seed = 50  # Fixed for reproducibility in inner CV
    tune_hyperparameters = False
    feature_selection = False
    dataset_name = 'demographic+digital'
    perform_pca = False
    target_col = 'mmse'
    is_classification = False
    performance_metrics_df = perform(
        perform_pca=perform_pca,
        dataset_name=dataset_name,
        global_seed=global_seed,
        inner_cv_seed=inner_cv_seed,
        feature_selection=feature_selection,
        tune_hyperparameters=tune_hyperparameters,
        target_col=target_col,
        is_classification=is_classification
    )

    leave_one_out_metrics_df = calculate_metrics_leave_one_out_regression(performance_metrics_df)

    save_results(leave_one_out_metrics_df, dataset_name, feature_selection, perform_pca, performance_metrics_df,
                 tune_hyperparameters, is_classification)


if __name__ == "__main__":
    main()
