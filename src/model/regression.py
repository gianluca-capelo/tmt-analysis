import logging

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR

from src.model.classification import retrieve_dataset


def get_parameter_grid():
    return {
        "RandomForestRegressor": {
            "regressor__n_estimators": [100, 500, 700, 1000],
            "regressor__max_depth": [None, 10, 20, 30]
        },
        "SVR": {
            "regressor__C": [0.1, 1, 10],
            "regressor__kernel": ['linear', 'rbf'],
            "regressor__epsilon": [0.01, 0.1, 0.2]
        },
        "LinearRegression": {
            # LinearRegression has no hyperparameters to tune
        },
        "Ridge": {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0]
        },
        "Lasso": {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0]
        },
        "XGBRegressor": {
            "regressor__n_estimators": [100, 300],
            "regressor__max_depth": [3, 5],
            "regressor__learning_rate": [0.05, 0.1]
        }
    }


def get_models(random_state: int):
    return [
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        SVR(kernel="linear", C=1e5, epsilon=0.00001),
        LinearRegression(n_jobs=-1),
        Ridge(random_state=random_state),
        Lasso(max_iter=10000, random_state=random_state, alpha=0.0001),
        xgb.XGBRegressor(random_state=random_state, tree_method="hist", n_jobs=-1)
    ]


def perform(perform_pca: bool, dataset_name: str, cv_type: str, n_splits: int, n_repeats: int, global_seed: int,
            inner_cv_seed: int, feature_selection: bool, tune_hyperparameters: bool, target_col):
    X, y, feature_names = retrieve_dataset(dataset_name, target_col)

    unique, counts = np.unique(y, return_counts=True)

    logging.info(f"Class distribution:{dict(zip(unique, counts))}")

    param_grids = get_parameter_grid()

    models = get_models(global_seed)

    outer_cv = LeaveOneOut()

    performance_metrics_df = perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca,
                                                      feature_selection, tune_hyperparameters,
                                                      inner_cv_seed, feature_names)

    return performance_metrics_df
