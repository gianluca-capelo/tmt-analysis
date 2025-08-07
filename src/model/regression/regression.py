import logging

import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.classification.classification import calculate_feature_importance_for_fold, \
    save_results, retrieve_dataset, calculate_feature_importance
from tqdm import tqdm


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
        # SVR(kernel="linear", C=1e5, epsilon=0.00001), #TODO GIAN: porque se traba?
        LinearRegression(n_jobs=-1),
        Ridge(random_state=random_state),
        Lasso(max_iter=10000, random_state=random_state, alpha=0.0001),
        xgb.XGBRegressor(random_state=random_state, tree_method="hist", n_jobs=-1)
    ]


def perform(perform_pca: bool, dataset_name: str, global_seed: int,
            inner_cv_seed: int, feature_selection: bool, tune_hyperparameters: bool, target_col, is_classification):
    X, y, feature_names = retrieve_dataset(dataset_name, target_col)

    param_grids = get_parameter_grid()

    models = get_models(global_seed)

    outer_cv = LeaveOneOut()

    performance_metrics_df = perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca,
                                                      feature_selection, tune_hyperparameters,
                                                      inner_cv_seed, feature_names, is_classification)

    return performance_metrics_df


def perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                             tune_hyperparameters: bool, inner_cv_seed: int, feature_names, is_classification):
    all_fold_metrics = []

    for model in models:
        model_name = model.__class__.__name__

        logging.info(f"\n🧪 CV for: {model_name}")

        param_grid = param_grids.get(model_name, {})

        fold_metrics = perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca,
                                                          feature_selection,
                                                          tune_hyperparameters,
                                                          inner_cv_seed,
                                                          feature_names, is_classification)

        all_fold_metrics.extend(fold_metrics)

    return pd.DataFrame(all_fold_metrics)


def perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                                       tune_hyperparameters: bool, inner_cv_seed: int,
                                       feature_names, is_classification):
    if is_classification:
        select_score_func = f_classif
        pipeline_name = 'classifier'
    else:
        select_score_func = f_regression
        pipeline_name = 'regressor'

    max_pca_components = 4
    max_selected_features = 20
    model_name = model.__class__.__name__
    logging.info(f"\n🧪 CV for: {model_name}")

    fold_metrics = []

    n_folds = outer_cv.get_n_splits(X)
    fold_iterator = tqdm(
        enumerate(outer_cv.split(X, y)),
        total=n_folds,
        desc=f"Model: {model_name}",
        position=0,
        leave=True
    )

    for fold, (train_idx, test_idx) in fold_iterator:
        logging.info(f'Fold number: {fold}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Steps
        pca_step = (
            ('pca', PCA(n_components=min(max_pca_components, X_train.shape[1])))
            if perform_pca else ('pca_noop', 'passthrough')
        )

        select_step = (
            ('select', SelectKBest(score_func=select_score_func, k=min(max_selected_features, X_train.shape[1])))
            if feature_selection else ('select_noop', 'passthrough')
        )

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            select_step,
            ('scaler', StandardScaler()),
            pca_step,
            (pipeline_name, model)
        ])

        if tune_hyperparameters and param_grid:
            inner_cv = (
                StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_cv_seed)
                if is_classification
                else KFold(n_splits=3, shuffle=True, random_state=inner_cv_seed)
            )
            scoring = 'roc_auc' if is_classification else 'r2'
            grid = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        # Only compute importance if PCA is OFF
        if not perform_pca and is_classification:  # TODO GIAN: adaptar a regresion
            importance_dict = calculate_feature_importance_for_fold(X_train, best_model, feature_names,
                                                                    feature_selection, model_name)
        else:
            importance_dict = {}

        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if is_classification else None
        y_pred = best_model.predict(X_test)

        fold_metrics.append({
            'model': model_name,
            'fold': fold,
            'y_test': y_test[0],
            'y_pred': y_pred[0],
            'y_pred_proba': y_pred_proba[0] if y_pred_proba is not None else None,
            'feature_importances': importance_dict,
            'feature_names': feature_names
        })

    return fold_metrics


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics_leave_one_out_regression(performance_df):
    model_dfs = []
    for model_name in performance_df['model'].unique():
        df = performance_df[performance_df['model'] == model_name]
        y_true = df['y_test'].tolist()
        y_pred = df['y_pred'].tolist()

        model_dfs.append(pd.DataFrame({
            'model': [model_name],
            'r2': [r2_score(y_true, y_pred)],
            'mse': [mean_squared_error(y_true, y_pred)],
            'mae': [mean_absolute_error(y_true, y_pred)],
            'y_true': [y_true],
            'y_pred': [y_pred],
            'feature_importances': [calculate_feature_importance(df)]
        }))

    return pd.concat(model_dfs, ignore_index=True)


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
