import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from src.config import PROCESSED_FOR_MODEL_DIR, CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR, DATASETS, \
    MODEL_INNER_SEED, MODEL_OUTER_SEED, PERFORM_PCA, PERFORM_FEATURE_SELECTION, TUNE_HYPERPARAMETERS
from src.hand_analysis.loader.load_last_split import load_last_analysis

CLASSIFICATION_TARGET_COLUMN_NAME = 'group'


def get_target_column(target_col, df):
    last_analysis, _ = load_last_analysis()
    if target_col not in last_analysis.columns:
        raise ValueError(f"Target column '{target_col}' not found in the last analysis DataFrame.")

    target_df = last_analysis[['subject_id', target_col]].drop_duplicates(subset='subject_id')
    merged_df = pd.merge(df, target_df, on='subject_id', how='inner')

    return merged_df[target_col].values


def split_features_and_target_for_classification(df):
    target_col = CLASSIFICATION_TARGET_COLUMN_NAME

    df_copy = df.copy()

    if target_col not in df_copy.columns:
        raise ValueError(f" For classification, the target column '{target_col}' must be present in the DataFrame.")

    y = df_copy[target_col].values

    df_copy = df_copy.drop(columns=['subject_id', target_col])

    X = df_copy.values

    feature_names = df_copy.columns

    return X, y, feature_names


def split_features_and_target_for_regression(df, target_col):
    df_copy = df.copy()

    if target_col in df_copy.columns:
        y = df_copy[target_col].values
        # TODO GIAN: por las dudas, dsp borrar esta linea
        df_copy = df_copy.drop(columns=[target_col])
        assert np.array_equal(y, get_target_column(target_col, df_copy)), \
            f"Target column '{target_col}' values do not match with the last analysis DataFrame"
    else:
        y = get_target_column(target_col, df_copy)

    df_copy = df_copy.drop(columns=['subject_id'])

    df_copy = df_copy.drop(columns=[CLASSIFICATION_TARGET_COLUMN_NAME])

    X = df_copy.values

    feature_names = df_copy.columns

    return X, y, feature_names


def join_on_subject(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Avoid duplicating group column if it exists in both DataFrames
    if CLASSIFICATION_TARGET_COLUMN_NAME in df1.columns and CLASSIFICATION_TARGET_COLUMN_NAME in df2.columns:
        df2 = df2.drop(columns=CLASSIFICATION_TARGET_COLUMN_NAME)

    return pd.merge(df1, df2, on='subject_id', how='inner')


def load_all_datasets() -> dict:
    path = os.path.join(PROCESSED_FOR_MODEL_DIR)
    return {
        'df_digital_tmt_with_target': pd.read_csv(os.path.join(path, 'df_digital_tmt_with_target.csv')),
        'demographic_df': pd.read_csv(os.path.join(path, 'demographic_df.csv')),
        'non_digital_df': pd.read_csv(os.path.join(path, 'non_digital_df.csv')),
        # 'df_digital_hand_and_eye': pd.read_csv(os.path.join(path, 'df_digital_hand_and_eye.csv')),
        # 'digital_test_less_subjects': pd.read_csv(os.path.join(path, 'digital_test_less_subjects.csv')),
        # 'non_digital_test_less_subjects': pd.read_csv(os.path.join(path, 'non_digital_test_less_subjects.csv')),
    }


def retrieve_dataset(dataset_name, target_col, is_classification):
    datasets = load_all_datasets()

    match dataset_name:
        case 'demographic':
            df = datasets['demographic_df']

        case 'demographic+digital':
            df = join_on_subject(datasets['df_digital_tmt_with_target'], datasets['demographic_df'])

        case 'non_digital_tests':
            df = datasets['non_digital_df']

        case 'digital_test':
            df = datasets['df_digital_tmt_with_target']

        case 'non_digital_tests+demo':
            df = join_on_subject(datasets['non_digital_df'], datasets['demographic_df'])

        # case 'demographic_less_subjects':
        #     df = datasets['demographic_df'].loc[datasets['df_digital_hand_and_eye'].index]
        # case 'demographic+digital_less':
        #     subset = datasets['df_digital_tmt_with_target'].loc[datasets['df_digital_hand_and_eye'].index]
        #     df = join_on_subject(subset, datasets['demographic_df'])
        # case 'non_digital_test_less_subjects':
        #     df = datasets['non_digital_test_less_subjects']
        #
        # case 'non_digital_test_less_subjects+demo':
        #     df = join_on_subject(datasets['non_digital_test_less_subjects'], datasets['demographic_df'])
        #
        # case 'digital_test_less_subjects':
        #     df = datasets['digital_test_less_subjects']
        #
        # case 'hand_and_eye':
        #     df = datasets['df_digital_hand_and_eye']
        #
        # case 'hand_and_eye_demo':
        #     df = join_on_subject(datasets['df_digital_hand_and_eye'], datasets['demographic_df'])

        case _:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")

    if is_classification:
        return split_features_and_target_for_classification(df)
    else:
        return split_features_and_target_for_regression(df, target_col)


def get_parameter_grid(is_classification):
    if is_classification:
        return {
            "RandomForestClassifier": {
                "classifier__n_estimators": [100, 500, 700, 1000],
                "classifier__max_depth": [None, 10, 20, 30]
            },
            "SVC": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ['linear', 'rbf']
            },
            "LogisticRegression": {
                "classifier__C": [0.1, 1, 10],
                "classifier__penalty": ['l2']
            },
            "XGBClassifier": {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1]
            }
        }
    else:
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


def get_models(random_state: int, is_classification):
    if is_classification:
        return [
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            SVC(random_state=random_state, probability=True, kernel='linear'),
            LogisticRegression(max_iter=1000, random_state=random_state, solver='saga', n_jobs=-1),
            xgb.XGBClassifier(random_state=random_state, tree_method="hist", eval_metric='logloss', n_jobs=-1)
        ]
    else:
        return [
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            SVR(),
            LinearRegression(n_jobs=-1),
            Ridge(random_state=random_state),
            Lasso(random_state=random_state),
            xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
            DummyRegressor()
        ]


def perform(perform_pca: bool, dataset_name: str, global_seed: int,
            inner_cv_seed: int, feature_selection: bool, tune_hyperparameters: bool, target_col, is_classification):
    X, y, feature_names = retrieve_dataset(dataset_name, target_col, is_classification)

    param_grids = get_parameter_grid(is_classification)

    models = get_models(global_seed, is_classification)

    outer_cv = LeaveOneOut()

    performance_metrics_df = perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca,
                                                      feature_selection, tune_hyperparameters,
                                                      inner_cv_seed, feature_names, is_classification)

    return performance_metrics_df, feature_names


def perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                             tune_hyperparameters: bool, inner_cv_seed: int, feature_names, is_classification):
    all_fold_metrics = []

    for model in models:
        model_name = model.__class__.__name__

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

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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
        if not perform_pca and is_classification:  # TODO GIAN:
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


def calculate_feature_importance_for_fold(X_train, best_model, feature_names, feature_selection, model_name):
    classifier = best_model.named_steps['classifier']
    select = best_model.named_steps['select']
    selected_features = select.get_support(indices=True) if feature_selection else np.arange(X_train.shape[1])
    selected_feature_names = feature_names[selected_features]

    if hasattr(classifier, 'feature_importances_'):  # RandomForest, XGBoost
        importances = classifier.feature_importances_
        return dict(zip(selected_feature_names, importances))

    elif model_name == 'LogisticRegression':
        importances = np.abs(classifier.coef_).flatten()
        return dict(zip(selected_feature_names, importances))

    elif model_name == 'SVC' and classifier.kernel == 'linear':
        importances = np.abs(classifier.coef_).flatten()
        return dict(zip(selected_feature_names, importances))

    raise ValueError(f"Model {model_name} does not support feature importance extraction.")


def calculate_metrics_leave_one_out_for_model_for_classification(df, model_name):
    model_df = df[df['model'] == model_name]
    y_true = model_df['y_test'].tolist()
    y_pred_proba = model_df['y_pred_proba'].tolist()
    y_pred = model_df['y_pred'].tolist()

    return pd.DataFrame({
        'model': [model_name],
        'auc': [roc_auc_score(y_true, y_pred_proba)],
        'accuracy': [accuracy_score(y_true, y_pred)],
        'balanced_accuracy': [balanced_accuracy_score(y_true, y_pred)],
        'precision': [precision_score(y_true, y_pred, zero_division=0)],
        'recall': [recall_score(y_true, y_pred, zero_division=0)],
        'f1': [f1_score(y_true, y_pred, zero_division=0)],
        'y_true': [y_true],
        'y_pred_proba': [y_pred_proba],
        'feature_importances': [calculate_feature_importance(model_df)]
    })


def calculate_metrics_leave_one_out_for_model(performance_df, model_name, is_classification):
    if is_classification:
        return calculate_metrics_leave_one_out_for_model_for_classification(performance_df, model_name)
    else:
        return calculate_metrics_leave_one_out_regression(performance_df, model_name)


def calculate_metrics_leave_one_out_regression(performance_df, model_name):
    df = performance_df[performance_df['model'] == model_name]
    y_true = df['y_test'].tolist()
    y_pred = df['y_pred'].tolist()

    return pd.DataFrame({
        'model': [model_name],
        'r2': [r2_score(y_true, y_pred)],
        'mse': [mean_squared_error(y_true, y_pred)],
        'mae': [mean_absolute_error(y_true, y_pred)],
        'y_true': [y_true],
        'y_pred': [y_pred],
        'feature_importances': [calculate_feature_importance(df)]
    })


def calculate_feature_importance(model_df):
    """
    Calculates the average importance of each feature across all folds.

    Parameters
    ----------
    model_df : pd.DataFrame
        Must contain a column 'feature_importances' with dicts of feature: importance.

    Returns
    -------
    dict
        A dictionary with features as keys and their average importance as values.
    """

    feature_importances = model_df['feature_importances'].dropna().tolist()

    if not feature_importances:
        logging.warning("No feature importance found for this model.")
        return {}

    sum_importance = defaultdict(float)

    for feature_dict in feature_importances:
        for feature, importance in feature_dict.items():
            sum_importance[feature] += importance

    num_folds = len(feature_importances)

    avg_importance = {feature: total_importance / num_folds for feature, total_importance in sum_importance.items()}

    return avg_importance


def calculate_metrics_leave_one_out(performance_metrics_df, is_classification):
    model_dfs = [
        calculate_metrics_leave_one_out_for_model(performance_metrics_df, model_name, is_classification)
        for model_name in performance_metrics_df['model'].unique()
    ]

    metrics_global = pd.concat(model_dfs, ignore_index=True)

    return metrics_global


def save_results(leave_one_out_metrics, dataset_name, feature_selection, perform_pca, performance_metrics_df,
                 tune_hyperparameters, is_classification, timestamp: str, feature_names):
    """
    Guarda los resultados y la configuración del experimento en un directorio por fecha y dataset.

    Args:
        leave_one_out_metrics (pd.DataFrame): Métricas agregadas por modelo.
        dataset_name (str): Nombre del dataset usado.
        feature_selection (bool): Si se aplicó selección de características.
        perform_pca (bool): Si se aplicó PCA.
        performance_metrics_df (pd.DataFrame): Métricas por fold.
        tune_hyperparameters (bool): Si se usó GridSearchCV.
        is_classification (bool): Si es una tarea de clasificación.
        timestamp (str): Marca de tiempo para la carpeta (formato "%Y-%m-%d_%H%M").
    """

    base_dir = CLASSIFICATION_RESULTS_DIR if is_classification else REGRESSION_RESULTS_DIR
    dataset_dir = os.path.join(base_dir, timestamp, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Guardar métricas por fold
    performance_metrics_df.to_csv(os.path.join(dataset_dir, "folds.csv"), index=False)

    # Guardar métricas globales
    leave_one_out_metrics.to_csv(os.path.join(dataset_dir, "summary.csv"), index=False)

    # Guardar configuración
    config = {
        "dataset": dataset_name,
        "feature_selection": feature_selection,
        "perform_pca": perform_pca,
        "tune_hyperparameters": tune_hyperparameters,
        "is_classification": is_classification,
        "timestamp": timestamp,
        "n_folds": len(performance_metrics_df),
        "feature_names": feature_names.tolist(),
    }

    with open(os.path.join(dataset_dir, f"config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    logging.info(f"Resuls saves in: {dataset_dir}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    is_classification, target_col = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    for dataset_name in DATASETS:
        logging.info(f"Processing dataset: {dataset_name}")

        run_experiment(dataset_name,
                       PERFORM_FEATURE_SELECTION,
                       MODEL_OUTER_SEED,
                       MODEL_INNER_SEED,
                       is_classification,
                       PERFORM_PCA,
                       target_col, timestamp,
                       tune_hyperparameters=TUNE_HYPERPARAMETERS)


def run_experiment(dataset_name, feature_selection, global_seed, inner_cv_seed, is_classification, perform_pca,
                   target_col, timestamp, tune_hyperparameters):
    performance_metrics_df, feature_names = perform(
        perform_pca=perform_pca,
        dataset_name=dataset_name,
        global_seed=global_seed,
        inner_cv_seed=inner_cv_seed,
        feature_selection=feature_selection,
        tune_hyperparameters=tune_hyperparameters,
        target_col=target_col,
        is_classification=is_classification
    )
    leave_one_out_metrics_df = calculate_metrics_leave_one_out(performance_metrics_df, is_classification)
    save_results(leave_one_out_metrics_df, dataset_name, feature_selection, perform_pca, performance_metrics_df,
                 tune_hyperparameters, is_classification=is_classification, timestamp=timestamp,
                 feature_names=feature_names)


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification or regression pipelines.")

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Tipo de tarea a ejecutar."
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Columna objetivo. En clasificación debe ser 'group'. En regresión, por defecto 'mmse' si no se pasa."
    )

    args = parser.parse_args()

    if args.task is None or args.task not in ["classification", "regression"]:
        raise ValueError("task must be provided and be either 'classification' or 'regression'")

    is_classification = args.task == "classification"

    if not is_classification:
        if args.target_col is None:
            raise ValueError("For regression, target_col must be provided")
        target_col = args.target_col
    else:
        target_col = CLASSIFICATION_TARGET_COLUMN_NAME

    return is_classification, target_col


if __name__ == "__main__":
    main()
