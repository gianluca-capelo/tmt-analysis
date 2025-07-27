import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROCESSED_FOR_MODEL_DIR, CLASSIFICATION_RESULTS_DIR


def split_features_and_target(df, target_col):
    df_copy = df.copy().drop('subject_id', axis=1)

    X = df_copy.drop(columns=target_col).values

    y = df_copy[target_col].values

    feature_names = df_copy.drop(columns=target_col).columns

    return X, y, feature_names


def join_on_subject_and_move_target_last(df1: pd.DataFrame, df2: pd.DataFrame, target_col) -> pd.DataFrame:
    """
    Merge two DataFrames on 'subject_id' and move the target column to the end.

    Args:
        df1 (pd.DataFrame): First DataFrame, must include 'subject_id' and the target column.
        df2 (pd.DataFrame): Second DataFrame, will be merged after dropping the target column if present.
        target_col (str): The name of the target column to move to the end.

    Returns:
        pd.DataFrame: Merged DataFrame with the target column as the last column.
    """
    if target_col not in df1.columns:
        raise ValueError(f"'{target_col}' must be present in df1 to retain it as the target.")

    # Drop the target column from df2 if it exists to avoid duplication
    if target_col in df2.columns:
        df2 = df2.drop(columns=target_col)

    # Perform the merge on subject_id
    df = pd.merge(df1, df2, on='subject_id', how='inner')

    # Move the target column to the end
    reordered_cols = [col for col in df.columns if col != target_col] + [target_col]
    df = df[reordered_cols]

    return df


def load_all_datasets() -> dict:
    path = os.path.join(PROCESSED_FOR_MODEL_DIR)
    return {
        'df_digital_tmt_with_target': pd.read_csv(os.path.join(path, 'df_digital_tmt_with_target.csv')),
        'demographic_df': pd.read_csv(os.path.join(path, 'demographic_df.csv')),
        'non_digital_df': pd.read_csv(os.path.join(path, 'non_digital_df.csv')),
        'df_digital_hand_and_eye': pd.read_csv(os.path.join(path, 'df_digital_hand_and_eye.csv')),
        'digital_test_less_subjects': pd.read_csv(os.path.join(path, 'digital_test_less_subjects.csv')),
        'non_digital_test_less_subjects': pd.read_csv(os.path.join(path, 'non_digital_test_less_subjects.csv')),
    }


def retrieve_dataset(dataset_name, target_col):
    datasets = load_all_datasets()

    match dataset_name:
        case 'demographic':
            df = datasets['demographic_df']

        case 'demographic_less_subjects':
            df = datasets['demographic_df'].loc[datasets['df_digital_hand_and_eye'].index]

        case 'demographic+digital':
            df = join_on_subject_and_move_target_last(datasets['df_digital_tmt_with_target'],
                                                      datasets['demographic_df'], target_col)

        case 'demographic+digital_less':
            subset = datasets['df_digital_tmt_with_target'].loc[datasets['df_digital_hand_and_eye'].index]
            df = join_on_subject_and_move_target_last(subset, datasets['demographic_df'], target_col)

        case 'non_digital_tests':
            df = datasets['non_digital_df']

        case 'non_digital_tests+demo':
            df = join_on_subject_and_move_target_last(datasets['non_digital_df'].drop(columns='subject_id'),
                                                      datasets['demographic_df'], target_col)

        case 'non_digital_test_less_subjects':
            df = datasets['non_digital_test_less_subjects']

        case 'non_digital_test_less_subjects+demo':
            df = join_on_subject_and_move_target_last(
                datasets['non_digital_test_less_subjects'].drop(columns='subject_id'),
                datasets['demographic_df'],
                target_col)

        case 'digital_test':
            df = datasets['df_digital_tmt_with_target']

        case 'digital_test_less_subjects':
            df = datasets['digital_test_less_subjects']

        case 'hand_and_eye':
            df = datasets['df_digital_hand_and_eye']

        case 'hand_and_eye_demo':
            df = join_on_subject_and_move_target_last(datasets['df_digital_hand_and_eye'], datasets['demographic_df'],
                                                      target_col)

        case _:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")

    return split_features_and_target(df, target_col)


def get_parameter_grid():
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


def get_models(random_state: int):
    return [
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        SVC(random_state=random_state, probability=True, kernel='linear'),
        LogisticRegression(max_iter=1000, random_state=random_state, solver='saga', n_jobs=-1),
        xgb.XGBClassifier(random_state=random_state, tree_method="hist", eval_metric='logloss', n_jobs=-1)
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


def perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                             tune_hyperparameters: bool, inner_cv_seed: int, feature_names):
    all_fold_metrics = []

    for model in models:
        model_name = model.__class__.__name__

        logging.info(f"\n🧪 CV for: {model_name}")

        param_grid = param_grids.get(model_name, {})

        fold_metrics = perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca,
                                                          feature_selection,
                                                          tune_hyperparameters,
                                                          inner_cv_seed,
                                                          feature_names)

        all_fold_metrics.extend(fold_metrics)

    return pd.DataFrame(all_fold_metrics)


def perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                                       tune_hyperparameters: bool, inner_cv_seed: int,
                                       feature_names):
    max_pca_components = 4
    max_selected_features = 20
    model_name = model.__class__.__name__
    logging.info(f"\n🧪 CV for: {model_name}")

    fold_metrics = []

    # Enumeramos 'repeat' y 'fold' para guardar en métricas
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logging.info(f'Fold number: {fold}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Steps
        pca_step = ('pca', PCA(n_components=min(max_pca_components, X_train.shape[1]))) if perform_pca else ('noop',
                                                                                                             'passthrough')
        select_step = (
            ('select', SelectKBest(score_func=f_classif, k=min(max_selected_features, X_train.shape[1])))
            if feature_selection else ('noop', 'passthrough')
        )

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            select_step,
            ('scaler', StandardScaler()),
            pca_step,
            ('classifier', model)
        ])

        if tune_hyperparameters and param_grid:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_cv_seed)
            grid = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        # Only compute importance if PCA is OFF
        if not perform_pca:
            importance_dict = calculate_feature_importance_for_fold(X_train, best_model, feature_names,
                                                                    feature_selection, model_name)
        else:
            importance_dict = {}
            logging.info(f"Skipping feature importance for {model_name} in fold {fold} due to PCA.")

        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        fold_metrics.append({
            'model': model_name,
            'fold': fold,
            'y_test': y_test[0],
            'y_pred': y_pred[0],
            'y_pred_proba': y_pred_proba[0],
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


def calculate_metrics_leave_one_out_for_model(df, model_name):
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


def calculate_metrics_leave_one_out(performance_metrics_df):
    model_dfs = [
        calculate_metrics_leave_one_out_for_model(performance_metrics_df, model_name)
        for model_name in performance_metrics_df['model'].unique()
    ]

    metrics_global = pd.concat(model_dfs, ignore_index=True)

    return metrics_global


def save_results(leave_one_out_metrics, dataset_name, feature_selection, perform_pca, performance_metrics_df,
                 tune_hyperparameters):
    classification_dir = CLASSIFICATION_RESULTS_DIR
    save_dir = os.path.join(classification_dir, datetime.now().strftime("%Y-%m-%d"))

    os.makedirs(save_dir, exist_ok=True)

    if perform_pca:
        save_path = f'{save_dir}/{dataset_name}_feature_{feature_selection}_tune={tune_hyperparameters}_LOOCV_PCA_{datetime.now().strftime("%s")[-4:]}.csv'
    else:
        save_path = f'{save_dir}/{dataset_name}_feature_{feature_selection}_tune={tune_hyperparameters}_LOOCV_{datetime.now().strftime("%s")[-4:]}.csv'

    performance_metrics_df.to_csv(save_path, index=False)
    leave_one_out_metrics.to_csv(f'{save_dir}/{dataset_name}_leave_one_out.csv', index=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    n_splits = 2
    n_repeats = 1
    global_seed = 42
    inner_cv_seed = 50  # Fixed for reproducibility in inner CV
    type_of_cv = 'loo'
    tune_hyperparameters = False
    feature_selection = True
    dataset_name = 'demographic+digital'
    perform_pca = False
    target_col = 'group'

    performance_metrics_df = perform(
        perform_pca=perform_pca,
        dataset_name=dataset_name,
        cv_type=type_of_cv,
        n_splits=n_splits,
        n_repeats=n_repeats,
        global_seed=global_seed,
        inner_cv_seed=inner_cv_seed,
        feature_selection=feature_selection,
        tune_hyperparameters=tune_hyperparameters,
        target_col=target_col
    )

    leave_one_out_metrics_df = calculate_metrics_leave_one_out(performance_metrics_df)

    save_results(leave_one_out_metrics_df, dataset_name, feature_selection, perform_pca, performance_metrics_df,
                 tune_hyperparameters)


if __name__ == "__main__":
    main()
