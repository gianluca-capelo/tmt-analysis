import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import PROCESSED_FOR_MODEL_DIR, CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR, DATASETS, \
    MODEL_INNER_SEED, MODEL_OUTER_SEED, PERFORM_FEATURE_SELECTION, TUNE_HYPERPARAMETERS, \
    REGRESSION_TARGETS, CLASSIFICATION_TARGET, CLASSIFICATION_MODELS, REGRESSION_MODELS, CLASSIFICATION_PARAM_GRID, \
    REGRESSION_PARAM_GRID, MAX_SELECTED_FEATURES, INNER_CV_SPLITS
from src.hand_analysis.loader.load_last_split import load_last_analysis


def save_shap_plot(shap_values, dataset_dir, dataset_name, model_name,
                   plot_type="bar", file_format="png", max_display=20):
    """
    Guarda un gráfico de SHAP en el formato elegido.

    Args:
        shap_values: shap.Explanation ya normalizado (2D).
        dataset_dir (str): carpeta destino.
        dataset_name (str): nombre del dataset.
        model_name (str): nombre del modelo.
        plot_type (str): "bar" o "beeswarm".
        file_format (str): "png", "pdf", "svg", etc.
        max_display (int): features a mostrar.
    """
    k = min(max_display, shap_values.values.shape[1])

    fig = plt.figure(figsize=(9, 0.48 * k + 1.4))

    if plot_type == "bar":
        shap.plots.bar(shap_values, max_display=k, show=False)
    elif plot_type == "beeswarm":
        shap.plots.beeswarm(shap_values, max_display=k, show=False)
    else:
        raise ValueError(f"Plot type '{plot_type}' not supported.")

    ax = plt.gca()
    ax.set_title(f"SHAP {plot_type.capitalize()} - {dataset_name}", pad=12)

    fig.subplots_adjust(left=0.40, right=0.96, top=0.95, bottom=0.08)

    out_path = os.path.join(dataset_dir, f"{dataset_name}_{model_name}_shap_{plot_type}.{file_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved SHAP {plot_type} plot to {out_path}")

def get_target_column(target_col, df):
    last_analysis, _ = load_last_analysis()
    if target_col not in last_analysis.columns:
        raise ValueError(f"Target column '{target_col}' not found in the last analysis DataFrame.")

    target_df = last_analysis[['subject_id', target_col]].drop_duplicates(subset='subject_id')
    merged_df = pd.merge(df, target_df, on='subject_id', how='inner')

    return merged_df[target_col].values


def split_features_and_target_for_classification(df):
    target_col = CLASSIFICATION_TARGET

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

    df_copy = df_copy.drop(columns=[CLASSIFICATION_TARGET])

    X = df_copy.values

    feature_names = df_copy.columns

    return X, y, feature_names


def join_on_subject(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Avoid duplicating group column if it exists in both DataFrames
    if CLASSIFICATION_TARGET in df1.columns and CLASSIFICATION_TARGET in df2.columns:
        df2 = df2.drop(columns=CLASSIFICATION_TARGET)

    return pd.merge(df1, df2, on='subject_id', how='inner')


def load_all_datasets() -> dict:
    path = os.path.join(PROCESSED_FOR_MODEL_DIR)
    return {
        'df_digital_tmt_with_target': pd.read_csv(os.path.join(path, 'df_digital_tmt_with_target.csv')),
        'demographic_df': pd.read_csv(os.path.join(path, 'demographic_df.csv')),
        'non_digital_df': pd.read_csv(os.path.join(path, 'non_digital_df.csv')),
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

        case _:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")

    if is_classification:
        return split_features_and_target_for_classification(df)
    else:
        return split_features_and_target_for_regression(df, target_col)


def get_parameter_grid(is_classification):
    if is_classification:
        return CLASSIFICATION_PARAM_GRID
    else:
        return REGRESSION_PARAM_GRID


def get_models(random_state: int, is_classification):
    if is_classification:
        return CLASSIFICATION_MODELS(random_state)
    else:
        return REGRESSION_MODELS(random_state)


def perform(dataset_name: str, global_seed: int,
            inner_cv_seed: int, feature_selection: bool, tune_hyperparameters: bool, target_col, is_classification):
    X, y, feature_names = retrieve_dataset(dataset_name, target_col, is_classification)

    param_grids = get_parameter_grid(is_classification)

    models = get_models(global_seed, is_classification)

    outer_cv = LeaveOneOut()

    performance_metrics_df = perform_cross_validation(param_grids, models, outer_cv, X, y, feature_selection, tune_hyperparameters,
                                                      inner_cv_seed, feature_names, is_classification)

    return performance_metrics_df, feature_names


def perform_cross_validation(param_grids, models, outer_cv, X, y, feature_selection: bool,
                             tune_hyperparameters: bool, inner_cv_seed: int, feature_names, is_classification):
    all_fold_metrics = []

    for model in models:
        model_name = model.__class__.__name__

        param_grid = param_grids.get(model_name, {})

        fold_metrics = perform_cross_validation_for_model(param_grid, model, outer_cv, X, y,
                                                          feature_selection,
                                                          tune_hyperparameters,
                                                          inner_cv_seed,
                                                          feature_names, is_classification)

        all_fold_metrics.extend(fold_metrics)

    return pd.DataFrame(all_fold_metrics)


def perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, feature_selection: bool,
                                       tune_hyperparameters: bool, inner_cv_seed: int,
                                       feature_names, is_classification):
    if is_classification:
        select_score_func = f_classif
        pipeline_name = 'classifier'
    else:
        select_score_func = f_regression
        pipeline_name = 'regressor'

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

        select_step = (
            ('select', SelectKBest(score_func=select_score_func, k=min(MAX_SELECTED_FEATURES, X_train.shape[1])))
            if feature_selection else ('select_noop', 'passthrough')
        )

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            select_step,
            ('scaler', StandardScaler()),
            (pipeline_name, model)
        ])

        if tune_hyperparameters and param_grid:
            inner_cv = (
                StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=inner_cv_seed)
                if is_classification
                else KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=inner_cv_seed)
            )
            scoring = 'roc_auc' if is_classification else 'neg_mean_absolute_error'  # neg_mean_absolute_error is for MAE

            grid = GridSearchCV(pipeline, refit=True, param_grid=param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1,
                                verbose=1)

            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if is_classification else None
        y_pred = best_model.predict(X_test)

        mask = best_model.named_steps["select"].get_support()
        selected_features = feature_names[mask]

        fold_metrics.append({
            'model': model_name,
            'fold': fold,
            'y_test': y_test[0],
            'y_pred': y_pred[0],
            'y_pred_proba': y_pred_proba[0] if y_pred_proba is not None else None,
            'feature_names': feature_names.tolist(),
            'hyperparameters': best_model.named_steps[pipeline_name].get_params(),
            'select_k_best_features': selected_features.tolist()
        })

    return fold_metrics


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
    })


def calculate_metrics_leave_one_out(performance_metrics_df, is_classification):
    model_dfs = [
        calculate_metrics_leave_one_out_for_model(performance_metrics_df, model_name, is_classification)
        for model_name in performance_metrics_df['model'].unique()
    ]

    metrics_global = pd.concat(model_dfs, ignore_index=True)

    return metrics_global


def save_results(leave_one_out_metrics, dataset_name, feature_selection, performance_metrics_df,
                 tune_hyperparameters, is_classification, timestamp, feature_names, dataset_dir):
    """
    Guarda los resultados y la configuración del experimento en un directorio por fecha y dataset.

    Args:
        leave_one_out_metrics (pd.DataFrame): Métricas agregadas por modelo.
        dataset_name (str): Nombre del dataset usado.
        feature_selection (bool): Si se aplicó selección de características.
        performance_metrics_df (pd.DataFrame): Métricas por fold.
        tune_hyperparameters (bool): Si se usó GridSearchCV.
        is_classification (bool): Si es una tarea de clasificación.
        timestamp (str): Marca de tiempo para la carpeta (formato "%Y-%m-%d_%H%M").
        feature_names (list): Nombres de las características originales.
        target_column (str): Nombre de la columna objetivo.
    """

    # Guardar métricas por fold
    performance_metrics_df.to_csv(os.path.join(dataset_dir, "folds.csv"), index=False)

    # Guardar métricas globales
    leave_one_out_metrics.to_csv(os.path.join(dataset_dir, "summary.csv"), index=False)

    # Guardar configuración
    config = {
        "dataset": dataset_name,
        "feature_selection": feature_selection,
        "perform_pca": False,
        "tune_hyperparameters": tune_hyperparameters,
        "is_classification": is_classification,
        "timestamp": timestamp,
        "n_folds": len(performance_metrics_df),
        "feature_names": feature_names.tolist(),
    }

    with open(os.path.join(dataset_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    logging.info(f"Results saves in: {dataset_dir}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    is_classification, target_cols = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    for target_col in target_cols:
        logging.info(f"Running {target_col}...")
        for dataset_name in DATASETS:
            logging.info(f"Processing dataset: {dataset_name}")

            run_experiment(dataset_name,
                           PERFORM_FEATURE_SELECTION,
                           MODEL_OUTER_SEED,
                           MODEL_INNER_SEED,
                           is_classification,
                           target_col, timestamp,
                           tune_hyperparameters=TUNE_HYPERPARAMETERS)


def run_experiment(dataset_name, feature_selection, global_seed, inner_cv_seed, is_classification, target_col, timestamp, tune_hyperparameters):
    logging.info(f"Running {target_col}...")
    logging.info(f"Processing dataset: {dataset_name}")
    performance_metrics_df, feature_names = perform(
        dataset_name=dataset_name,
        global_seed=global_seed,
        inner_cv_seed=inner_cv_seed,
        feature_selection=feature_selection,
        tune_hyperparameters=tune_hyperparameters,
        target_col=target_col,
        is_classification=is_classification
    )
    leave_one_out_metrics_df = calculate_metrics_leave_one_out(performance_metrics_df, is_classification)

    base_dir = CLASSIFICATION_RESULTS_DIR if is_classification else REGRESSION_RESULTS_DIR
    dataset_dir = os.path.join(base_dir, timestamp, target_col, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 5) Persist standard outputs (folds + summary + config)
    save_results(leave_one_out_metrics_df, dataset_name, feature_selection, performance_metrics_df,
                 tune_hyperparameters, is_classification, timestamp, feature_names, dataset_dir
                 )


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

    if is_classification:
        target_cols = [CLASSIFICATION_TARGET]
    else:
        if args.target_col is None:
            target_cols = REGRESSION_TARGETS
        else:
            target_cols = [args.target_col]

    return is_classification, target_cols


if __name__ == "__main__":
    main()
