import ast
import os

import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_SELECTED_FEATURES, CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR
from src.model.run_models import retrieve_dataset, get_models


def _fresh_estimator(model_name, global_seed, is_classification):
    models = get_models(random_state=global_seed, is_classification=is_classification)
    for m in models:
        if m.__class__.__name__ == model_name:
            return m.__class__(**m.get_params())
    raise ValueError(f"Estimator '{model_name}' not found in model zoo.")


def _build_pipeline(model, is_classification, feature_selection, X_train_shape, select_score_func):
    step_name = 'classifier' if is_classification else 'regressor'
    select_step = (
        ('select', SelectKBest(score_func=select_score_func, k=min(MAX_SELECTED_FEATURES, X_train_shape[1])))
        if feature_selection else ('select_noop', 'passthrough')
    )
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        select_step,
        ('scaler', StandardScaler()),
        (step_name, model),
    ])
    return pipe, step_name


def parse_hparams(s):
    d = ast.literal_eval(s)
    if not isinstance(d, dict):
        raise ValueError("Hyperparameters string is not a valid dictionary.")
    else:
        return d


def shap_after_nested_cv(
        dataset_name: str,
        target_col,
        is_classification: bool,
        feature_selection: bool,
        global_seed: int,
        model_name_to_explain: str,
        folds_csv_path: str
):
    """
    Refit per outer LOO fold with stored hyperparameters,
    compute SHAP for that fold's test sample(s), and return:
      shap_values_df: DataFrame [samples x features]
      mean_abs_shap: Series (mean |SHAP| per feature)
    """
    X, y, feature_names = retrieve_dataset(dataset_name, target_col, is_classification)
    feature_names = np.array(feature_names)

    folds_df = load_folds_info(folds_csv_path, model_name_to_explain)

    loo = LeaveOneOut()
    fold_splits = list(loo.split(X, y))

    if len(fold_splits) != len(folds_df):
        raise RuntimeError("Current LOO fold count and folds.csv rows differ. "
                           "Persist and reuse train/test indices to guarantee identity.")

    select_score_func = f_classif if is_classification else f_regression

    explanations = []
    for fold_id, ((train_idx, test_idx), fold_row) in enumerate(zip(fold_splits, folds_df.itertuples(index=False))):
        if fold_id != fold_row.fold:
            raise RuntimeError(f"Fold order mismatch at fold {fold_id} vs {fold_row.fold}.")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        estimator = _fresh_estimator(model_name_to_explain, global_seed, is_classification)
        pipeline, estimator_step_name = _build_pipeline(estimator, is_classification, feature_selection, X_train.shape,
                                                        select_score_func)
        # Apply per-fold estimator params
        final_est = pipeline.named_steps[estimator_step_name]
        final_est.set_params(**fold_row.hyperparameters)
        pipeline.named_steps[estimator_step_name] = final_est

        # Fit on training portion of this fold
        pipeline.fit(X_train, y_train)

        expl = compute_shap_for_pipeline(X_test, X_train, estimator_step_name, feature_names, feature_selection,
                                         pipeline, is_classification)

        explanations.append(expl)

    return explanations


def _callable_for_shap(estimator, is_classification):
    if is_classification:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba
        else:
            raise ValueError("At the moment, only classifiers with predict_proba are supported for SHAP.")
    else:
        return estimator.predict


def compute_shap_for_pipeline(X_test, X_train, estimator_step_name, feature_names, feature_selection, pipeline,
                              is_classification=True):
    # SHAP computation
    preprocess = pipeline[:-1]
    estimator = pipeline.named_steps[estimator_step_name]
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
    # Map names after SelectKBest
    if feature_selection:
        support_idx = pipeline.named_steps['select'].get_support(indices=True)
        shap_feature_names = np.asarray(feature_names)[support_idx]
    else:
        shap_feature_names = np.asarray(feature_names)

    f_callable = _callable_for_shap(estimator, is_classification)
    explainer = shap.Explainer(f_callable, X_train_transformed, feature_names=shap_feature_names)

    kind = f"{explainer.__module__}.{explainer.__class__.__name__}"
    link = getattr(explainer.link, "__class__", type(explainer.link)).__name__
    print(f"[SHAP] backend: {kind} | link: {link}")
    # Explain the test sample(s)
    return explainer(X_test_transformed)  # Explanation


def load_folds_info(folds_csv_path, model_name_to_explain):
    folds_df = pd.read_csv(folds_csv_path)
    folds_df = folds_df[folds_df['model'] == model_name_to_explain].copy()
    if folds_df.empty:
        raise ValueError(f"No rows for model '{model_name_to_explain}' in {folds_csv_path}")
    # Parse hyperparameters (stringified dict → dict)
    if folds_df['hyperparameters'].dtype == object:
        folds_df['hyperparameters'] = folds_df['hyperparameters'].apply(
            lambda s: parse_hparams(s)
        )
    folds_df = folds_df.sort_values('fold').reset_index(drop=True)
    return folds_df


from src.config import MODEL_OUTER_SEED


def run_shap(task: str, target_col: str, dataset_name: str, timestamp: str, model_name_to_explain: str):
    if task != 'classification' and task != 'regression':
        raise ValueError("task must be 'classification' or 'regression'")

    is_classification = task == 'classification'

    if is_classification and target_col != 'group':
        raise ValueError("For classification, target_col must be 'group'")

    results_dir = CLASSIFICATION_RESULTS_DIR if is_classification else REGRESSION_RESULTS_DIR

    folds_path = os.path.join(
        results_dir,
        timestamp,
        target_col,
        dataset_name,
        "folds.csv"
    )

    shap_explanations = shap_after_nested_cv(
        dataset_name=dataset_name,
        target_col=target_col,
        is_classification=is_classification,
        feature_selection=True,
        global_seed=MODEL_OUTER_SEED,
        model_name_to_explain=model_name_to_explain,
        folds_csv_path=folds_path,
    )

    print(shap_explanations)

    return shap_explanations


if __name__ == "__main__":
    run_shap(
        dataset_name="demographic+digital",
        target_col="group",
        task="classification",
        model_name_to_explain="SVC",
        timestamp="2025-09-12_1559"
    )
