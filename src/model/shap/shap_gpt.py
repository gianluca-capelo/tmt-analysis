import ast

import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_SELECTED_FEATURES
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


# TODO GIAN: usar predict o predict proba????
def _callable_for_shap(fitted_pipeline, is_classification):
    if is_classification:
        if hasattr(fitted_pipeline, "predict_proba"):
            return lambda X: fitted_pipeline.predict_proba(X)[:, 1]
        else:
            raise ValueError("")
    else:
        return lambda X: fitted_pipeline.predict(X)


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
    Refit per outer LOO fold with stored hyperparameters (no PCA),
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

    rows = []
    row_index = []
    for fold_id, ((train_idx, test_idx), fold_row) in enumerate(zip(fold_splits, folds_df.itertuples(index=False))):
        if fold_id != fold_row.fold:
            raise RuntimeError(f"Fold order mismatch at fold {fold_id} vs {fold_row.fold}.")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        estimator = _fresh_estimator(model_name_to_explain, global_seed, is_classification)
        pipe, step_name = _build_pipeline(estimator, is_classification, feature_selection, X_train.shape,
                                          select_score_func)
        # Apply per-fold estimator params
        final_est = pipe.named_steps[step_name]
        final_est.set_params(**fold_row.hyperparameters)
        pipe.named_steps[step_name] = final_est

        # Fit on training portion of this fold
        pipe.fit(X_train, y_train)

        # TODO GIAN: ver si usar background y si esta bien usar todo como background
        background_X = X_train

        # Build a callable on original features
        f = _callable_for_shap(pipe, is_classification)

        # Create explainer (model-agnostic)
        masker = shap.maskers.Independent(background_X)
        explainer = shap.Explainer(f, masker=masker, feature_names=feature_names)

        # Explain the test sample(s)
        expl = explainer(X_test)  # Explanation
        vals = np.array(expl.values).reshape(len(X_test), len(feature_names))

        rows.append(vals)
        # index = (fold_id, each test sample index)
        ti = test_idx if isinstance(test_idx, (list, np.ndarray)) else np.array([test_idx])
        row_index.extend([(fold_id, int(i)) for i in ti])

    # Assemble per-sample SHAP
    shap_values = np.vstack(rows) if len(rows) else np.empty((0, len(feature_names)))
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_values_df.index = pd.MultiIndex.from_tuples(row_index, names=["fold", "test_idx"])

    # Aggregate: mean absolute SHAP per feature
    mean_abs_shap = shap_values_df.abs().mean(axis=0).sort_values(ascending=False)

    return shap_values_df, mean_abs_shap


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


def run_shap():
    from src.config import MODEL_OUTER_SEED
    shap_values_df, mean_abs_shap = shap_after_nested_cv(
        dataset_name="demographic+digital",
        target_col="group",
        is_classification=True,
        feature_selection=True,
        global_seed=MODEL_OUTER_SEED,
        model_name_to_explain="SVC",
        folds_csv_path="/home/gianluca/Research/tmt-analysis/results/classification/2025-09-12_1559/group/demographic+digital/folds.csv",
    )

    print(mean_abs_shap)


if __name__ == "__main__":
    run_shap()
