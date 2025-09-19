import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import shap

from src.config import MAX_SELECTED_FEATURES
from src.model.run_models import retrieve_dataset, get_models


def _fresh_estimator(models, model_name):
    for m in models:
        if m.__class__.__name__ == model_name:
            return m.__class__(**m.get_params())
    raise ValueError(f"Estimator '{model_name}' not found in model zoo.")

def _build_pipeline_no_pca(model, is_classification, feature_selection, X_train_shape, select_score_func):
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

#TODO GIAN: Ver si pasar esto a shap
# def _callable_for_shap(fitted_pipeline, is_classification, positive_class_index=1):
#     if is_classification:
#         if hasattr(fitted_pipeline, "predict_proba"):
#             return lambda X: fitted_pipeline.predict_proba(X)[:, positive_class_index]
#         elif hasattr(fitted_pipeline, "decision_function"):
#             return lambda X: fitted_pipeline.decision_function(X)
#         else:
#             return lambda X: fitted_pipeline.predict(X)  # last resort
#     else:
#         return lambda X: fitted_pipeline.predict(X)

def parse_hparams(s):
    if isinstance(s, dict):
        return s
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    s = s.strip()
    # 1) JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) Python literal (maneja comillas simples, None, True/False)
    try:
        d = ast.literal_eval(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        pass
    # 3) fallback simple
    s2 = (s.replace("'", '"')
            .replace(" None", " null").replace(": None", ": null")
            .replace(" True", " true").replace(": True", ": true")
            .replace(" False", " false").replace(": False", ": false"))
    return json.loads(s2)


def shap_after_nested_cv(
    dataset_name: str,
    target_col,
    is_classification: bool,
    feature_selection: bool,
    global_seed: int,
    model_name_to_explain: str,
    folds_csv_path: str,
    positive_class_index: int = 1,
    background_cap: int = 100,
    # If you stored indices originally, pass them here to guarantee identical folds:
    stored_indices_csv: str = None
):
    """
    Refit per outer LOO fold with stored hyperparameters (no PCA),
    compute SHAP for that fold's test sample(s), and return:
      shap_values_df: DataFrame [samples x features]
      mean_abs_shap: Series (mean |SHAP| per feature)
    """
    X, y, feature_names = retrieve_dataset(dataset_name, target_col, is_classification)
    feature_names = np.array(feature_names)
    models = get_models(global_seed, is_classification)

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

        est = _fresh_estimator(models, model_name_to_explain)
        pipe, step_name = _build_pipeline_no_pca(
            est, is_classification, feature_selection, X_train.shape, select_score_func
        )
        # Apply per-fold estimator params
        final_est = pipe.named_steps[step_name]
        final_est.set_params(**fold_row.hyperparameters)
        pipe.named_steps[step_name] = final_est

        # Fit on training portion of this fold
        pipe.fit(X_train, y_train)

        # Model-agnostic explainer on original features
        #TODO GIAN: ver si usar background

        # # Map names after SelectKBest
        # if feature_selection and 'select' in pipe.named_steps:
        #     support_idx = pipe.named_steps['select'].get_support(indices=True)
        #     shap_feature_names = np.asarray(feature_names)[support_idx]
        #     print(f"Fold {fold_id}: selected {len(shap_feature_names)}/{len(feature_names)} features for SHAP.")
        #     print(f"Selected features: {shap_feature_names}")
        # else:
        #     shap_feature_names = np.asarray(feature_names)

        explainer = shap.Explainer(pipe.predict, feature_names=feature_names)

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


if __name__ == "__main__":
    from src.config import MODEL_OUTER_SEED

    shap_after_nested_cv(
        dataset_name="demographic+digital",
        target_col="group",
        is_classification=True,
        feature_selection=True,
        global_seed=MODEL_OUTER_SEED,
        model_name_to_explain="SVC",
        folds_csv_path="/home/gianluca/Research/tmt-analysis/results/classification/2025-09-12_1559/group/demographic+digital/folds.csv",
    )