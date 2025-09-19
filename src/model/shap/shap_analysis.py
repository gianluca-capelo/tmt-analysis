import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import MAX_SELECTED_FEATURES


def perform_shap_cross_validation_for_model(model, cv_folds_df: pd.DataFrame, outer_cv, X, y, feature_selection: bool,
                                            feature_names,
                                            is_classification):
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

    model_cv_folds_df = cv_folds_df[cv_folds_df['model'] == model_name].copy()

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
