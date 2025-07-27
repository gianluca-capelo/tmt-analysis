import logging

import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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


is_classification = False


def perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                                       tune_hyperparameters: bool, inner_cv_seed: int,
                                       feature_names):
    if (is_classification):
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

    # Enumeramos 'repeat' y 'fold' para guardar en métricas
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logging.info(f'Fold number: {fold}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Steps
        pca_step = ('pca', PCA(n_components=min(max_pca_components, X_train.shape[1]))) if perform_pca else ('noop',
                                                                                                             'passthrough')

        select_step = ('select',
                       SelectKBest(score_func=select_score_func, k=min(max_selected_features, X_train.shape[1]))) \
            if feature_selection else ('noop', 'passthrough')

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            select_step,
            ('scaler', StandardScaler()),
            pca_step,
            (pipeline_name, model)
        ])

        if tune_hyperparameters and param_grid:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_cv_seed) if is_classification \
                else (KFold(n_splits=3, shuffle=True, random_state=inner_cv_seed))
            grid = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, scoring='r2', n_jobs=-1, verbose=0)
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
            logging.info(f"Skipping feature importance for {model_name} in fold {fold} due to PCA.")

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
