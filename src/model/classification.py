import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROCESSED_FOR_MODEL_DIR, CLASSIFICATION_RESULTS_DIR


def extract_X_y_features(df):
    df = df.drop('subject_id', axis=1)
    print("group in X", 'group' in df.iloc[:, :-1].columns)
    print("suj in X", 'subject_id' in df.iloc[:, :-1].columns)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]
    assert 'subject_id' not in df.columns, "'subject_id' still in the final dataframe"
    return X, y, feature_names


def join_and_reorder(df1, df2):
    df = pd.merge(
        df1,
        df2.drop(columns='group'),
        on='subject_id',
        how='inner'
    )
    cols = [col for col in df.columns if col != 'group'] + ['group']
    df = df[cols]
    assert df.columns[-1] == 'group', "'group' is not the last column after reordering"
    return df


def retrieve_dataset(dataset_name):
    processed_path = PROCESSED_FOR_MODEL_DIR + '/'
    # Read data
    df_digital_tmt_with_target = pd.read_csv(processed_path + 'df_digital_tmt_with_target.csv')
    demographic_df = pd.read_csv(processed_path + 'demographic_df.csv')
    non_digital_df = pd.read_csv(processed_path + 'non_digital_df.csv')
    df_digital_hand_and_eye = pd.read_csv(processed_path + 'df_digital_hand_and_eye.csv')
    digital_test_less_subjects = pd.read_csv(processed_path + 'digital_test_less_subjects.csv')
    non_digital_test_less_subjects = pd.read_csv(processed_path + 'non_digital_test_less_subjects.csv')

    match dataset_name:
        case 'demographic':
            X, y, feature_names = extract_X_y_features(demographic_df)

        case 'demographic_less_subjects':
            df = demographic_df.loc[df_digital_hand_and_eye.index]
            X, y, feature_names = extract_X_y_features(df)

        case 'demographic+digital':
            df = join_and_reorder(df_digital_tmt_with_target, demographic_df)
            X, y, feature_names = extract_X_y_features(df)

        case 'demographic+digital_less':
            df = join_and_reorder(df_digital_tmt_with_target.loc[df_digital_hand_and_eye.index], demographic_df)
            X, y, feature_names = extract_X_y_features(df)

        case 'non_digital_tests':
            X, y, feature_names = extract_X_y_features(non_digital_df)

        case 'non_digital_tests+demo':
            df = join_and_reorder(non_digital_df.drop(columns='subject_id'), demographic_df)
            X, y, feature_names = extract_X_y_features(df)

        case 'non_digital_test_less_subjects':
            X, y, feature_names = extract_X_y_features(non_digital_test_less_subjects)

        case 'non_digital_test_less_subjects+demo':
            df = join_and_reorder(non_digital_test_less_subjects.drop(columns='subject_id'), demographic_df)
            X, y, feature_names = extract_X_y_features(df)

        case 'digital_test':
            X, y, feature_names = extract_X_y_features(df_digital_tmt_with_target)

        case 'digital_test_less_subjects':
            X, y, feature_names = extract_X_y_features(digital_test_less_subjects)

        case 'hand_and_eye':
            X, y, feature_names = extract_X_y_features(df_digital_hand_and_eye)

        case 'hand_and_eye_demo':
            df = join_and_reorder(df_digital_hand_and_eye, demographic_df)
            X, y, feature_names = extract_X_y_features(df)

        case _:
            raise ValueError(f'Please select a valid dataset')

    return X, y, feature_names


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


def get_models():
    return [
        RandomForestClassifier(random_state=42, n_jobs=-1),
        SVC(random_state=42, probability=True, kernel='linear'),
        LogisticRegression(max_iter=1000, random_state=42, solver='saga', n_jobs=-1),
        xgb.XGBClassifier(random_state=42, tree_method="hist", eval_metric='logloss', n_jobs=-1)
    ]


def get_cv(cv_type: str, n_splits: int = 5, n_repeats: int = 10, global_seed: int = 42):
    match cv_type:
        case 'stratified':
            print(f"RepeatedStratifiedKFold selected with n_splits = {n_splits} and n_repeats = {n_repeats}")
            outer_cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=global_seed  # Global seed
            )
        case 'loo':
            print("LeaveOneOut selected")
            outer_cv = LeaveOneOut()
        case _:
            raise ValueError(f"Invalid cv_type: {cv_type}. Choose 'stratified' or 'loo'.")

    return outer_cv


def perform(perform_pca: bool, dataset_name: str, cv_type: str, n_splits: int, n_repeats: int, global_seed: int,
            inner_cv_seed: int, feature_selection: bool, tune_hyperparameters: bool):
    X, y, feature_names = retrieve_dataset(dataset_name)

    unique, counts = np.unique(y, return_counts=True)

    print("Class distribution:", dict(zip(unique, counts)))

    param_grids = get_parameter_grid()

    models = get_models()

    outer_cv = get_cv(cv_type, n_splits, n_repeats, global_seed)

    performance_metrics_df = pd.DataFrame(columns=get_performance_metrics())

    perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca, feature_selection, tune_hyperparameters,
                             performance_metrics_df, inner_cv_seed, feature_names)

    # save
    save_results(dataset_name, feature_selection, perform_pca, performance_metrics_df, tune_hyperparameters)


def save_results(dataset_name, feature_selection, perform_pca, performance_metrics_df, tune_hyperparameters):
    classification_dir = CLASSIFICATION_RESULTS_DIR
    save_dir = os.path.join(classification_dir, datetime.now().strftime("%Y-%m-%d"))

    os.makedirs(save_dir, exist_ok=True)

    if perform_pca:
        save_path = f'{save_dir}/{dataset_name}_feature_{feature_selection}_tune={tune_hyperparameters}_LOOCV_PCA_{datetime.now().strftime("%s")[-4:]}.csv'
    else:
        save_path = f'{save_dir}/{dataset_name}_feature_{feature_selection}_tune={tune_hyperparameters}_LOOCV_{datetime.now().strftime("%s")[-4:]}.csv'

    performance_metrics_df.to_csv(save_path, index=False)


def get_performance_metrics():
    return [
        'model', 'repeat', 'fold',
        'accuracy', 'balanced_accuracy', 'precision',
        'recall', 'f1', 'auc', 'specificity'
    ]


def perform_cross_validation(param_grids, models, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                             tune_hyperparameters: bool, performance_metrics_df, inner_cv_seed: int,
                             feature_names):
    for model in models:
        model_name = model.__class__.__name__

        print(f"\n🧪 CV for: {model_name}")

        param_grid = param_grids.get(model_name, {})

        performance_metrics_df = perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca,
                                                                    feature_selection,
                                                                    tune_hyperparameters, performance_metrics_df,
                                                                    inner_cv_seed,
                                                                    feature_names)


def perform_cross_validation_for_model(param_grid, model, outer_cv, X, y, perform_pca: bool, feature_selection: bool,
                                       tune_hyperparameters: bool, performance_metrics_df, inner_cv_seed: int,
                                       feature_names):
    model_name = model.__class__.__name__
    print(f"\n🧪 CV for: {model_name}")

    tprs, aucs, best_params_list, fold_metrics = [], [], [], []

    all_y_true, all_y_pred = [], []

    # Enumeramos 'repeat' y 'fold' para guardar en métricas
    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        n_features = 20
        n_components = 4
        fold = outer_idx  # index of the left-out observation
        print('fold:', fold)

        # ── Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Inner CV: estratificado 3-fold con la MISMA semilla por repetición
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_cv_seed)

        if perform_pca:
            n_components = min(n_components, X_train.shape[1])
            print("n_components:", n_components)
            pca_step = ('pca', PCA(n_components=n_components))
        else:
            pca_step = ('noop', 'passthrough')

        if feature_selection:
            n_features = min(n_features, X_train.shape[1])
            print("n_features:", n_features)
            feature_selection_step = ('select', SelectKBest(score_func=f_classif, k=n_features))
        else:
            feature_selection_step = ('noop', 'passthrough')

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # or 'median' depending on your data
            feature_selection_step,
            ('scaler', StandardScaler()),
            pca_step,
            ('classifier', model)
        ])

        if tune_hyperparameters and param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params_list.append(grid.best_params_)
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params_list.append("no tuning")

        # Only compute importance if PCA is OFF
        importance_dict = {}
        if not perform_pca:
            # Extract pipeline steps
            try:
                classifier = best_model.named_steps['classifier']
                select = best_model.named_steps['select']
                selected_features = select.get_support(indices=True) if feature_selection else np.arange(
                    X_train.shape[1])
                selected_feature_names = feature_names[selected_features]

                if hasattr(classifier, 'feature_importances_'):  # RandomForest, XGBoost
                    importances = classifier.feature_importances_
                    importance_dict = dict(zip(selected_feature_names, importances))

                elif model_name == 'LogisticRegression':
                    importances = np.abs(classifier.coef_).flatten()
                    importance_dict = dict(zip(selected_feature_names, importances))

                elif model_name == 'SVC' and classifier.kernel == 'linear':
                    importances = np.abs(classifier.coef_).flatten()
                    importance_dict = dict(zip(selected_feature_names, importances))

            except Exception as e:
                print(f"❌ Could not extract feature importance for {model_name} in fold {fold}: {e}")

        # ── Predicción
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold_metrics.append({
            'model': model_name,
            'fold': fold,
            'y_test': y_test[0],
            'y_pred': y_pred[0],
            'y_pred_proba': y_pred_proba[0],
            'feature_importances': importance_dict,
            'feature_names': feature_names.values
        })

    print(fold_metrics)
    # ── Guardamos métricas
    performance_metrics_df = pd.concat([performance_metrics_df,
                                        pd.DataFrame(fold_metrics)],
                                       ignore_index=True)

    return performance_metrics_df


def main():
    # Example usage
    n_splits = 2
    n_repeats = 1
    global_seed = 42
    inner_cv_seed = 50  # Fixed for reproducibility in inner CV
    type_of_cv = 'loo'
    tune_hyperparameters = False
    feature_selection = True
    perform(
        perform_pca=False,
        dataset_name='demographic+digital',
        cv_type=type_of_cv,
        n_splits=n_splits,
        n_repeats=n_repeats,
        global_seed=global_seed,
        inner_cv_seed=inner_cv_seed,
        feature_selection=feature_selection,
        tune_hyperparameters=tune_hyperparameters
    )


if __name__ == "__main__":
    main()
