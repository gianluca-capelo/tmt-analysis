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
