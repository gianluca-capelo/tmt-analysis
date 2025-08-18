from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
import pandas as pd
import numpy as np
from pathlib import Path
import re
import ast

def calculate_metrics_leave_one_out(df, model_name, plot_roc=False):
    model_df = df[df['model'] == model_name]
    y_true = model_df['y_test'].tolist()
    y_pred_proba = model_df['y_pred_proba'].tolist()
    y_pred = model_df['y_pred'].tolist()

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = np.nan

    # Aggregate feature importances across folds
    importance_agg = {}
    try:
        all_importances = [ast.literal_eval(row) for row in model_df['feature_importances'].dropna()]
        total = len(all_importances)
        if total > 0:
            sum_importance = {}
            for imp in all_importances:
                for k, v in imp.items():
                    sum_importance[k] = sum_importance.get(k, 0) + v
            importance_agg = {k: v / total for k, v in sum_importance.items()}
    except Exception as e:
        print(f"⚠️ Error parsing feature importances for {model_name}: {e}")
        importance_agg = {}

    return pd.DataFrame({
        'model': [model_name],
        'auc': [auc],
        'accuracy': [accuracy_score(y_true, y_pred)],
        'balanced_accuracy': [balanced_accuracy_score(y_true, y_pred)],
        'precision': [precision_score(y_true, y_pred, zero_division=0)],
        'recall': [recall_score(y_true, y_pred, zero_division=0)],
        'f1': [f1_score(y_true, y_pred, zero_division=0)],
        'y_true': [y_true],
        'y_pred_proba': [y_pred_proba],
        'feature_importances': [importance_agg]
    })

# ───────────────────────────────────────────────────────────────
# File loading
# ───────────────────────────────────────────────────────────────

dir2 = Path('./results/modelling/classification/2025-07-08')
file_paths = [f.resolve() for f in dir2.rglob('*') if f.is_file()]

all_dataset_metrics = []

for path in file_paths:
    print(path)
    pattern = re.compile(r"(.*?)_LOOCV")
    match = pattern.search(path.stem)
    dataset = match.group(1)
    print(dataset)

    all_metrics_df = pd.read_csv(path)

    model_dfs = [
        calculate_metrics_leave_one_out(all_metrics_df, model_name)
        for model_name in all_metrics_df['model'].unique()
    ]

    metrics_global = pd.concat(model_dfs, ignore_index=True)
    metrics_global['dataset'] = dataset
    metrics_global['PCA'] = 'PCA' in str(path)
    metrics_global['features'] = 'feature' in str(path)
    metrics_global['complete'] = not (('less' in str(path)) or ('eye' in str(path)))

    all_dataset_metrics.append(metrics_global)

all_datasets_df = pd.concat(all_dataset_metrics, ignore_index=True)

all_datasets_df