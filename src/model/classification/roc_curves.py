import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.config import CLASSIFICATION_RESULTS_DIR, DATASETS_PLOT, DATASETS_PLOT_FOLDER
from src.model.permutation_tests import permutation_test


def plot_top_n_datasets_roc(date_folder: str, top_n: int = 5, save_path: str = None, datasets_filter: list = None):
    """
    Genera una gráfica de curvas ROC para los top N modelos/datasets según el AUC promedio.
    Permite filtrar por datasets específicos.

    Parameters
    ----------
    date_folder : str
        Carpeta de fecha (por ejemplo '2025-08-06') donde están los resultados *_leave_one_out.csv.
    top_n : int
        Número de curvas con mejor AUC a mostrar.
    save_path : str, optional
        Ruta para guardar la imagen. Si None, solo se muestra.
    datasets_filter : list, optional
        Lista con los nombres exactos de datasets a incluir (coinciden con los usados en save_results).
        Si None, se usan todos los datasets.
    """
    results_dir = Path(os.path.join(CLASSIFICATION_RESULTS_DIR, date_folder))

    # ───────────────────────────────────────────────────────────────
    # Load all summary.csv into one dataframe
    # ───────────────────────────────────────────────────────────────
    summary_paths = [f for f in results_dir.rglob("summary.csv")
                     if (datasets_filter is None) or (f.parent.name in datasets_filter)]

    all_summary_dfs = []

    for path in summary_paths:
        df = pd.read_csv(path)
        dataset = path.parent.name
        df['dataset'] = dataset
        df['parent_date'] = path.parents[1].name  # e.g. 2025-08-19_1224
        all_summary_dfs.append(df)

    all_summaries = pd.concat(all_summary_dfs, ignore_index=True)

    # ───────────────────────────────────────────────────────────────
    # Get best AUC per dataset
    # ───────────────────────────────────────────────────────────────
    best_per_dataset = []
    for dataset, group in all_summaries.groupby("dataset"):
        best_idx = None
        best_auc = -1
        for idx, row in group.iterrows():
            try:
                y_true = eval(row['y_true']) if isinstance(row['y_true'], str) else row['y_true']
                y_pred_proba = eval(row['y_pred_proba']) if isinstance(row['y_pred_proba'], str) else row[
                    'y_pred_proba']
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
            except Exception:
                continue
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_idx = idx
        if best_idx is not None:
            best_per_dataset.append(all_summaries.loc[best_idx])

    best_df = pd.DataFrame(best_per_dataset)

    # ───────────────────────────────────────────────────────────────
    # Define groups, colors, and linestyles
    # ───────────────────────────────────────────────────────────────
    group_map = {
        'Non-digital': 'non_digital',
        'Non-digital + Demographic': 'non_digital',
        'Demographic': 'demographic',
        'Demographic + Digital': 'digital',
        'Digital': 'digital'
    }

    # Color-blind friendly palette
    group_colors = {
        'non_digital': '#1f77b4',  # blue
        'demographic': '#2ca02c',  # green
        'digital': '#d62728'  # red
    }

    group_linestyles = {
        'non_digital': 'solid',
        'demographic': 'solid',
        'digital': 'solid'
    }

    # ───────────────────────────────────────────────────────────────
    # Plot ROC curves
    # ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 7))

    map_dataset_names = {
        'demographic': 'Demographic',
        'demographic+digital': 'Demographic + Digital',
        'digital_test': 'Digital',
        'non_digital_tests': 'Non-digital',
        'non_digital_tests+demo': 'Non-digital + Demographic',
    }

    best_df['dataset'] = best_df['dataset'].map(map_dataset_names)
    print(best_df['dataset'].unique())

    for idx, row in best_df.iterrows():
        y_true = eval(row['y_true']) if isinstance(row['y_true'], str) else row['y_true']
        y_pred_proba = eval(row['y_pred_proba']) if isinstance(row['y_pred_proba'], str) else row['y_pred_proba']
        auc_score, p_val = permutation_test(y_true, y_pred_proba, n_permutations=1000)
        print(f"{row['model']} on {row['dataset']} → AUC = {auc_score:.3f}, p = {p_val:.4f}")

        dataset = row['dataset']
        group = group_map.get(dataset, "other")

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr,
                 color=group_colors[group],
                 linestyle=group_linestyles[group],
                 lw=2.5,
                 label=f"{dataset} (AUC={roc_auc:.2f})")

    # Diagonal reference
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')

    # Formatting for paper
    plt.title("Best ROC per Dataset", fontsize=16, weight='bold')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=11, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    date_folder = DATASETS_PLOT_FOLDER
    save_path = os.path.join(CLASSIFICATION_RESULTS_DIR, date_folder, "classification_roc_curves.png")
    datasets_filter = DATASETS_PLOT or None

    plot_top_n_datasets_roc(date_folder, 20, save_path, datasets_filter)

    print(f"ROC curves saved to {save_path}")
