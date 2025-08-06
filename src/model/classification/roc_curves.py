import ast
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.config import CLASSIFICATION_RESULTS_DIR


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
    results_dir = os.path.join(CLASSIFICATION_RESULTS_DIR, date_folder)
    leave_one_out_files = glob.glob(os.path.join(results_dir, "*_leave_one_out.csv"))

    auc_list = []

    for file_path in leave_one_out_files:
        dataset_name = os.path.basename(file_path).replace("_leave_one_out.csv", "")

        # Si hay filtro de datasets, aplicarlo
        if datasets_filter and dataset_name not in datasets_filter:
            continue

        df = pd.read_csv(file_path)
        df['y_true'] = df['y_true'].apply(ast.literal_eval)
        df['y_pred_proba'] = df['y_pred_proba'].apply(ast.literal_eval)

        for _, row in df.iterrows():
            model_name = row['model']
            y_true = row['y_true']
            y_pred_proba = row['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_list.append({
                "dataset": dataset_name,
                "model": model_name,
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            })

    # Ordenar por AUC y quedarnos con los top_n
    top_auc = sorted(auc_list, key=lambda x: x["auc"], reverse=True)[:top_n]

    # Graficar
    plt.figure(figsize=(10, 8))
    for entry in top_auc:
        plt.plot(entry["fpr"], entry["tpr"], lw=1.5,
                 label=f"{entry['dataset']} | {entry['model']} (AUC = {entry['auc']:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Top {top_n} modelos/datasets")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    date_folder = "2025-08-06"
    save_path = os.path.join(CLASSIFICATION_RESULTS_DIR, f"{date_folder}_roc_curves.png")
    datasets_filter = [
        'demographic',
        #'demographic_less_subjects',
        #'demographic+digital',
        #'demographic+digital_less',
        #'non_digital_tests',
        #'non_digital_tests+demo',
        #'non_digital_test_less_subjects',
        #'non_digital_test_less_subjects+demo',
        'digital_test',
        'digital_test_less_subjects',
        #'hand_and_eye',
        #'hand_and_eye_demo'
    ]

    plot_top_n_datasets_roc(date_folder, 20, save_path, datasets_filter)

    print(f"ROC curves saved to {save_path}")
