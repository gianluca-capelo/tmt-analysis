import os
import ast
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.config import CLASSIFICATION_RESULTS_DIR


def plot_all_datasets_roc(date_folder: str, save_path: str = None):
    """
    Genera una gráfica de curvas ROC para todos los datasets a partir de los resultados *_leave_one_out.csv.

    Parameters
    ----------
    date_folder : str
        Nombre de la carpeta de fecha (por ejemplo '2025-08-06') donde están los resultados.
    save_path : str, optional
        Ruta para guardar la imagen. Si None, solo se muestra.
    """
    results_dir = os.path.join(CLASSIFICATION_RESULTS_DIR, date_folder)
    leave_one_out_files = glob.glob(os.path.join(results_dir, "*_leave_one_out.csv"))

    plt.figure(figsize=(10, 8))

    for file_path in leave_one_out_files:
        df = pd.read_csv(file_path)

        # Convierte strings en listas
        df['y_true'] = df['y_true'].apply(ast.literal_eval)
        df['y_pred_proba'] = df['y_pred_proba'].apply(ast.literal_eval)

        dataset_name = os.path.basename(file_path).replace("_leave_one_out.csv", "")

        for _, row in df.iterrows():
            model_name = row['model']
            y_true = row['y_true']
            y_pred_proba = row['y_pred_proba']

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1.5, label=f"{dataset_name} | {model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Todos los Datasets")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # Ejemplo de uso
    date_folder = "2025-08-06"  # Cambia esto por la fecha deseada
    save_path = os.path.join(CLASSIFICATION_RESULTS_DIR, f"{date_folder}_roc_curves.png")

    plot_all_datasets_roc(date_folder, save_path)
    print(f"ROC curves saved to {save_path}")