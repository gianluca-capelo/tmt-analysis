import os

import pandas as pd


def load_all_summaries(base_dir: str, timestamp: str, targets: list, datasets: list) -> dict:
    """
    Carga todos los summary.csv en una estructura de diccionarios anidados:
    { target : { dataset : DataFrame } }

    Args:
        base_dir (str): Directorio raíz (REGRESSION_RESULTS_DIR o CLASSIFICATION_RESULTS_DIR).
        timestamp (str): Marca de tiempo usada en la corrida (ej: "2025-09-12_1030").
        targets (list[str]): Lista de targets (p.ej. REGRESSION_TARGETS).
        datasets (list[str]): Lista de datasets (p.ej. DATASETS).

    Returns:
        dict: Diccionario de diccionarios con DataFrames de pandas.
    """
    results = {}
    for target in targets:
        results[target] = {}
        for dataset in datasets:
            summary_path = os.path.join(base_dir, timestamp, target, dataset, "summary.csv")
            if os.path.exists(summary_path):
                results[target][dataset] = pd.read_csv(summary_path)
            else:
                results[target][dataset] = None  # o podrías saltar con continue
    return results