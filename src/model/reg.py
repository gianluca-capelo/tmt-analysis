import os
import pandas as pd
from typing import List, Dict, Optional

def load_all_summaries_with_pvalues(
    base_dir: str,
    timestamp: str,
    targets: List[str],
    datasets: List[str],
    task: str,                 # "classification" | "regression"
    metric: str               # "auc" | "mae" | "mse" | "r2"
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Carga {target:{dataset: DataFrame}} desde summary.csv y agrega 'p_value'
    mergeando con los resultados de permutación:
      {base}/{timestamp}/{target}/permutation_test_results/{metric}/permutation_test_results_{task}.csv

    Reglas de merge:
      1) Si el summary tiene la columna de la métrica (p.ej. 'mae'): merge por ['dataset','model',metric]
         tras renombrar 'score'->metric en p-values.
      2) Si no, merge por ['dataset','model'] y si hay duplicados en p-values para esa pareja,
         se usa el p_value mínimo (criterio conservador).
    """
    results: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

    pvals_filename = f"permutation_test_results_{task}.csv"

    for target in targets:
        results[target] = {}

        # p-values de este target
        pvals_path = os.path.join(
            base_dir, timestamp, target, "permutation_test_results", metric, pvals_filename
        )
        pvals_df = pd.read_csv(pvals_path).copy()

        colmap = {"Dataset": "dataset", "Model": "model"}
        pvals_df.rename(columns=colmap, inplace=True)

        pvals_df = pvals_df[["dataset", "model", "p_value"]]

        for dataset in datasets:
            summary_path = os.path.join(base_dir, timestamp, target, dataset, "summary.csv")
            if not os.path.exists(summary_path):
                raise ValueError(f"summary.csv no encontrado en {summary_path}")

            df = pd.read_csv(summary_path).copy()

            df['dataset'] = dataset

            df = df.merge(
                pvals_df,
                on=["dataset", "model"],
                how="left",
                validate="m:1"
            )

            df.rename(columns={"p_value": "permutation_test_p_value"}, inplace=True)

            results[target][dataset] = df

    return results
