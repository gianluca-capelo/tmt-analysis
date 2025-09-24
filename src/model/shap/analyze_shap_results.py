from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import shap

def extract_positive_class_explanations(explanations):
    positive_class_explanations = []

    for explanation in explanations:
        vals = explanation.values.copy()
        base = explanation.base_values.copy()
        positive_class_idx = 1
        # Caso 1: (n_samples, n_classes, n_features)
        if vals.ndim == 3 and vals.shape[1] == 2:
            vals = vals[:, positive_class_idx, :]
            base = base[:, positive_class_idx] if base.ndim == 2 else base

        # Caso 2: (n_samples, n_features, n_classes)
        elif vals.ndim == 3 and vals.shape[2] == 2:
            vals = vals[:, :, positive_class_idx]
            base = base[:, positive_class_idx] if base.ndim == 2 else base

        positive_class_explanation = shap.Explanation(
            values=vals,
            base_values=base,
            data=explanation.data,
            feature_names=explanation.feature_names
        )

        positive_class_explanations.append(positive_class_explanation)

    return positive_class_explanations

def build_shap_absolute_df(explanations):
    # Lista de dicts: {feature: abs(shap)}
    rows = []

    for expl in explanations:
        absolute_values = np.abs(expl.values)  # (1, n_features)
        names = expl.feature_names
        # Armo dict con valores de este fold
        row = dict(zip(names, absolute_values.flatten()))
        rows.append(row)

    # DataFrame folds × features (NaN si el feature no apareció en ese fold)
    df = pd.DataFrame(rows)
    return df

def analyze_shap_results(explanations: Iterable[shap.Explanation]) -> pd.DataFrame:
    """
    Analyze SHAP explanations to compute mean absolute SHAP values for each feature.

    Parameters:
    explanations (Iterable[shap.Explanation]): An iterable of SHAP Explanation objects.

    Returns:
    pd.DataFrame: A DataFrame with features as index and their mean absolute SHAP values.
    """
    # Extraer explicaciones para la clase positiva
    positive_class_explanations = extract_positive_class_explanations(explanations)

    # Construir DataFrame de valores absolutos de SHAP
    shap_abs_df = build_shap_absolute_df(positive_class_explanations)

    freq = shap_abs_df.notna().mean()

    # Promedio condicional (solo folds con valor, ignora NaN)
    mean_cond = shap_abs_df.mean(skipna=True)

    # Resultado combinado
    result_df = pd.DataFrame({
        "mean_abs_shap_cond": mean_cond,
        "freq_selection": freq
    }).sort_values("mean_abs_shap_cond", ascending=False)

    return result_df