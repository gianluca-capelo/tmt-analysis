from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
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


def analyze_shap_results(explanations: Iterable[shap.Explanation], fillna_zero=False) -> pd.DataFrame:
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

    result_df = build_mean_shap_df(shap_abs_df, fillna_zero=fillna_zero)

    return result_df


def build_mean_shap_df(shap_abs_df, fillna_zero=False):
    """
    Build a summary DataFrame with mean absolute SHAP values and selection frequency.

    Args:
        shap_abs_df: DataFrame (folds x features) with absolute SHAP values.
                     NaN indicates the feature was not selected in that fold.
        fillna_zero: if True, compute the mean replacing NaN with 0
                     (global mean including non-selected folds).
                     if False, compute the mean only across selected folds.

    Returns:
        result_df: DataFrame with columns:
                   - 'mean_abs_shap': mean |SHAP| per feature
                   - 'freq_selection': proportion of folds where the feature was selected
    """
    freq = shap_abs_df.notna().mean()

    if fillna_zero:
        mean_vals = shap_abs_df.fillna(0).mean()
    else:
        mean_vals = shap_abs_df.mean(skipna=True)

    result_df = pd.DataFrame({
        "mean_abs_shap": mean_vals,
        "freq_selection": freq
    }).sort_values("mean_abs_shap", ascending=False)

    return result_df


def plot_shap_summary(df, top_n=20, plot_freq=False):
    """
    Plotea el DataFrame resultante de shap_mean_conditional en formato horizontal.

    Args:
        df: DataFrame con columnas ['mean_abs_shap', 'freq_selection'].
        top_n: cuántos features mostrar (ordenados por mean_abs_shap).
        plot_freq: si True, agrega la frecuencia de selección en un eje X secundario.
    """
    # Ordenar y limitar al top_n
    df_plot = df.sort_values("mean_abs_shap", ascending=True).tail(top_n)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eje 1: barras horizontales de |SHAP| medio condicional
    ax1.barh(df_plot.index, df_plot["mean_abs_shap"], color="steelblue", alpha=0.7)
    ax1.set_xlabel("Mean |SHAP| (condicional)", color="steelblue")
    ax1.tick_params(axis="x", labelcolor="steelblue")

    # Eje 2 opcional: frecuencia de selección
    if plot_freq:
        ax2 = ax1.twiny()
        ax2.plot(df_plot["freq_selection"], df_plot.index, color="darkorange",
                 marker="o", linestyle="-", linewidth=2)
        ax2.set_xlabel("Frecuencia de selección", color="darkorange")
        ax2.tick_params(axis="x", labelcolor="darkorange")
        ax2.set_xlim(0, 1)  # frecuencia en [0,1]

    plt.title("Importancia media de SHAP (clase positiva)" +
              (" y frecuencia de selección" if plot_freq else ""))
    plt.tight_layout()
    plt.show()
