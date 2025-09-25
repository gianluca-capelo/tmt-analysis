from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import shap

from src.model.shap.run_shap import run_shap


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


def analyze_shap_results(explanations: Iterable[shap.Explanation], task: str, fillna_zero=False) -> pd.DataFrame:
    """
    Analyze SHAP explanations to compute mean absolute SHAP values for each feature.

    Parameters:
    explanations (Iterable[shap.Explanation]): An iterable of SHAP Explanation objects.

    Returns:
    pd.DataFrame: A DataFrame with features as index and their mean absolute SHAP values.
    """
    # Extraer explicaciones para la clase positiva
    explanations = extract_positive_class_explanations(explanations) if task == "classification" else explanations

    # Construir DataFrame de valores absolutos de SHAP
    shap_abs_df = build_shap_absolute_df(explanations)

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


import matplotlib.pyplot as plt
from src.config import FIGURES_DIR


def clean_target_col(target_col):
    if target_col.lower() == "mmse":
        return "MMSE"

    return target_col.upper()


def plot_shap_summary(df, is_classification, target_col, top_n=20, plot_freq=False, annotate_values=True,
                      save_filename=None):
    """
    Plot SHAP summary in horizontal format.

    Args:
        df: DataFrame with columns ['mean_abs_shap', 'freq_selection'].
        top_n: number of features to display (ordered by mean_abs_shap).
        plot_freq: if True, add selection frequency on a secondary X axis.
        annotate_values: if True, annotate each bar with its mean SHAP value.
        save_filename: if not None, save the figure in FIGURES directory with this filename.
    """
    # Sort by mean absolute SHAP value and keep the top_n features
    df_plot = df.sort_values("mean_abs_shap", ascending=True).tail(top_n)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: horizontal bars of mean absolute SHAP values
    bar_color = "steelblue"


    # Warning: Complete feature labels for classification, need to be completed for regression
    feature_labels = {
        "is_valid_sum_A": "Valid sum (Part A)",
        "non_cut_correct_targets_touches_PART_B": "Complete correct touches (Part B)",
        "is_valid_sum_B": "Valid sum (Part B)",
        "max_duration_B_A_ratio": "Hesitation Max duration (B/A ratio)",
        "non_cut_correct_targets_touches_B_A_ratio": "Complete correct touches (B/A ratio)",
        "travel_time_PART_A": "Travel time (Part A)",
        "travel_time_PART_B": "Travel time (Part B)",
        "total_hesitations_PART_A": "Hesitations (Part A)",
        "travel_distance_B_A_ratio": "Travel distance (B/A ratio)",
        "state_transitions_PART_A": "State transitions (Part A)",
        "average_duration_PART_A": "Average duration (Part A)",
        "non_cut_rt_PART_B": "Complete reaction time (Part B)",
        "age": "Age",
        "search_time_PART_A": "Search time (Part A)",
        "max_duration_PART_B": "Hesitation max duration (Part B)",
        "state_transitions_B_A_ratio": "State transitions (B/A ratio)",
        "average_duration_PART_B": "Average duration (Part B)",
        "non_cut_correct_targets_touches_PART_A": "Complete correct touches (Part A)",
        "travel_avg_speed_B_A_ratio": "Average speed (B/A ratio)",
        "hesitation_time_PART_A": "Hesitation time (Part A)",
        "travel_distance_PART_B": "Travel distance (Part B)",
    }
    df_plot.index = df_plot.index.map(feature_labels)
    bars = ax1.barh(df_plot.index, df_plot["mean_abs_shap"], alpha=0.7)
    ax1.set_xlabel("Mean |SHAP| (across selected folds)")
    ax1.tick_params(axis="x")

    # Add text annotations at the end of each bar
    if annotate_values:
        max_val = df_plot["mean_abs_shap"].max()
        for bar in bars:
            width = bar.get_width()
            ax1.text(width,
                     bar.get_y() + bar.get_height() / 2,
                     f"+{width:.3f}",
                     va="center", ha="left", fontsize=9, color=bar_color)
        # Extend x-axis limit to avoid cutting off text
        ax1.set_xlim(0, max_val * 1.1)

    # Optional Axis 2: selection frequency
    if plot_freq:
        ax2 = ax1.twiny()
        ax2.plot(df_plot["freq_selection"], df_plot.index, color="darkorange",
                 marker="o", linestyle="-", linewidth=2)
        ax2.set_xlabel("Selection frequency", color="darkorange")
        ax2.tick_params(axis="x", labelcolor="darkorange")
        ax2.set_xlim(0, 1)

    # Titles and layout
    if is_classification:
        title = "Mean SHAP importance"
    else:
        title = f"Mean SHAP importance for {clean_target_col(target_col)}"

    plt.title(title +
              (" and selection frequency" if plot_freq else ""))
    plt.tight_layout()

    # Save or show
    if save_filename:
        save_path = os.path.join(FIGURES_DIR, save_filename)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def run_analysis(dataset_name, is_classification, model, timestamp, save_filename, target_col):
    shap_explanations = run_shap(
        dataset_name=dataset_name,
        target_col=target_col,
        task="classification" if is_classification else "regression",
        model_name_to_explain=model,
        timestamp=timestamp
    )
    shap_df = analyze_shap_results(shap_explanations, fillna_zero=False,
                                   task="classification" if is_classification else "regression")
    print(shap_df.head(10))
    plot_shap_summary(shap_df, top_n=20,
                      save_filename=save_filename, is_classification=is_classification, target_col=target_col)


def main(is_classification, timestamp):
    if is_classification:
        target_col = "group"  # Binary classification
        model = "SVC"
        timestamp = timestamp
        save_filename = "shap_summary_classification.png"
        dataset_name = "demographic+digital"
    else:
        target_col = "mmse"
        model = "RandomForestRegressor"
        timestamp = timestamp
        save_filename = "shap_summary_regression.png"
        dataset_name = "demographic+digital"
    run_analysis(dataset_name, is_classification, model, timestamp, save_filename, target_col)


if __name__ == "__main__":
    is_classification = True
    timestamp = "2025-09-14_0038" if is_classification else "2025-09-14_1008"
    main(is_classification, timestamp)
