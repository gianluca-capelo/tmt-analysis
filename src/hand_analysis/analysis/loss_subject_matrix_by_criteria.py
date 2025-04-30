import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR
from src.hand_analysis.loader.load_last_split import load_analysis_by_date


def plot_loss_matrix(data, ax=None, title_suffix=''):
    subjects = data['subject_id'].unique()
    total_subjects = len(subjects)
    number_of_trials = len(data['trial_order_of_appearance'].unique())
    number_of_trials_range = range(1, number_of_trials + 1)
    target_touches_range = range(0, 16)

    # build subject_counts
    subject_counts = pd.DataFrame(index=subjects, columns=target_touches_range, dtype=int)
    for M in target_touches_range:
        counts = (
            data
            .groupby('subject_id')['correct_targets_touches']
            .apply(lambda touches: (touches >= M).sum())
        )
        subject_counts[M] = counts

    # build lost_matrix
    lost_matrix = pd.DataFrame(index=number_of_trials_range, columns=target_touches_range, dtype=int)
    for N in number_of_trials_range:
        for M in target_touches_range:
            retained = (subject_counts[M] >= N).sum()
            lost_matrix.at[N, M] = total_subjects - retained

    # if no Axes passed in, make a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        lost_matrix.astype(int),
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar_kws={'label': 'Number of Subjects Lost'},
        ax=ax
    )
    ax.set_title(f'Subjects Lost by Retention Criteria {title_suffix}')
    ax.set_xlabel('M: Correct Touches Threshold')
    ax.set_ylabel('N: Minimum Qualifying Trials')


if __name__ == "__main__":
    train_set, _ = load_analysis_by_date("2025-04-25_12-55-48")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    plot_loss_matrix(
        train_set[train_set['trial_type'] == 'PART_A'],
        ax=axes[0],
        title_suffix='(PART_A)'
    )

    plot_loss_matrix(
        train_set[train_set['trial_type'] == 'PART_B'],
        ax=axes[1],
        title_suffix='(PART_B)'
    )
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/loss_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    plot_loss_matrix(
        train_set,
        ax=None,
        title_suffix=None
    )

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/loss_matrix_all.png"
                , dpi=300, bbox_inches='tight')
    plt.show()
