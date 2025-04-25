import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.hand_analysis.loader.load_last_split import load_analysis_by_date


def plot_loss_matrix(data):
    subjects = data['subject_id'].unique()
    total_subjects = len(subjects)

    # Maximum number of trials per subject (should be 15)
    max_trials = data['trial_id'].nunique()

    # Prepare a grid for N (required number of qualifying trials) and M (threshold correct touches)
    N_values = range(1, max_trials + 1)  # 1 through 15
    M_values = range(0, 16)  # 0 through 15 possible touches

    # Initialize a DataFrame to store the number of subjects lost for each (N, M)
    lost_matrix = pd.DataFrame(index=N_values, columns=M_values)

    # Precompute per-subject counts for each M threshold
    # subject_counts[M] = number of trials where correct_targets_touches >= M
    subject_counts = pd.DataFrame(index=subjects, columns=M_values)

    for M in M_values:
        counts = data.groupby('subject_id')['correct_targets_touches'] \
            .apply(lambda touches: (touches >= M).sum())
        subject_counts[M] = counts

    # Compute lost subjects for each (N, M)
    for N in N_values:
        for M in M_values:
            retained = (subject_counts[M] >= N).sum()
            lost = total_subjects - retained
            lost_matrix.at[N, M] = lost

    # Plot heatmap of lost subjects
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        lost_matrix.astype(int),
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar_kws={'label': 'Number of Subjects Lost'}
    )
    plt.title('Subjects Lost by Retention Criteria\n(N trials with ≥M correct touches)')
    plt.xlabel('M: Correct Touches Threshold')
    plt.ylabel('N: Minimum Qualifying Trials')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_set, _ = load_analysis_by_date("2025-04-25_12-55-48")
    plot_loss_matrix(train_set)
