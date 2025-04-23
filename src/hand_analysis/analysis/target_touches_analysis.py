import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from src.config import TRAIN_SET_PATH


def plot_feature_aggregation_by_subject_id(data: pd.DataFrame, feature: str, agg_function: str = 'mean'):
    subject_stats = (
        data.groupby('subject_id')
        .agg(feature_agg=(feature, agg_function),
             group=('group', 'first'))
    )

    subject_stats = subject_stats.sort_values('feature_agg')

    group_colors = {'HC': 'C0', 'MCI': 'C1'}
    color_list = subject_stats['group'].map(group_colors)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(subject_stats.index.astype(str), subject_stats['feature_agg'], color=color_list)
    ax.set_xlabel('Subject ID')
    ax.set_ylabel(f'{feature} ({agg_function})')
    ax.set_title(f'{feature} Aggregation ({agg_function}) by Subject ID')
    plt.xticks(rotation=90)

    legend_handles = [Patch(color=color, label=grp) for grp, color in group_colors.items()]
    ax.legend(handles=legend_handles, title='Group')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_SET_PATH)

    plot_feature_aggregation_by_subject_id(df, 'correct_targets_touches', 'mean')
    plot_feature_aggregation_by_subject_id(df, 'correct_targets_touches', 'sum')

    plot_feature_aggregation_by_subject_id(df, 'wrong_targets_touches', 'mean')
    plot_feature_aggregation_by_subject_id(df, 'wrong_targets_touches', 'sum')
