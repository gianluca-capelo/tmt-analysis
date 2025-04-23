import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks

from src.config import TRAIN_SET_PATH



def compare_trial_by_group(df, trial_discriminant: str):
    if trial_discriminant not in df.columns:
        raise ValueError(f"'{trial_discriminant}' is not a valid column in the DataFrame.")
    if trial_discriminant not in ('trial_id', 'trial_order_of_appearance'):
        raise ValueError("trial_discriminant must be either 'trial_id' or 'trial_order_of_appearance'.")

    # compute percent correct
    df = df.copy()
    df['percent_correct'] = df['correct_targets_touches'] / 15 * 100

    # ensure numeric index
    df[trial_discriminant] = df[trial_discriminant].astype(float)
    mean_df = df.groupby([trial_discriminant, 'group'])['percent_correct'] \
                .mean() \
                .unstack('group')

    # plot
    plt.figure(figsize=(8,5))
    for g in mean_df.columns:
        plt.plot(mean_df.index, mean_df[g], marker='o', label=g)

    # force x-axis from 0 to 20 with integer ticks
    x_ticks_array = sorted(df[trial_discriminant].unique())
    plt.xticks(x_ticks_array)

    plt.xlabel(trial_discriminant)
    plt.ylabel('Mean Percentage Correct')
    plt.title('Mean Percent Correct Across Trials by Group')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_SET_PATH)

    compare_trial_by_group(df, 'trial_id')
    compare_trial_by_group(df, 'trial_order_of_appearance')
