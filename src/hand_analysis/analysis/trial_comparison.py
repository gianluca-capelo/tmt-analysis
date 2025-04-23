import pandas as pd
from matplotlib import pyplot as plt

from src.config import TRAIN_SET_PATH


def compare_trial_by_group(df, trial_discriminant: str):
    if trial_discriminant not in df.columns:
        raise ValueError(f"'{trial_discriminant}' is not a valid column in the DataFrame.")

    if trial_discriminant != 'trial_id' and trial_discriminant != 'trial_order_of_appearance':
        raise ValueError("trial_discriminant must be either 'trial_id' or 'trial_order_of_appearance'.")

    df['percent_correct'] = df['correct_targets_touches'] / 15 * 100

    df[trial_discriminant] = df[trial_discriminant].astype(float)
    mean_df = df.groupby([trial_discriminant, 'group'])['percent_correct'].mean().unstack('group')

    plt.figure()
    for g in mean_df.columns:
        plt.plot(mean_df.index, mean_df[g], marker='o', label=g)
    plt.xlabel(trial_discriminant)
    plt.ylabel('Mean Percentage Correct')
    plt.title('Mean Percent Correct Across Trials by Group')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv(TRAIN_SET_PATH)

    compare_trial_by_group(df, 'trial_id')
    compare_trial_by_group(df, 'trial_order_of_appearance')
