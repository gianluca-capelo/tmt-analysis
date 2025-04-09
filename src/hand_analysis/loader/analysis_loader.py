import logging

import pandas as pd

from src import config
from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_default_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis() -> pd.DataFrame:
    analysis = run_analysis_with_default_parameters()

    metrics_df = analysis.get_metrics_dataframe()

    metrics_df_with_metadata = add_metadata_to_metrics(metrics_df)

    valid_metrics_df = filter_invalid_subjects(metrics_df_with_metadata)

    # save the valid metrics DataFrame to a CSV file
    valid_metrics_df.to_csv(config.ANALYSIS_PATH, index=False)

    return valid_metrics_df


def main():
    # Load the analysis and get the DataFrame
    metrics_df = load_analysis()
    # Print the DataFrame
    print("Metrics DataFrame:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
