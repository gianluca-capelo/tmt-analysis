import pandas as pd

from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_default_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis() -> pd.DataFrame:
    analysis = run_analysis_with_default_parameters()

    metrics_df = analysis.get_metrics_dataframe()

    metrics_df_with_metadata = add_metadata_to_metrics(metrics_df)

    valid_metrics_df = filter_invalid_subjects(metrics_df_with_metadata)

    return valid_metrics_df
