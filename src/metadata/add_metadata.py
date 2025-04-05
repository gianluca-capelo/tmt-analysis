import pandas as pd

from src import config
from src.metadata.metadata_process import process_metadata


def add_metadata_to_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Add metadata to the metrics DataFrame.
    """
    df = pd.read_csv(config.METADATA_CSV)
    metadata_df = process_metadata(df)

    # Remove metadata columns from the metrics DataFrame
    metrics_copy = metrics.copy()
    metrics_copy = metrics_copy.drop(columns=["age", "gender"])

    merged_df = pd.merge(metrics_copy, metadata_df, on='subject_id', how='inner')

    return merged_df
