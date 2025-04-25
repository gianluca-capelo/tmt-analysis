import pandas as pd

from src import config
from src.config import RANDOM_STATE
from src.hand_analysis.dataset_split.eval_train_split import split_subjectwise_evaluation_set_stratified
from src.hand_analysis.filter.filter_invalid_subjects import filter_invalid_subjects
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_default_parameters
from src.metadata.add_metadata import add_metadata_to_metrics


def load_analysis(random_state, test_size, split=False) -> pd.DataFrame:
    analysis = run_analysis_with_default_parameters()

    metrics_df = analysis.get_metrics_dataframe()

    metrics_df_with_metadata = add_metadata_to_metrics(metrics_df)

    valid_metrics_df = filter_invalid_subjects(metrics_df_with_metadata)

    valid_metrics_df.to_csv(config.ANALYSIS_PATH, index=False)

    if split:
        split_subjectwise_evaluation_set_stratified(valid_metrics_df, test_size=test_size, random_state=random_state)

    return valid_metrics_df


def main():
    test_size = 0.2
    metrics_df = load_analysis(RANDOM_STATE, test_size)
    print("Metrics DataFrame:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
