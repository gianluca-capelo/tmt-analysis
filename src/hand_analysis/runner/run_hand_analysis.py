import logging

from neurotask.tmt.tmt_analyzer import TMTAnalyzer

from src import config
from src.hand_analysis.mapper.psychopy_mapper import PsychopyTMTMapper


def log_and_run_tmt_analysis(dataset_path, output_path, correct_targets_minimum, consecutive_points, cut_criteria):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    hand_analysis = TMTAnalyzer(
        mapper=PsychopyTMTMapper(),
        dataset_path=dataset_path,
        output_path=output_path
    )

    hand_analysis.run(correct_targets_minimum, consecutive_points, cut_criteria=cut_criteria)
    return hand_analysis


def run_analysis_with_configuration_parameters():
    # Cargamos el análisis de TMT
    threshold = config.CORRECT_THRESHOLD
    cut_criteria = config.CUT_CRITERIA
    points = config.CONSECUTIVE_POINTS

    if threshold is None and cut_criteria == "MINIMUM_TARGETS":
        raise ValueError("`correct_targets_minimum` must be set when `cut_criteria` is 'MINIMUM_TARGETS'.")

    analysis = log_and_run_tmt_analysis(
        dataset_path=config.PYXATIONS_PATIENTS_DATA_DIR,
        output_path=config.HAND_ANALYSIS_FOLDER,
        correct_targets_minimum=threshold,
        consecutive_points=points,
        cut_criteria=cut_criteria
    )

    # Obtenemos el DataFrame de métricas
    df_metrics = analysis.get_metrics_dataframe()
    print("Metrics DataFrame:")
    print(df_metrics.head())

    experiment = analysis.experiment
    print(f"Len of subjects: {len(experiment.subjects)}")

    return analysis


if __name__ == "__main__":
    run_analysis_with_configuration_parameters()
