from neurotask.tmt.visualization.visualization import plot_with_color

from src import config
from src.hand_analysis.runner.run_hand_analysis import run_analysis_with_configuration_parameters


def main():

    analysis = run_analysis_with_configuration_parameters(config.ANALYSIS_PATH)
    experiment = analysis.experiment

