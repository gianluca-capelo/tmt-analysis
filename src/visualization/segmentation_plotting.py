from typing import Tuple, List

import matplotlib.pyplot as plt
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo
from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation

from src.visualization.trial_plotting_labels import  plot_with_labels_scatter


def plot_segmentation(trial: TMTTrial, target_radius: float, speed_threshold: float, cmap_name='tab10'):
    segmentation: List[Tuple[str, CursorInfo]] = classify_cursor_positions_with_hesitation(trial,
                                                                                           target_radius,
                                                                                           speed_threshold)
    segmentation_labels = [label for (label, color) in segmentation]

    labels_title = 'Segmentation Labels'

    fig = plot_with_labels_scatter(trial, target_radius, segmentation_labels, labels_title=labels_title, cmap_name=cmap_name)

    return fig
