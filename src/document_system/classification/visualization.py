"""Classification report visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .evaluation import ClassificationReport

matplotlib.use("Agg")
from matplotlib import pyplot as plt

CONFUSION_MATRIX_MIN_FIGURE_SIZE = 7.0
CONFUSION_MATRIX_FIGURE_SIZE_FACTOR = 0.48
CONFUSION_MATRIX_COLORBAR_FRACTION = 0.046
CONFUSION_MATRIX_COLORBAR_PADDING = 0.04
CONFUSION_MATRIX_TICK_FONT_SIZE = 7
CONFUSION_MATRIX_TICK_ROTATION = 55
CONFUSION_MATRIX_DPI = 160


def save_confusion_matrix(
    report: ClassificationReport,
    target_names: tuple[str, ...],
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = max(
        CONFUSION_MATRIX_MIN_FIGURE_SIZE,
        len(target_names) * CONFUSION_MATRIX_FIGURE_SIZE_FACTOR,
    )
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(report.confusion_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(
        image,
        ax=axis,
        fraction=CONFUSION_MATRIX_COLORBAR_FRACTION,
        pad=CONFUSION_MATRIX_COLORBAR_PADDING,
    )
    axis.set(
        title="20 Newsgroups Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=np.arange(len(target_names)),
        yticks=np.arange(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=CONFUSION_MATRIX_TICK_ROTATION,
        ha="right",
        fontsize=CONFUSION_MATRIX_TICK_FONT_SIZE,
    )
    plt.setp(axis.get_yticklabels(), fontsize=CONFUSION_MATRIX_TICK_FONT_SIZE)
    figure.tight_layout()
    figure.savefig(output, dpi=CONFUSION_MATRIX_DPI)
    plt.close(figure)
