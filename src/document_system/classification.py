"""Batched linear-SVM training and classification evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_RANDOM_STATE
from .privacy import make_safe_snippet
from .sparse_matrix import SparseMatrix

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LINEAR_SVM_LOSS = "hinge"
LINEAR_SVM_DESCRIPTION = "linear SVM trained with SGDClassifier hinge loss"
SGD_MAX_ITER_PER_BATCH = 1
CONFUSION_MATRIX_MIN_FIGURE_SIZE = 7.0
CONFUSION_MATRIX_FIGURE_SIZE_FACTOR = 0.48
CONFUSION_MATRIX_COLORBAR_FRACTION = 0.046
CONFUSION_MATRIX_COLORBAR_PADDING = 0.04
CONFUSION_MATRIX_TICK_FONT_SIZE = 7
CONFUSION_MATRIX_TICK_ROTATION = 55
CONFUSION_MATRIX_DPI = 160


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    misclassifications: list[dict[str, int | str]]
    model_settings: dict[str, int | float | str]

    def metrics_dict(self) -> dict[str, object]:
        return {
            "model": LINEAR_SVM_DESCRIPTION,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "model_settings": self.model_settings,
            "confusion_matrix_shape": list(self.confusion_matrix.shape),
            "misclassification_count": len(self.misclassifications),
        }


def train_linear_svm(
    matrix: SparseMatrix,
    labels: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SGDClassifier:
    """Train a hinge-loss linear SVM without densifying the full matrix."""

    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != matrix.shape[0]:
        raise ValueError("labels must contain one value per matrix row")
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("linear SVM training requires at least two classes")

    model = SGDClassifier(
        loss=LINEAR_SVM_LOSS,
        random_state=random_state,
        max_iter=SGD_MAX_ITER_PER_BATCH,
        tol=None,
        shuffle=False,
    )
    random_generator = np.random.default_rng(random_state)
    first_batch = True
    row_ids = np.arange(matrix.shape[0])
    for _ in range(epochs):
        shuffled_row_ids = random_generator.permutation(row_ids)
        for batch_start in range(0, shuffled_row_ids.size, batch_size):
            batch_row_ids = shuffled_row_ids[batch_start : batch_start + batch_size]
            batch_features = matrix.to_dense_rows(batch_row_ids)
            batch_labels = labels[batch_row_ids]
            if first_batch:
                model.partial_fit(batch_features, batch_labels, classes=classes)
                first_batch = False
            else:
                model.partial_fit(batch_features, batch_labels)
    return model


def predict_sparse(
    model: SGDClassifier,
    matrix: SparseMatrix,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions: list[np.ndarray] = []
    for batch_start in range(0, matrix.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, matrix.shape[0])
        batch_row_ids = range(batch_start, batch_end)
        predictions.append(model.predict(matrix.to_dense_rows(batch_row_ids)))
    if not predictions:
        return np.array([], dtype=np.int32)
    return np.concatenate(predictions)


def evaluate_classifier(
    model: SGDClassifier,
    matrix: SparseMatrix,
    labels: np.ndarray,
    snippets: Sequence[str],
    target_names: tuple[str, ...],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    document_ids: Sequence[int] | None = None,
) -> ClassificationReport:
    labels = np.asarray(labels)
    if labels.size != matrix.shape[0] or len(snippets) != matrix.shape[0]:
        raise ValueError("matrix, labels, and snippets must have matching row counts")
    if document_ids is None:
        document_ids = range(matrix.shape[0])
    if len(document_ids) != matrix.shape[0]:
        raise ValueError("document_ids must contain one ID per matrix row")
    predictions = predict_sparse(model, matrix, batch_size=batch_size)
    class_ids = np.arange(len(target_names))
    misclassifications: list[dict[str, int | str]] = []
    for row_id, (actual, predicted) in enumerate(zip(labels, predictions, strict=True)):
        if actual == predicted:
            continue
        misclassifications.append(
            {
                "doc_id": int(document_ids[row_id]),
                "actual": target_names[int(actual)],
                "predicted": target_names[int(predicted)],
                "text_snippet": make_safe_snippet(snippets[row_id]),
            }
        )
    return ClassificationReport(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=class_ids),
        predictions=predictions,
        misclassifications=misclassifications,
        model_settings={
            "loss": LINEAR_SVM_LOSS,
            "batch_size": batch_size,
            "random_state": int(model.random_state or 0),
            "input_representation": "NumPy dense batches",
        },
    )


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
