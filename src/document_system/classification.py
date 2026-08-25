"""Batched linear-SVM training and classification evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .privacy import make_safe_snippet
from .sparse_matrix import SparseMatrix

matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
            "model": "linear SVM trained with SGDClassifier hinge loss",
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
    batch_size: int = 128,
    epochs: int = 6,
    random_state: int = 42,
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
        loss="hinge",
        random_state=random_state,
        max_iter=1,
        tol=None,
        shuffle=False,
    )
    generator = np.random.default_rng(random_state)
    first_batch = True
    row_ids = np.arange(matrix.shape[0])
    for _ in range(epochs):
        shuffled = generator.permutation(row_ids)
        for start in range(0, shuffled.size, batch_size):
            batch_ids = shuffled[start : start + batch_size]
            features = matrix.to_dense_rows(batch_ids)
            batch_labels = labels[batch_ids]
            if first_batch:
                model.partial_fit(features, batch_labels, classes=classes)
                first_batch = False
            else:
                model.partial_fit(features, batch_labels)
    return model


def predict_sparse(
    model: SGDClassifier,
    matrix: SparseMatrix,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        predictions.append(model.predict(matrix.to_dense_rows(range(start, end))))
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
    batch_size: int = 128,
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
    errors: list[dict[str, int | str]] = []
    for row_id, (actual, predicted) in enumerate(zip(labels, predictions, strict=True)):
        if actual == predicted:
            continue
        errors.append(
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
        misclassifications=errors,
        model_settings={
            "loss": "hinge",
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
    size = max(7.0, len(target_names) * 0.48)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(report.confusion_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="20 Newsgroups Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=np.arange(len(target_names)),
        yticks=np.arange(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
