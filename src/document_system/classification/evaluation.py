"""Classification metrics and safe error records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..constants import DEFAULT_BATCH_SIZE
from ..privacy import make_safe_snippet
from ..sparse_matrix import SparseMatrix
from .model import LINEAR_SVM_LOSS, predict_sparse

LINEAR_SVM_DESCRIPTION = "linear SVM trained with SGDClassifier hinge loss"


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
