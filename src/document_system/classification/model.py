"""Batched linear-SVM training and sparse prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from ..constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_RANDOM_STATE
from ..sparse_matrix import SparseMatrix

LINEAR_SVM_LOSS = "hinge"
SGD_MAX_ITER_PER_BATCH = 1


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
