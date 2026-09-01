"""Retrieval metrics used by document-system experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..search import sparse_dot
from ..sparse_matrix import SparseMatrix

RETRIEVAL_TOP_K = 10


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    map_at_k: float


def evaluate_ranked_labels(
    *,
    query_labels: Sequence[int],
    ranked_labels: Sequence[Sequence[int]],
    corpus_labels: Sequence[int],
    k: int,
) -> RetrievalMetrics:
    """Evaluate label relevance for ranked results with P@k and MAP@k."""

    if k < 1:
        raise ValueError("k must be positive")
    if len(query_labels) != len(ranked_labels):
        raise ValueError("query labels and rankings must have the same length")

    relevant_counts = Counter(int(label) for label in corpus_labels)
    precision_at_k_values = []
    average_precision_at_k_values = []
    for query_label, ranking in zip(query_labels, ranked_labels, strict=True):
        if len(ranking) < k:
            raise ValueError("each ranking must contain at least k results")
        relevant_seen = 0
        precision_sum = 0.0
        for rank, result_label in enumerate(ranking[:k], start=1):
            if int(result_label) == int(query_label):
                relevant_seen += 1
                precision_sum += relevant_seen / rank
        precision_at_k_values.append(relevant_seen / k)
        relevant_result_count = min(relevant_counts[int(query_label)], k)
        average_precision_at_k_values.append(
            precision_sum / relevant_result_count if relevant_result_count else 0.0
        )
    return RetrievalMetrics(
        precision_at_k=(
            float(np.mean(precision_at_k_values)) if precision_at_k_values else 0.0
        ),
        map_at_k=(
            float(np.mean(average_precision_at_k_values))
            if average_precision_at_k_values
            else 0.0
        ),
    )


def evaluate_label_retrieval(
    query_matrix: SparseMatrix,
    query_labels: Sequence[int],
    corpus_matrix: SparseMatrix,
    corpus_labels: Sequence[int],
    *,
    k: int = RETRIEVAL_TOP_K,
) -> RetrievalMetrics:
    """Rank corpus documents for each query and score same-label relevance."""

    if query_matrix.shape[1] != corpus_matrix.shape[1]:
        raise ValueError("query and corpus matrices must share a vocabulary")
    if query_matrix.shape[0] != len(query_labels):
        raise ValueError("query labels must contain one label per query")
    if corpus_matrix.shape[0] != len(corpus_labels):
        raise ValueError("corpus labels must contain one label per document")
    if not 1 <= k <= corpus_matrix.shape[0]:
        raise ValueError("k must be between 1 and the corpus size")

    corpus_label_array = np.asarray(corpus_labels)
    ranked_result_labels: list[list[int]] = []
    for query_row_id in range(query_matrix.shape[0]):
        query_indices, query_values = query_matrix.get_sparse_row(query_row_id)
        scores = np.zeros(corpus_matrix.shape[0], dtype=np.float64)
        for corpus_row_id in range(corpus_matrix.shape[0]):
            document_indices, document_values = corpus_matrix.get_sparse_row(
                corpus_row_id
            )
            scores[corpus_row_id] = sparse_dot(
                query_indices,
                query_values,
                document_indices,
                document_values,
            )
        ranked_row_ids = np.argsort(-scores, kind="stable")[:k]
        ranked_result_labels.append(corpus_label_array[ranked_row_ids].tolist())
    return evaluate_ranked_labels(
        query_labels=query_labels,
        ranked_labels=ranked_result_labels,
        corpus_labels=corpus_labels,
        k=k,
    )
