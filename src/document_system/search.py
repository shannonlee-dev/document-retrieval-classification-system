"""Exact cosine search over normalized NumPy sparse vectors."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer


@dataclass(frozen=True)
class SearchResult:
    score: float
    doc_id: int
    label: str
    text_snippet: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def sparse_dot(
    left_indices: np.ndarray,
    left_data: np.ndarray,
    right_indices: np.ndarray,
    right_data: np.ndarray,
) -> float:
    """Return an exact dot product using sorted nonzero-index intersections."""

    if left_indices.size != left_data.size or right_indices.size != right_data.size:
        raise ValueError("indices and data lengths must match")
    left = right = 0
    result = 0.0
    while left < left_indices.size and right < right_indices.size:
        left_column = int(left_indices[left])
        right_column = int(right_indices[right])
        if left_column == right_column:
            result += float(left_data[left] * right_data[right])
            left += 1
            right += 1
        elif left_column < right_column:
            left += 1
        else:
            right += 1
    return result


@dataclass(frozen=True)
class DocumentSearch:
    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    texts: Sequence[str]
    labels: np.ndarray
    target_names: tuple[str, ...]

    def __post_init__(self) -> None:
        document_count = self.matrix.shape[0]
        if len(self.texts) != document_count or len(self.labels) != document_count:
            raise ValueError("matrix, texts, and labels must have the same row count")
        if self.matrix.shape[1] != len(self.vectorizer.vocabulary_):
            raise ValueError("matrix columns must match vectorizer vocabulary")

    def search(self, query: str, topk: int = 5) -> list[SearchResult]:
        document_count = self.matrix.shape[0]
        if not 1 <= topk <= document_count:
            raise ValueError(f"topk must be between 1 and {document_count}")
        query_matrix = self.vectorizer.transform([query])
        query_indices, query_values = query_matrix.row(0)
        if query_indices.size == 0:
            raise ValueError("query has no terms in the fitted vocabulary")

        scores = np.zeros(document_count, dtype=np.float64)
        for doc_id in range(document_count):
            document_indices, document_values = self.matrix.row(doc_id)
            scores[doc_id] = sparse_dot(
                query_indices,
                query_values,
                document_indices,
                document_values,
            )
        doc_ids = np.arange(document_count)
        ranked_ids = np.lexsort((doc_ids, -scores))[:topk]
        return [
            SearchResult(
                score=float(scores[doc_id]),
                doc_id=int(doc_id),
                label=self.target_names[int(self.labels[doc_id])],
                text_snippet=_snippet(self.texts[doc_id]),
            )
            for doc_id in ranked_ids
        ]


def _snippet(text: str, limit: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]
