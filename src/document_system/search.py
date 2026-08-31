"""Exact cosine search over normalized NumPy sparse vectors."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .constants import DEFAULT_TOP_K, SNIPPET_LIMIT
from .privacy import is_safe_text
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
    _, left_positions, right_positions = np.intersect1d(
        left_indices,
        right_indices,
        assume_unique=True,
        return_indices=True,
    )
    return float(np.dot(left_data[left_positions], right_data[right_positions]))


@dataclass(frozen=True)
class DocumentSearch:
    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    snippets: Sequence[str]
    labels: np.ndarray
    target_names: tuple[str, ...]
    document_ids: np.ndarray

    def __post_init__(self) -> None:
        document_count = self.matrix.shape[0]
        if (
            len(self.snippets) != document_count
            or len(self.labels) != document_count
            or len(self.document_ids) != document_count
        ):
            raise ValueError(
                "matrix, snippets, labels, and document_ids must have the same row count"
            )
        if self.matrix.shape[1] != len(self.vectorizer.vocabulary_):
            raise ValueError("matrix columns must match vectorizer vocabulary")
        if any(
            not isinstance(snippet, str)
            or len(snippet) > SNIPPET_LIMIT
            or not is_safe_text(snippet)
            for snippet in self.snippets
        ):
            raise ValueError(
                "snippets must be safe, nonblank, and within the length limit"
            )

    def search(self, query: str, topk: int = DEFAULT_TOP_K) -> list[SearchResult]:
        document_count = self.matrix.shape[0]
        if not 1 <= topk <= document_count:
            raise ValueError(f"topk must be between 1 and {document_count}")
        if not query.strip():
            raise ValueError("query must not be blank")
        query_matrix = self.vectorizer.transform([query])
        query_indices, query_values = query_matrix.get_sparse_row(0)
        if query_indices.size == 0:
            raise ValueError("query has no terms in the fitted vocabulary")

        scores = np.zeros(document_count, dtype=np.float64)
        for matrix_row_id in range(document_count):
            document_indices, document_values = self.matrix.get_sparse_row(
                matrix_row_id
            )
            scores[matrix_row_id] = sparse_dot(
                query_indices,
                query_values,
                document_indices,
                document_values,
            )
        positive_score_row_ids = np.flatnonzero(scores > 0)
        ranked_row_ids = positive_score_row_ids[
            np.lexsort(
                (self.document_ids[positive_score_row_ids], -scores[positive_score_row_ids])
            )
        ][:topk]
        return [
            SearchResult(
                score=float(scores[matrix_row_id]),
                doc_id=int(self.document_ids[matrix_row_id]),
                label=self.target_names[int(self.labels[matrix_row_id])],
                text_snippet=_snippet(self.snippets[matrix_row_id]),
            )
            for matrix_row_id in ranked_row_ids
        ]


def _snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]
