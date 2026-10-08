"""Compact term postings and sparse candidate-only score accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sparse_matrix import SparseMatrix


@dataclass(frozen=True)
class InvertedIndex:
    term_indptr: np.ndarray
    document_rows: np.ndarray
    weights: np.ndarray
    document_count: int

    def __post_init__(self) -> None:
        if (
            self.term_indptr.ndim != 1
            or self.document_rows.ndim != 1
            or self.weights.ndim != 1
            or not np.issubdtype(self.term_indptr.dtype, np.integer)
            or not np.issubdtype(self.document_rows.dtype, np.integer)
            or self.term_indptr.size == 0
            or self.term_indptr[0] != 0
            or self.term_indptr[-1] != self.weights.size
            or self.document_rows.size != self.weights.size
            or np.any(np.diff(self.term_indptr) < 0)
            or np.any(self.document_rows < 0)
            or np.any(self.document_rows >= self.document_count)
            or not np.all(np.isfinite(self.weights))
        ):
            raise ValueError("invalid posting arrays")

    @classmethod
    def from_matrix(cls, matrix: SparseMatrix) -> InvertedIndex:
        order = np.argsort(matrix.indices, kind="stable")
        rows = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int32), np.diff(matrix.indptr)
        )
        counts = np.bincount(matrix.indices, minlength=matrix.shape[1])
        pointers = np.concatenate(([0], np.cumsum(counts))).astype(matrix.indptr.dtype)
        return cls(pointers, rows[order], matrix.data[order], matrix.shape[0])

    def posting(self, term_id: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= term_id < self.term_indptr.size - 1:
            raise ValueError("term_id is outside the vocabulary")
        start, end = map(int, self.term_indptr[term_id : term_id + 2])
        return self.document_rows[start:end], self.weights[start:end]

    def score(self, indices: np.ndarray, values: np.ndarray) -> dict[int, float]:
        """Accumulate only touched rows, visiting terms in vocabulary order."""
        scores: dict[int, float] = {}
        for term_id, query_weight in zip(indices, values, strict=True):
            rows, weights = self.posting(int(term_id))
            contributions = weights * query_weight
            for row, contribution in zip(rows, contributions, strict=True):
                row_id = int(row)
                scores[row_id] = scores.get(row_id, 0.0) + float(contribution)
        return scores

    @property
    def memory_bytes(self) -> int:
        """Resident NumPy payload bytes, excluding Python object overhead."""
        return self.term_indptr.nbytes + self.document_rows.nbytes + self.weights.nbytes
