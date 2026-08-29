"""A minimal CSR-like sparse matrix backed only by NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseMatrix:
    """Store nonzero row values with sorted column indices."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        indices = np.asarray(self.indices)
        indptr = np.asarray(self.indptr)
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError("data, indices, and indptr must be one-dimensional")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(
            indptr.dtype, np.integer
        ):
            raise ValueError("indices and indptr must use integer dtypes")
        if len(self.shape) != 2 or self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError("shape must contain two non-negative dimensions")
        rows, columns = self.shape
        if indptr.size != rows + 1 or indptr.size == 0:
            raise ValueError("indptr length must equal row count plus one")
        if int(indptr[0]) != 0 or int(indptr[-1]) != data.size:
            raise ValueError("indptr must start at zero and end at nnz")
        if data.size != indices.size:
            raise ValueError("data and indices must have the same length")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be monotonic")
        if indices.size and (np.any(indices < 0) or np.any(indices >= columns)):
            raise ValueError("column index is outside matrix shape")
        for row_id in range(rows):
            start, end = int(indptr[row_id]), int(indptr[row_id + 1])
            if end - start > 1 and np.any(np.diff(indices[start:end]) <= 0):
                raise ValueError("row column indices must be strictly increasing")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "indptr", indptr)

    @property
    def _nnz(self) -> int:
        return int(self.data.size)

    def row(self, row_id: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= row_id < self.shape[0]:
            raise IndexError(f"row index {row_id} is outside matrix")
        start, end = int(self.indptr[row_id]), int(self.indptr[row_id + 1])
        return self.indices[start:end], self.data[start:end]

    def to_dense_rows(self, row_ids: Iterable[int]) -> np.ndarray:
        selected = [int(row_id) for row_id in row_ids]
        dense = np.zeros((len(selected), self.shape[1]), dtype=np.float64)
        for output_row, row_id in enumerate(selected):
            indices, values = self.row(row_id)
            dense[output_row, indices] = values
        return dense

    def memory_stats(self) -> dict[str, int | float | list[int]]:
        total_elements = self.shape[0] * self.shape[1]
        density = self._nnz / total_elements if total_elements else 0.0
        dense_bytes = total_elements * np.dtype(np.float64).itemsize
        sparse_bytes = self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
        compression_ratio = dense_bytes / sparse_bytes if sparse_bytes else 0.0
        return {
            "shape": [self.shape[0], self.shape[1]],
            "nnz": self._nnz,
            "density": density,
            "sparsity": 1.0 - density,
            "dense_bytes": dense_bytes,
            "sparse_bytes": sparse_bytes,
            "compression_ratio": compression_ratio,
        }
