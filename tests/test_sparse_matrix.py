import numpy as np
import pytest

from document_system.sparse_matrix import SparseMatrix


def make_matrix() -> SparseMatrix:
    return SparseMatrix(
        data=np.array([2.0, 1.0, 3.0]),
        indices=np.array([0, 2, 1], dtype=np.int32),
        indptr=np.array([0, 2, 3], dtype=np.int32),
        shape=(2, 3),
    )


def test_sparse_matrix_restores_selected_rows() -> None:
    matrix = make_matrix()

    np.testing.assert_array_equal(
        matrix.to_dense_rows([1]),
        np.array([[0.0, 3.0, 0.0]]),
    )


def test_sparse_matrix_row_returns_nonzero_views() -> None:
    indices, values = make_matrix().get_sparse_row(0)

    np.testing.assert_array_equal(indices, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(values, np.array([2.0, 1.0]))


def test_memory_stats_use_actual_numpy_bytes() -> None:
    matrix = make_matrix()

    stats = matrix.memory_stats()

    assert stats["shape"] == [2, 3]
    assert stats["nnz"] == 3
    assert stats["density"] == pytest.approx(0.5)
    assert stats["sparsity"] == pytest.approx(0.5)
    assert stats["dense_bytes"] == 2 * 3 * 8
    assert stats["sparse_bytes"] == sum(
        array.nbytes for array in (matrix.data, matrix.indices, matrix.indptr)
    )


def test_sparse_matrix_rejects_unsorted_row_indices() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseMatrix(
            data=np.array([1.0, 2.0]),
            indices=np.array([2, 0], dtype=np.int32),
            indptr=np.array([0, 2], dtype=np.int32),
            shape=(1, 3),
        )
