import numpy as np
import pytest

from document_system.inverted_index import InvertedIndex
from document_system.search import DocumentSearch
from document_system.tfidf import NumpyTfidfVectorizer


def _make_searcher(mode="raw"):
    texts = ["orbit space", "orbit rocket", "space orbit", "baseball game", "the"]
    vectorizer = NumpyTfidfVectorizer(tf_mode=mode)
    return DocumentSearch(
        vectorizer,
        vectorizer.fit_transform(texts),
        texts,
        np.array([0, 0, 0, 1, 0]),
        ("space", "sport"),
        np.array([90, 3, 7, 12, 99]),
    )


def test_postings_use_internal_rows_and_preserve_weights():
    searcher = _make_searcher()
    index = InvertedIndex.from_matrix(searcher.matrix)
    rows, weights = index.posting(searcher.vectorizer.vocabulary_["orbit"])
    np.testing.assert_array_equal(rows, [0, 1, 2])
    assert weights[0] == pytest.approx(weights[2])
    assert index.memory_bytes > 0
    assert index.score(np.array([], dtype=int), np.array([])) == {}


@pytest.mark.parametrize("mode", ["raw", "log", "double"])
@pytest.mark.parametrize("query", ["space", "orbit rocket space", "orbit orbit"])
@pytest.mark.parametrize("topk", [1, 3, 5])
def test_scan_and_inverted_have_identical_ranks_scores_and_ties(mode, query, topk):
    searcher = _make_searcher(mode)
    scan = searcher.search(query, topk, strategy="scan")
    inverted = searcher.search(query, topk, strategy="inverted")
    assert [r.doc_id for r in scan] == [r.doc_id for r in inverted]
    np.testing.assert_allclose(
        [r.score for r in scan], [r.score for r in inverted], atol=1e-12, rtol=0
    )
    if query == "space" and topk >= 3:
        assert [r.doc_id for r in scan] == [7, 90]


@pytest.mark.parametrize("strategy", ["scan", "inverted"])
def test_search_strategy_errors(strategy):
    searcher = _make_searcher()
    for query in ["", "unknownzz"]:
        with pytest.raises(ValueError):
            searcher.search(query, 1, strategy=strategy)
    with pytest.raises(ValueError, match="strategy"):
        searcher.search("space", 1, strategy="bad")


def test_single_document_and_absent_posting():
    vectorizer = NumpyTfidfVectorizer()
    matrix = vectorizer.fit_transform(["space orbit"])
    searcher = DocumentSearch(
        vectorizer, matrix, ["space orbit"], np.array([0]), ("space",), np.array([42])
    )
    assert searcher.search("space", 1, strategy="inverted")[0].doc_id == 42


def test_known_term_with_empty_posting_returns_no_positive_results():
    from document_system.sparse_matrix import SparseMatrix

    vectorizer = NumpyTfidfVectorizer()
    full = vectorizer.fit_transform(["space orbit", "rocket"])
    indices, values = full.get_sparse_row(0)
    matrix = SparseMatrix(
        values, indices, np.array([0, len(values)], dtype=np.int32), (1, full.shape[1])
    )
    postings = InvertedIndex.from_matrix(matrix)
    rows, weights = postings.posting(vectorizer.vocabulary_["rocket"])
    assert rows.size == weights.size == 0
    searcher = DocumentSearch(
        vectorizer,
        matrix,
        ["space orbit"],
        np.array([0]),
        ("space",),
        np.array([42]),
        postings,
    )
    assert searcher.search("rocket", 1, strategy="scan") == []
    assert searcher.search("rocket", 1, strategy="inverted") == []
