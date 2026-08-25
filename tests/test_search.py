import numpy as np
import pytest

from document_system.preprocessing import EnglishPreprocessor
from document_system.search import DocumentSearch, sparse_dot
from document_system.tfidf import NumpyTfidfVectorizer


def make_small_searcher() -> DocumentSearch:
    texts = ["space shuttle orbit", "baseball pitcher game"]
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    matrix = vectorizer.fit_transform(texts)
    return DocumentSearch(
        vectorizer=vectorizer,
        matrix=matrix,
        texts=texts,
        labels=np.array([0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
    )


def test_sparse_dot_equals_dense_dot() -> None:
    left_indices = np.array([0, 3], dtype=np.int32)
    right_indices = np.array([1, 3], dtype=np.int32)

    result = sparse_dot(
        left_indices,
        np.array([0.5, 0.7]),
        right_indices,
        np.array([0.2, 0.4]),
    )

    assert result == pytest.approx(0.28)


def test_search_returns_score_id_label_and_snippet() -> None:
    searcher = make_small_searcher()

    result = searcher.search("shuttle orbit", topk=1)[0]

    assert result.doc_id == 0
    assert result.label == "space"
    assert result.score > 0
    assert result.text_snippet.startswith("space shuttle")


def test_search_rejects_oov_or_empty_query() -> None:
    searcher = make_small_searcher()

    with pytest.raises(ValueError, match="vocabulary"):
        searcher.search("zzzzunknown", topk=1)
    with pytest.raises(ValueError, match="vocabulary"):
        searcher.search("", topk=1)


def test_search_rejects_invalid_topk() -> None:
    searcher = make_small_searcher()

    with pytest.raises(ValueError, match="topk"):
        searcher.search("space", topk=0)
    with pytest.raises(ValueError, match="topk"):
        searcher.search("space", topk=3)


def test_equal_scores_are_ordered_by_document_id() -> None:
    searcher = make_small_searcher()

    results = searcher.search("space", topk=2)

    assert [result.doc_id for result in results] == [0, 1]
