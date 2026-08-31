import numpy as np
import pytest

from document_system.preprocessing import EnglishPreprocessor
from document_system.privacy import SNIPPET_LIMIT
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
        snippets=texts,
        labels=np.array([0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
        document_ids=np.array([42, 99], dtype=np.int64),
    )


def test_sparse_dot_equals_dense_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    left_indices = np.array([0, 3], dtype=np.int32)
    right_indices = np.array([1, 3], dtype=np.int32)

    intersect1d = np.intersect1d
    dot = np.dot
    calls: list[str] = []

    def record_intersect1d(*args: object, **kwargs: object) -> object:
        calls.append("intersect1d")
        return intersect1d(*args, **kwargs)

    def record_dot(*args: object, **kwargs: object) -> object:
        calls.append("dot")
        return dot(*args, **kwargs)

    monkeypatch.setattr(np, "intersect1d", record_intersect1d)
    monkeypatch.setattr(np, "dot", record_dot)

    result = sparse_dot(
        left_indices,
        np.array([0.5, 0.7]),
        right_indices,
        np.array([0.2, 0.4]),
    )

    assert result == pytest.approx(0.28)
    assert calls == ["intersect1d", "dot"]


def test_search_returns_score_id_label_and_snippet() -> None:
    searcher = make_small_searcher()

    result = searcher.search("shuttle orbit", topk=1)[0]

    assert result.doc_id == 42
    assert result.label == "space"
    assert result.score > 0
    assert result.text_snippet.startswith("space shuttle")


def test_search_rejects_oov_query() -> None:
    searcher = make_small_searcher()

    with pytest.raises(ValueError, match="vocabulary"):
        searcher.search("zzzzunknown", topk=1)


def test_search_rejects_blank_query() -> None:
    searcher = make_small_searcher()

    with pytest.raises(ValueError, match="blank"):
        searcher.search("", topk=1)


def test_search_rejects_invalid_topk() -> None:
    searcher = make_small_searcher()

    with pytest.raises(ValueError, match="topk"):
        searcher.search("space", topk=0)
    with pytest.raises(ValueError, match="topk"):
        searcher.search("space", topk=3)


def test_search_rejects_unsafe_or_blank_snippets() -> None:
    searcher = make_small_searcher()

    for snippets in (["space shuttle orbit", "Alice"], ["space shuttle orbit", " "]):
        with pytest.raises(ValueError, match="safe"):
            DocumentSearch(
                vectorizer=searcher.vectorizer,
                matrix=searcher.matrix,
                snippets=snippets,
                labels=searcher.labels,
                target_names=searcher.target_names,
                document_ids=searcher.document_ids,
            )

    with pytest.raises(ValueError, match="snippet"):
        DocumentSearch(
            vectorizer=searcher.vectorizer,
            matrix=searcher.matrix,
            snippets=["space shuttle orbit", "x" * (SNIPPET_LIMIT + 1)],
            labels=searcher.labels,
            target_names=searcher.target_names,
            document_ids=searcher.document_ids,
        )


def test_search_omits_zero_score_results() -> None:
    searcher = make_small_searcher()

    results = searcher.search("space", topk=2)

    assert [result.doc_id for result in results] == [42]


def test_equal_positive_scores_are_ordered_by_document_id() -> None:
    texts = ["space shuttle", "space galaxy"]
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    searcher = DocumentSearch(
        vectorizer=vectorizer,
        matrix=vectorizer.fit_transform(texts),
        snippets=texts,
        labels=np.array([0, 0], dtype=np.int32),
        target_names=("space",),
        document_ids=np.array([99, 42], dtype=np.int64),
    )

    results = searcher.search("space", topk=2)

    assert results[0].score > 0
    assert results[0].score == pytest.approx(results[1].score)
    assert [result.doc_id for result in results] == [42, 99]
