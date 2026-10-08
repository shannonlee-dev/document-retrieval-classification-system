import numpy as np
import pytest

from document_system.bm25 import NumpyBM25Retriever
from document_system.preprocessing import EnglishPreprocessor


def make_bm25(**kwargs):
    return NumpyBM25Retriever.from_texts(
        ["alpha alpha beta", "beta", "the"],
        snippets=["alpha alpha beta", "beta", "the"],
        labels=np.array([0, 1, 0]),
        target_names=("first", "second"),
        document_ids=np.array([90, 7, 42]),
        **kwargs,
    )


def test_bm25_matches_hand_idf_and_length_normalization():
    model = make_bm25()
    np.testing.assert_array_equal(model.document_lengths, [3, 1, 0])
    assert model.avgdl == pytest.approx(4 / 3)
    alpha = model.vectorizer.vocabulary_["alpha"]
    assert model.idf[alpha] == pytest.approx(np.log(1 + 2.5 / 1.5))
    expected = (
        np.log(1 + 2.5 / 1.5) * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 3 / (4 / 3)))
    )
    results = model.search("alpha", topk=3)
    assert [r.doc_id for r in results] == [90]
    assert results[0].score == pytest.approx(expected)


def test_unique_query_terms_and_parameter_effects():
    model = make_bm25()
    assert model.search("alpha beta", 2) == model.search("alpha alpha beta beta", 2)
    no_norm = make_bm25(b=0)
    assert no_norm.search("alpha", 1)[0].score > model.search("alpha", 1)[0].score
    saturated = make_bm25(k1=0)
    assert saturated.search("alpha", 1)[0].score == pytest.approx(np.log(1 + 2.5 / 1.5))


@pytest.mark.parametrize(
    "kwargs",
    [{"k1": -1}, {"k1": float("nan")}, {"b": -1}, {"b": 1.1}, {"b": float("inf")}],
)
def test_bm25_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        make_bm25(**kwargs)


def test_bm25_errors_and_external_id_ties():
    model = make_bm25()
    for query in ["", "unseenzz"]:
        with pytest.raises(ValueError):
            model.search(query, 1)
    for topk in [0, -1, 4]:
        with pytest.raises(ValueError):
            model.search("beta", topk)
    model = NumpyBM25Retriever.from_texts(
        ["alpha", "alpha"],
        snippets=["alpha", "alpha"],
        labels=np.array([0, 0]),
        target_names=("topic",),
        document_ids=np.array([99, 3]),
        preprocessor=EnglishPreprocessor(stop_words=frozenset()),
    )
    assert [r.doc_id for r in model.search("alpha", 2)] == [3, 99]


def test_zero_average_document_length_and_absent_posting_are_safe():
    from document_system.sparse_matrix import SparseMatrix
    from document_system.tfidf import NumpyTfidfVectorizer

    vectorizer = NumpyTfidfVectorizer().fit(["alpha"])
    counts = SparseMatrix(
        np.array([]),
        np.array([], dtype=np.int32),
        np.array([0, 0, 0], dtype=np.int32),
        (2, 1),
    )
    model = NumpyBM25Retriever(
        vectorizer,
        counts,
        ["the", "and"],
        np.array([0, 0]),
        ("topic",),
        np.array([99, 42]),
        np.array([0, 0]),
    )
    assert model.avgdl == 0
    assert model.search("alpha", 2) == []
