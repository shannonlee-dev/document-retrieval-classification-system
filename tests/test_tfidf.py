import numpy as np
import pytest

from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer


def make_processor() -> EnglishPreprocessor:
    return EnglishPreprocessor(stop_words=frozenset())


def test_tfidf_exposes_raw_tf_smoothed_idf_and_l2_values() -> None:
    texts = ["apple apple banana", "banana carrot"]
    vectorizer = NumpyTfidfVectorizer(make_processor())

    stages = vectorizer.fit_transform_with_stages(texts)

    apple = vectorizer.vocabulary_["apple"]
    banana = vectorizer.vocabulary_["banana"]
    dense_counts = stages.counts.to_dense_rows([0])
    assert dense_counts[0, apple] == 2.0
    assert dense_counts[0, banana] == 1.0
    assert vectorizer.idf_[apple] == pytest.approx(np.log(3 / 2) + 1)
    assert vectorizer.idf_[banana] == pytest.approx(1.0)
    assert np.linalg.norm(stages.tfidf.to_dense_rows([0])[0]) == pytest.approx(1.0)


def test_vocabulary_is_deterministic_and_alphabetical() -> None:
    vectorizer = NumpyTfidfVectorizer(make_processor()).fit(
        ["zebra apple", "banana apple"]
    )

    assert vectorizer.feature_names_ == ("apple", "banana", "zebra")
    assert vectorizer.vocabulary_ == {"apple": 0, "banana": 1, "zebra": 2}


def test_transform_ignores_out_of_vocabulary_terms_and_keeps_zero_row() -> None:
    vectorizer = NumpyTfidfVectorizer(make_processor()).fit(["known term"])

    transformed = vectorizer.transform(["unknown"])

    assert transformed.nnz == 0
    np.testing.assert_array_equal(
        transformed.to_dense_rows([0]),
        np.array([[0.0, 0.0]]),
    )


def test_fit_rejects_corpus_without_tokens() -> None:
    with pytest.raises(ValueError, match="vocabulary is empty"):
        NumpyTfidfVectorizer(make_processor()).fit(["123", "!"])


def test_transform_requires_fitted_vectorizer() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        NumpyTfidfVectorizer(make_processor()).transform(["apple"])
