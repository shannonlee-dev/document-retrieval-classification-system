import numpy as np
import pytest

from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer
from document_system.validation import validate_against_sklearn


@pytest.mark.parametrize("mode", ["raw", "log"])
def test_variants_match_sklearn_and_normalize_empty_rows(mode):
    texts = ["orbit orbit space", "space rocket", "", "the and"]
    vectorizer = NumpyTfidfVectorizer(tf_mode=mode)
    matrix = vectorizer.fit_transform(texts)
    result = validate_against_sklearn(texts, vectorizer, matrix)
    assert result.passed
    assert result.max_absolute_error <= 1e-6
    assert result.settings["sublinear_tf"] == (mode == "log")
    assert np.linalg.norm(matrix.get_sparse_row(0)[1]) == pytest.approx(1)
    assert matrix.get_sparse_row(2)[1].size == 0
    assert vectorizer.transform(["unknownzz"])._nnz == 0


def test_double_tf_matches_hand_calculation_and_reuses_counts():
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset()), tf_mode="double"
    )
    stages = vectorizer.fit_transform_with_stages(["alpha alpha beta", "beta", ""])
    # df(alpha)=1, df(beta)=2; double TF is [1, .75] in first row.
    expected = np.array([1 + np.log(4 / 2), 0.75 * (1 + np.log(4 / 3))])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(stages.tfidf.get_sparse_row(0)[1], expected, atol=1e-6)
    np.testing.assert_array_equal(stages.counts.get_sparse_row(0)[1], [2, 1])
    np.testing.assert_array_equal(stages.tfidf.get_sparse_row(1)[1], [1])
    assert stages.tfidf.get_sparse_row(2)[1].size == 0
    np.testing.assert_array_equal(
        vectorizer.weight_counts(stages.counts).data, stages.tfidf.data
    )
    with pytest.raises(ValueError, match="double"):
        validate_against_sklearn(["alpha"], vectorizer, stages.tfidf)


def test_invalid_tf_mode_is_rejected():
    with pytest.raises(ValueError, match="tf_mode"):
        NumpyTfidfVectorizer(tf_mode="unsupported")
