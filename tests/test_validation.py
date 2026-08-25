import numpy as np

from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer
from document_system.validation import stage_example, validate_against_sklearn


def test_numpy_tfidf_matches_sklearn_without_dense_full_matrix() -> None:
    texts = ["apple apple banana", "banana carrot", "carrot date"]
    processor = EnglishPreprocessor(stop_words=frozenset())
    vectorizer = NumpyTfidfVectorizer(processor)
    custom = vectorizer.fit_transform(texts)

    result = validate_against_sklearn(texts, vectorizer, custom)

    assert result.passed is True
    assert result.max_absolute_error <= 1e-6
    assert result.mean_absolute_error <= 1e-6
    assert result.shape == (3, 4)
    assert result.settings == {
        "smooth_idf": True,
        "sublinear_tf": False,
        "norm": "l2",
        "use_idf": True,
        "dtype": "float64",
    }


def test_stage_example_reports_values_from_each_tfidf_step() -> None:
    texts = ["apple apple banana", "banana carrot"]
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    stages = vectorizer.fit_transform_with_stages(texts)

    example = stage_example(stages, vectorizer, max_terms=2)

    assert example["document_id"] == 0
    apple = next(term for term in example["terms"] if term["term"] == "apple")
    assert apple["tf"] == 2.0
    assert apple["idf"] == vectorizer.idf_[vectorizer.vocabulary_["apple"]]
    assert apple["tfidf_before_norm"] == 2.0 * apple["idf"]
    expected = stages.tfidf.to_dense_rows([0])[0, vectorizer.vocabulary_["apple"]]
    assert apple["tfidf_after_norm"] == expected


def test_validation_detects_a_modified_custom_value() -> None:
    texts = ["apple banana", "banana carrot"]
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    custom = vectorizer.fit_transform(texts)
    custom.data[0] += 0.01

    result = validate_against_sklearn(texts, vectorizer, custom)

    assert result.passed is False
    assert result.max_absolute_error >= 0.009
