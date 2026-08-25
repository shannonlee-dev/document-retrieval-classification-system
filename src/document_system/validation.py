"""Sparse equivalence checks against Scikit-learn TF-IDF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer, TfidfStages

VALIDATION_SETTINGS: dict[str, bool | str] = {
    "smooth_idf": True,
    "sublinear_tf": False,
    "norm": "l2",
    "use_idf": True,
    "dtype": "float64",
}


@dataclass(frozen=True)
class ValidationResult:
    shape: tuple[int, int]
    max_absolute_error: float
    mean_absolute_error: float
    tolerance: float
    passed: bool
    settings: dict[str, bool | str]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


def validate_against_sklearn(
    texts: Sequence[str],
    vectorizer: NumpyTfidfVectorizer,
    custom: SparseMatrix,
    *,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Compare two sparse matrices exactly without densifying the corpus."""

    if vectorizer.idf_ is None:
        raise RuntimeError("vectorizer is not fitted")
    reference_vectorizer = TfidfVectorizer(
        analyzer=vectorizer.preprocessor.tokenize,
        vocabulary=vectorizer.vocabulary_,
        lowercase=False,
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
        use_idf=True,
        dtype=np.float64,
    )
    reference = reference_vectorizer.fit_transform(texts)
    reference.sort_indices()
    if reference.shape != custom.shape:
        raise ValueError(
            f"matrix shape mismatch: custom={custom.shape}, sklearn={reference.shape}"
        )

    max_error = 0.0
    absolute_error_sum = 0.0
    for row_id in range(custom.shape[0]):
        custom_indices, custom_values = custom.row(row_id)
        start, end = reference.indptr[row_id], reference.indptr[row_id + 1]
        reference_indices = reference.indices[start:end]
        reference_values = reference.data[start:end]
        left = right = 0
        while left < custom_indices.size or right < reference_indices.size:
            if right >= reference_indices.size or (
                left < custom_indices.size
                and custom_indices[left] < reference_indices[right]
            ):
                error = abs(float(custom_values[left]))
                left += 1
            elif left >= custom_indices.size or reference_indices[right] < custom_indices[left]:
                error = abs(float(reference_values[right]))
                right += 1
            else:
                error = abs(float(custom_values[left] - reference_values[right]))
                left += 1
                right += 1
            max_error = max(max_error, error)
            absolute_error_sum += error

    element_count = custom.shape[0] * custom.shape[1]
    mean_error = absolute_error_sum / element_count if element_count else 0.0
    return ValidationResult(
        shape=custom.shape,
        max_absolute_error=max_error,
        mean_absolute_error=mean_error,
        tolerance=tolerance,
        passed=max_error <= tolerance,
        settings=dict(VALIDATION_SETTINGS),
    )


def stage_example(
    stages: TfidfStages,
    vectorizer: NumpyTfidfVectorizer,
    *,
    max_terms: int = 5,
) -> dict[str, object]:
    """Return traceable TF, IDF, and TF-IDF values from one real row."""

    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    document_id = next(
        (row_id for row_id in range(stages.counts.shape[0]) if stages.counts.row(row_id)[0].size),
        None,
    )
    if document_id is None:
        raise ValueError("stage example requires a nonzero document")
    indices, counts = stages.counts.row(document_id)
    _, normalized = stages.tfidf.row(document_id)
    terms: list[dict[str, int | float | str]] = []
    for column, count, tfidf_value in zip(
        indices[:max_terms],
        counts[:max_terms],
        normalized[:max_terms],
        strict=True,
    ):
        column_id = int(column)
        idf = float(stages.idf[column_id])
        terms.append(
            {
                "term": vectorizer.feature_names_[column_id],
                "column": column_id,
                "tf": float(count),
                "idf": idf,
                "tfidf_before_norm": float(count * idf),
                "tfidf_after_norm": float(tfidf_value),
            }
        )
    return {
        "document_id": document_id,
        "formulas": {
            "tf": "count(t, d)",
            "idf": "log((1 + N) / (1 + df(t))) + 1",
            "tfidf": "tf * idf",
            "normalization": "tfidf / L2_norm(tfidf)",
        },
        "terms": terms,
    }
