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
DEFAULT_VALIDATION_TOLERANCE = 1e-6
DEFAULT_STAGE_EXAMPLE_TERMS = 5
MINIMUM_STAGE_EXAMPLE_TERMS = 1


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
    tolerance: float = DEFAULT_VALIDATION_TOLERANCE,
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

    max_absolute_error = 0.0
    absolute_error_sum = 0.0
    for row_id in range(custom.shape[0]):
        custom_indices, custom_values = custom.get_sparse_row(row_id)
        reference_row_start, reference_row_end = (
            reference.indptr[row_id],
            reference.indptr[row_id + 1],
        )
        reference_indices = reference.indices[reference_row_start:reference_row_end]
        reference_values = reference.data[reference_row_start:reference_row_end]
        custom_position = reference_position = 0
        while (
            custom_position < custom_indices.size
            or reference_position < reference_indices.size
        ):
            if reference_position >= reference_indices.size or (
                custom_position < custom_indices.size
                and custom_indices[custom_position]
                < reference_indices[reference_position]
            ):
                absolute_error = abs(float(custom_values[custom_position]))
                custom_position += 1
            elif (
                custom_position >= custom_indices.size
                or reference_indices[reference_position]
                < custom_indices[custom_position]
            ):
                absolute_error = abs(float(reference_values[reference_position]))
                reference_position += 1
            else:
                absolute_error = abs(
                    float(
                        custom_values[custom_position]
                        - reference_values[reference_position]
                    )
                )
                custom_position += 1
                reference_position += 1
            max_absolute_error = max(max_absolute_error, absolute_error)
            absolute_error_sum += absolute_error

    element_count = custom.shape[0] * custom.shape[1]
    mean_absolute_error = absolute_error_sum / element_count if element_count else 0.0
    return ValidationResult(
        shape=custom.shape,
        max_absolute_error=max_absolute_error,
        mean_absolute_error=mean_absolute_error,
        tolerance=tolerance,
        passed=max_absolute_error <= tolerance,
        settings=dict(VALIDATION_SETTINGS),
    )


def stage_example(
    stages: TfidfStages,
    vectorizer: NumpyTfidfVectorizer,
    *,
    max_terms: int = DEFAULT_STAGE_EXAMPLE_TERMS,
) -> dict[str, object]:
    """Return traceable TF, IDF, and TF-IDF values from one real row."""

    if max_terms < MINIMUM_STAGE_EXAMPLE_TERMS:
        raise ValueError("max_terms must be positive")
    document_id = None
    for row_id in range(stages.counts.shape[0]):
        column_indices, _ = stages.counts.get_sparse_row(row_id)
        if column_indices.size:
            document_id = row_id
            break
    if document_id is None:
        raise ValueError("stage example requires a nonzero document")
    column_indices, term_counts = stages.counts.get_sparse_row(document_id)
    _, normalized_tfidf_values = stages.tfidf.get_sparse_row(document_id)
    terms: list[dict[str, int | float | str]] = []
    for column, term_count, normalized_tfidf_value in zip(
        column_indices[:max_terms],
        term_counts[:max_terms],
        normalized_tfidf_values[:max_terms],
        strict=True,
    ):
        column_id = int(column)
        idf = float(stages.idf[column_id])
        terms.append(
            {
                "term": vectorizer.feature_names_[column_id],
                "column": column_id,
                "tf": float(term_count),
                "idf": idf,
                "tfidf_before_norm": float(term_count * idf),
                "tfidf_after_norm": float(normalized_tfidf_value),
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
