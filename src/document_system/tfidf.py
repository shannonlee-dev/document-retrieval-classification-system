"""NumPy-only sparse TF-IDF fitting and transformation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .preprocessing import EnglishPreprocessor
from .sparse_matrix import SparseMatrix


@dataclass(frozen=True)
class TfidfStages:
    counts: SparseMatrix
    idf: np.ndarray
    tfidf: SparseMatrix


class NumpyTfidfVectorizer:
    """Fit raw-count TF-IDF while storing only nonzero values."""

    def __init__(self, preprocessor: EnglishPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or EnglishPreprocessor()
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: tuple[str, ...] = ()
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> NumpyTfidfVectorizer:
        self.fit_transform_with_stages(texts)
        return self

    def fit_transform(self, texts: Sequence[str]) -> SparseMatrix:
        return self.fit_transform_with_stages(texts).tfidf

    def fit_transform_with_stages(self, texts: Sequence[str]) -> TfidfStages:
        feature_names = tuple(
            sorted(
                {
                    token
                    for text in texts
                    for token in self.preprocessor.tokenize(text)
                }
            )
        )
        if not feature_names:
            raise ValueError("vocabulary is empty after preprocessing")
        self.feature_names_ = feature_names
        self.vocabulary_ = {term: index for index, term in enumerate(feature_names)}
        counts = self._count_matrix(texts)
        document_frequency = np.bincount(
            counts.indices,
            minlength=len(self.vocabulary_),
        ).astype(np.float64)
        self.idf_ = np.log((1.0 + len(texts)) / (1.0 + document_frequency)) + 1.0
        tfidf = self._weight_and_normalize(counts)
        return TfidfStages(counts=counts, idf=self.idf_.copy(), tfidf=tfidf)

    def transform(self, texts: Sequence[str]) -> SparseMatrix:
        if self.idf_ is None or not self.vocabulary_:
            raise RuntimeError("vectorizer is not fitted")
        return self._weight_and_normalize(self._count_matrix(texts))

    def _count_matrix(self, texts: Sequence[str]) -> SparseMatrix:
        values: list[float] = []
        columns: list[int] = []
        row_pointers = [0]
        for text in texts:
            token_counts = Counter(self.preprocessor.tokenize(text))
            indexed_counts = sorted(
                (self.vocabulary_[token], count)
                for token, count in token_counts.items()
                if token in self.vocabulary_
            )
            columns.extend(index for index, _ in indexed_counts)
            values.extend(float(count) for _, count in indexed_counts)
            row_pointers.append(len(values))
        pointer_dtype = np.int32 if len(values) <= np.iinfo(np.int32).max else np.int64
        return SparseMatrix(
            data=np.asarray(values, dtype=np.float64),
            indices=np.asarray(columns, dtype=np.int32),
            indptr=np.asarray(row_pointers, dtype=pointer_dtype),
            shape=(len(texts), len(self.vocabulary_)),
        )

    def _weight_and_normalize(self, counts: SparseMatrix) -> SparseMatrix:
        if self.idf_ is None:
            raise RuntimeError("vectorizer is not fitted")
        weighted = counts.data * self.idf_[counts.indices]
        for row_id in range(counts.shape[0]):
            start, end = int(counts.indptr[row_id]), int(counts.indptr[row_id + 1])
            if start == end:
                continue
            norm = float(np.linalg.norm(weighted[start:end]))
            if norm:
                weighted[start:end] /= norm
        return SparseMatrix(
            data=weighted,
            indices=counts.indices.copy(),
            indptr=counts.indptr.copy(),
            shape=counts.shape,
        )
