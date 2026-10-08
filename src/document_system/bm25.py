"""Direct NumPy Okapi BM25, using unique query terms and full token lengths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import DEFAULT_TOP_K
from .inverted_index import InvertedIndex
from .preprocessing import EnglishPreprocessor
from .search import DocumentSearch, SearchResult
from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer


class NumpyBM25Retriever:
    def __init__(
        self,
        vectorizer: NumpyTfidfVectorizer,
        counts: SparseMatrix,
        snippets: Sequence[str],
        labels: np.ndarray,
        target_names: tuple[str, ...],
        document_ids: np.ndarray,
        document_lengths: np.ndarray,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        inverted_index: InvertedIndex | None = None,
    ) -> None:
        if not np.isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be finite and non-negative")
        if not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")
        lengths = np.asarray(document_lengths, dtype=np.float64)
        if (
            lengths.shape != (counts.shape[0],)
            or np.any(lengths < 0)
            or not np.all(np.isfinite(lengths))
        ):
            raise ValueError(
                "document lengths must be finite, non-negative and row aligned"
            )
        if np.any(counts.data <= 0) or not np.all(np.isfinite(counts.data)):
            raise ValueError("BM25 counts must be positive and finite")
        self.vectorizer = vectorizer
        self.counts = counts
        self.document_lengths = lengths
        self.avgdl = float(np.mean(lengths)) if lengths.size else 0.0
        self.k1, self.b = float(k1), float(b)
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log1p((counts.shape[0] - df + 0.5) / (df + 0.5))
        self.documents = DocumentSearch(
            vectorizer, counts, snippets, labels, target_names, document_ids
        )
        if inverted_index is None:
            row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
            ratios = lengths / self.avgdl if self.avgdl else np.zeros_like(lengths)
            denominator = counts.data + k1 * (1 - b + b * ratios[row_ids])
            weights = self.idf[counts.indices] * counts.data * (k1 + 1) / denominator
            weighted = SparseMatrix(
                weights, counts.indices, counts.indptr, counts.shape
            )
            inverted_index = InvertedIndex.from_matrix(weighted)
        self.inverted_index = inverted_index

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        snippets: Sequence[str],
        labels: np.ndarray,
        target_names: tuple[str, ...],
        document_ids: np.ndarray,
        preprocessor: EnglishPreprocessor | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> NumpyBM25Retriever:
        vectorizer = NumpyTfidfVectorizer(preprocessor)
        counts = vectorizer.fit_transform_with_stages(texts).counts
        lengths = np.array(
            [len(vectorizer.preprocessor.tokenize(text)) for text in texts]
        )
        return cls(
            vectorizer,
            counts,
            snippets,
            labels,
            target_names,
            document_ids,
            lengths,
            k1=k1,
            b=b,
        )

    def score_query(self, query: str) -> dict[int, float]:
        if not query.strip():
            raise ValueError("query must not be blank")
        terms = sorted(
            {
                self.vectorizer.vocabulary_[token]
                for token in self.vectorizer.preprocessor.tokenize(query)
                if token in self.vectorizer.vocabulary_
            }
        )
        if not terms:
            raise ValueError("query has no terms in the fitted vocabulary")
        return self.inverted_index.score(
            np.asarray(terms, dtype=np.int32), np.ones(len(terms))
        )

    def search(self, query: str, topk: int = DEFAULT_TOP_K) -> list[SearchResult]:
        self.documents.validate_topk(topk)
        return self.documents.results_from_scores(self.score_query(query), topk)
