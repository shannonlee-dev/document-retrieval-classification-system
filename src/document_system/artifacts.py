"""Versioned persistence for the reusable search index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import SNIPPET_LIMIT
from .preprocessing import EnglishPreprocessor
from .privacy import is_safe_text
from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer

ARTIFACT_VERSION = 2
SEARCH_FIT_SCOPE = "full_corpus"
PRIVACY_POLICY = "structured-pii-redaction-v3"
MATRIX_FILENAME = "matrix.npz"
METADATA_FILENAME = "metadata.json"
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "snippets",
        "document_ids",
        "privacy_policy",
        "fit_scope",
        "fit_document_count",
        "category_count",
    }
)


@dataclass(frozen=True)
class SearchArtifacts:
    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    snippets: tuple[str, ...]
    labels: np.ndarray
    target_names: tuple[str, ...]
    document_ids: np.ndarray


def save_search_artifacts(
    directory: str | Path,
    vectorizer: NumpyTfidfVectorizer,
    matrix: SparseMatrix,
    snippets: Sequence[str],
    labels: np.ndarray,
    target_names: tuple[str, ...],
    document_ids: np.ndarray,
) -> None:
    if vectorizer.idf_ is None:
        raise RuntimeError("cannot save an unfitted vectorizer")
    snippets = tuple(snippets)
    labels = np.asarray(labels).astype(int)
    document_ids = np.asarray(document_ids).astype(int)
    target_names = tuple(target_names)
    if (
        len(snippets) != matrix.shape[0]
        or labels.ndim != 1
        or labels.size != matrix.shape[0]
        or document_ids.ndim != 1
        or document_ids.size != matrix.shape[0]
    ):
        raise ValueError("search artifact inputs must have row-aligned values")
    if not target_names:
        raise ValueError("search artifact target names must not be empty")
    if labels.size and (
        int(labels.min()) < 0 or int(labels.max()) >= len(target_names)
    ):
        raise ValueError("search artifact labels must be within target names")
    if any(
        not isinstance(snippet, str)
        or len(snippet) > SNIPPET_LIMIT
        or not is_safe_text(snippet)
        for snippet in snippets
    ):
        raise ValueError("snippets must be safe, nonblank, and within the length limit")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / MATRIX_FILENAME,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        idf=vectorizer.idf_,
    )
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "shape": list(matrix.shape),
        "data_dtype": str(matrix.data.dtype),
        "index_dtype": str(matrix.indices.dtype),
        "feature_names": list(vectorizer.feature_names_),
        "stop_words": sorted(vectorizer.preprocessor.stop_words),
        "snippets": list(snippets),
        "labels": labels.tolist(),
        "target_names": list(target_names),
        "document_ids": document_ids.tolist(),
        "privacy_policy": PRIVACY_POLICY,
        "fit_scope": SEARCH_FIT_SCOPE,
        "fit_document_count": matrix.shape[0],
        "category_count": len(target_names),
    }
    (path / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def load_search_artifacts(directory: str | Path) -> SearchArtifacts:
    path = Path(directory)
    matrix_path = path / MATRIX_FILENAME
    metadata_path = path / METADATA_FILENAME
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"search artifacts are missing under {path}; run `python main.py build` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("artifact version mismatch; rebuild with `python main.py build`")
    if not REQUIRED_METADATA_FIELDS.issubset(metadata):
        raise ValueError("artifact metadata is outdated; rebuild with `python main.py build`")
    if metadata["privacy_policy"] != PRIVACY_POLICY:
        raise ValueError("artifact privacy policy is outdated; rebuild with `python main.py build`")
    with np.load(matrix_path, allow_pickle=False) as arrays:
        data = arrays["data"].copy()
        indices = arrays["indices"].copy()
        indptr = arrays["indptr"].copy()
        idf = arrays["idf"].copy()
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 2:
        raise ValueError("invalid artifact shape; rebuild with `python main.py build`")
    feature_names = tuple(str(term) for term in metadata["feature_names"])
    if shape[1] != len(feature_names) or idf.size != len(feature_names):
        raise ValueError("artifact vocabulary mismatch; rebuild with `python main.py build`")
    preprocessor = EnglishPreprocessor(
        stop_words=frozenset(str(word) for word in metadata["stop_words"])
    )
    vectorizer = NumpyTfidfVectorizer(preprocessor)
    vectorizer.feature_names_ = feature_names
    vectorizer.vocabulary_ = {term: index for index, term in enumerate(feature_names)}
    vectorizer.idf_ = idf
    matrix = SparseMatrix(data=data, indices=indices, indptr=indptr, shape=shape)
    snippets = tuple(str(snippet) for snippet in metadata["snippets"])
    target_names = tuple(str(name) for name in metadata["target_names"])
    labels = np.asarray(metadata["labels"], dtype=np.int32)
    document_ids = np.asarray(metadata["document_ids"], dtype=np.int64)
    if metadata["fit_scope"] != SEARCH_FIT_SCOPE:
        raise ValueError("artifact fit scope is outdated; rebuild with `python main.py build`")
    if int(metadata["fit_document_count"]) != shape[0]:
        raise ValueError(
            "artifact fit document count mismatch; rebuild with `python main.py build`"
        )
    if int(metadata["category_count"]) != len(target_names):
        raise ValueError(
            "artifact category count mismatch; rebuild with `python main.py build`"
        )
    if labels.size and (
        int(labels.min()) < 0 or int(labels.max()) >= len(target_names)
    ):
        raise ValueError(
            "artifact labels are outside target names; rebuild with `python main.py build`"
        )
    if (
        len(snippets) != shape[0]
        or labels.size != shape[0]
        or document_ids.size != shape[0]
    ):
        raise ValueError("artifact document count mismatch; rebuild with `python main.py build`")
    if any(len(snippet) > SNIPPET_LIMIT or not is_safe_text(snippet) for snippet in snippets):
        raise ValueError("artifact snippets failed privacy checks; rebuild with `python main.py build`")
    return SearchArtifacts(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=snippets,
        labels=labels,
        target_names=target_names,
        document_ids=document_ids,
    )
