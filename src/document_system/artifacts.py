"""Versioned persistence for the reusable search index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocessing import EnglishPreprocessor
from .privacy import is_safe_text
from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer


ARTIFACT_VERSION = 1


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
    if any(
        not isinstance(snippet, str) or not is_safe_text(snippet)
        for snippet in snippets
    ):
        raise ValueError("snippets must contain only nonblank safe terms")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "matrix.npz",
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
        "labels": np.asarray(labels).astype(int).tolist(),
        "target_names": list(target_names),
        "document_ids": np.asarray(document_ids).astype(int).tolist(),
        "privacy_policy": "safe-topic-terms-v1",
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def load_search_artifacts(directory: str | Path) -> SearchArtifacts:
    path = Path(directory)
    matrix_path = path / "matrix.npz"
    metadata_path = path / "metadata.json"
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"search artifacts are missing under {path}; run `python main.py build` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("artifact version mismatch; rebuild with `python main.py build`")
    required_fields = {"snippets", "document_ids", "privacy_policy"}
    if not required_fields.issubset(metadata):
        raise ValueError("artifact metadata is outdated; rebuild with `python main.py build`")
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
    labels = np.asarray(metadata["labels"], dtype=np.int32)
    target_names = tuple(str(name) for name in metadata["target_names"])
    document_ids = np.asarray(metadata["document_ids"], dtype=np.int64)
    if (
        len(snippets) != shape[0]
        or labels.size != shape[0]
        or document_ids.size != shape[0]
    ):
        raise ValueError("artifact document count mismatch; rebuild with `python main.py build`")
    return SearchArtifacts(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=snippets,
        labels=labels,
        target_names=target_names,
        document_ids=document_ids,
    )
