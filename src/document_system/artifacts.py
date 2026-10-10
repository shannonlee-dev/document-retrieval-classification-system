"""Versioned persistence for the reusable search index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import DEFAULT_BM25_B, DEFAULT_BM25_K1, NumpyBM25Retriever
from .constants import SNIPPET_LIMIT, TF_MODES
from .inverted_index import InvertedIndex
from .preprocessing import EnglishPreprocessor
from .privacy import is_safe_text
from .search import DocumentSearch
from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer

_HASH_CHUNK_BYTES = 1024 * 1024

ARTIFACT_VERSION = 3
SEARCH_FIT_SCOPE = "full_corpus"
PRIVACY_POLICY = "structured-pii-redaction-v3"
SEARCH_INDEX_ARRAYS_FILENAME = "search_index_arrays.npz"
SEARCH_INDEX_DATA_FILENAME = "search_index_data.json"
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "snippets",
        "document_ids",
        "privacy_policy",
        "fit_scope",
        "fit_document_count",
        "category_count",
        "tf_mode",
        "arrays_sha256",
        "metadata_sha256",
        "corpus_fingerprint",
    }
)


@dataclass(frozen=True)
class SearchArtifacts:
    """Hold a restored TF-IDF index and its validated metadata."""

    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    snippets: tuple[str, ...]
    labels: np.ndarray
    target_names: tuple[str, ...]
    document_ids: np.ndarray
    extra_arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class RetrievalArtifacts:
    """Hold full-corpus TF-IDF and BM25 indexes for offline retrieval."""

    tfidf: dict[str, DocumentSearch]
    bm25: NumpyBM25Retriever
    corpus_fingerprint: str
    metadata: dict[str, object]


def save_search_artifacts(
    directory: str | Path,
    vectorizer: NumpyTfidfVectorizer,
    matrix: SparseMatrix,
    snippets: Sequence[str],
    labels: np.ndarray,
    target_names: tuple[str, ...],
    document_ids: np.ndarray,
    *,
    counts: SparseMatrix | None = None,
    document_lengths: np.ndarray | None = None,
    corpus_fingerprint: str | None = None,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
) -> None:
    """Persist fitted arrays and safe previews with metadata and file checksums."""
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
    extra_arrays: dict[str, np.ndarray] = {}
    if counts is not None:
        if vectorizer.tf_mode != "raw":
            raise ValueError("full retrieval bundle must use raw baseline")
        if (
            counts.shape != matrix.shape
            or not np.array_equal(counts.indices, matrix.indices)
            or not np.array_equal(counts.indptr, matrix.indptr)
        ):
            raise ValueError("counts and TF-IDF must share sparse structure")
        if document_lengths is None:
            raise ValueError("full retrieval bundle requires document lengths")
        order = np.argsort(matrix.indices, kind="stable")
        inverted = InvertedIndex.from_matrix(matrix)
        extra_arrays.update(
            counts=counts.data,
            document_lengths=np.asarray(document_lengths),
            posting_order=order,
            posting_rows=inverted.document_rows,
            posting_indptr=inverted.term_indptr,
        )
        for mode in ("log", "double"):
            variant = NumpyTfidfVectorizer(vectorizer.preprocessor, tf_mode=mode)
            variant.vocabulary_ = vectorizer.vocabulary_
            variant.feature_names_ = vectorizer.feature_names_
            variant.idf_ = vectorizer.idf_
            extra_arrays[f"tf_{mode}"] = variant.weight_counts(counts).data
        bm25 = NumpyBM25Retriever(
            vectorizer,
            counts,
            snippets,
            labels,
            target_names,
            document_ids,
            document_lengths,
            k1=k1,
            b=b,
        )
        extra_arrays["bm25_weights"] = bm25.inverted_index.weights
        extra_arrays["bm25_idf"] = bm25.idf
    np.savez_compressed(
        path / SEARCH_INDEX_ARRAYS_FILENAME,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        idf=vectorizer.idf_,
        **extra_arrays,
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
        "tf_mode": vectorizer.tf_mode,
        "retrieval_methods": [*TF_MODES, "bm25"]
        if counts is not None
        else [vectorizer.tf_mode],
        "bm25_parameters": {"k1": k1, "b": b, "query_terms": "unique"},
        "dataset": "20 Newsgroups",
        "arrays_sha256": _file_digest(path / SEARCH_INDEX_ARRAYS_FILENAME),
    }
    metadata["corpus_fingerprint"] = corpus_fingerprint or _metadata_digest(metadata)
    metadata["metadata_sha256"] = _metadata_digest(metadata)
    (path / SEARCH_INDEX_DATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def load_search_artifacts(directory: str | Path) -> SearchArtifacts:
    """Restore a fitted index after validating version, integrity and previews."""
    path = Path(directory)
    matrix_path = path / SEARCH_INDEX_ARRAYS_FILENAME
    metadata_path = path / SEARCH_INDEX_DATA_FILENAME
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"search artifacts are missing under {path}; run `document-system build` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(
            "invalid artifact metadata; rebuild with `document-system build`"
        )
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            "artifact version mismatch; rebuild with `document-system build`"
        )
    if not REQUIRED_METADATA_FIELDS.issubset(metadata):
        raise ValueError(
            "artifact metadata is outdated; rebuild with `document-system build`"
        )
    if metadata["privacy_policy"] != PRIVACY_POLICY:
        raise ValueError(
            "artifact privacy policy is outdated; rebuild with `document-system build`"
        )
    if metadata["metadata_sha256"] != _metadata_digest(metadata) or metadata[
        "arrays_sha256"
    ] != _file_digest(matrix_path):
        raise ValueError(
            "artifact integrity/corpus mismatch; rebuild with `document-system build`"
        )
    with np.load(matrix_path, allow_pickle=False) as arrays:
        data = arrays["data"].copy()
        indices = arrays["indices"].copy()
        indptr = arrays["indptr"].copy()
        idf = arrays["idf"].copy()
        extra_arrays = {
            key: arrays[key].copy()
            for key in arrays.files
            if key not in {"data", "indices", "indptr", "idf"}
        }
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 2:
        raise ValueError("invalid artifact shape; rebuild with `document-system build`")
    feature_names = tuple(str(term) for term in metadata["feature_names"])
    if shape[1] != len(feature_names) or idf.size != len(feature_names):
        raise ValueError(
            "artifact vocabulary mismatch; rebuild with `document-system build`"
        )
    preprocessor = EnglishPreprocessor(
        stop_words=frozenset(str(word) for word in metadata["stop_words"])
    )
    vectorizer = NumpyTfidfVectorizer(preprocessor, tf_mode=metadata["tf_mode"])
    vectorizer.feature_names_ = feature_names
    vectorizer.vocabulary_ = {term: index for index, term in enumerate(feature_names)}
    vectorizer.idf_ = idf
    matrix = SparseMatrix(data=data, indices=indices, indptr=indptr, shape=shape)
    snippets = tuple(str(snippet) for snippet in metadata["snippets"])
    target_names = tuple(str(name) for name in metadata["target_names"])
    labels = np.asarray(metadata["labels"], dtype=np.int32)
    document_ids = np.asarray(metadata["document_ids"], dtype=np.int64)
    if metadata["fit_scope"] != SEARCH_FIT_SCOPE:
        raise ValueError(
            "artifact fit scope is outdated; rebuild with `document-system build`"
        )
    if int(metadata["fit_document_count"]) != shape[0]:
        raise ValueError(
            "artifact fit document count mismatch; rebuild with `document-system build`"
        )
    if int(metadata["category_count"]) != len(target_names):
        raise ValueError(
            "artifact category count mismatch; rebuild with `document-system build`"
        )
    if labels.size and (
        int(labels.min()) < 0 or int(labels.max()) >= len(target_names)
    ):
        raise ValueError(
            "artifact labels are outside target names; rebuild with `document-system build`"
        )
    if (
        len(snippets) != shape[0]
        or labels.size != shape[0]
        or document_ids.size != shape[0]
    ):
        raise ValueError(
            "artifact document count mismatch; rebuild with `document-system build`"
        )
    if any(
        len(snippet) > SNIPPET_LIMIT or not is_safe_text(snippet)
        for snippet in snippets
    ):
        raise ValueError(
            "artifact snippets failed privacy checks; rebuild with `document-system build`"
        )
    return SearchArtifacts(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=snippets,
        labels=labels,
        target_names=target_names,
        document_ids=document_ids,
        extra_arrays=extra_arrays,
        metadata=metadata,
    )


def load_retrievers(directory: str | Path) -> RetrievalArtifacts:
    """Restore full-corpus indexes without tokenization, fitting or sorting."""
    saved = load_search_artifacts(directory)
    arrays = saved.extra_arrays
    required = {
        "counts",
        "document_lengths",
        "posting_order",
        "posting_rows",
        "posting_indptr",
        "tf_log",
        "tf_double",
        "bm25_weights",
        "bm25_idf",
    }
    if not required.issubset(arrays):
        raise ValueError(
            "full retrieval artifacts are missing; rebuild with `document-system build`"
        )
    matrix = saved.matrix
    order = arrays["posting_order"]
    if (
        order.shape != matrix.data.shape
        or not np.issubdtype(order.dtype, np.integer)
        or np.any(order < 0)
        or np.any(order >= matrix._nnz)
    ):
        raise ValueError("invalid posting order; rebuild with `document-system build`")
    common = dict(
        snippets=saved.snippets,
        labels=saved.labels,
        target_names=saved.target_names,
        document_ids=saved.document_ids,
    )
    searchers = {}
    for mode in TF_MODES:
        vectorizer = NumpyTfidfVectorizer(saved.vectorizer.preprocessor, tf_mode=mode)
        vectorizer.vocabulary_ = saved.vectorizer.vocabulary_
        vectorizer.feature_names_ = saved.vectorizer.feature_names_
        vectorizer.idf_ = saved.vectorizer.idf_
        values = matrix.data if mode == "raw" else arrays[f"tf_{mode}"]
        weighted = SparseMatrix(values, matrix.indices, matrix.indptr, matrix.shape)
        postings = InvertedIndex(
            arrays["posting_indptr"],
            arrays["posting_rows"],
            values[order],
            matrix.shape[0],
        )
        searchers[mode] = DocumentSearch(
            vectorizer, weighted, **common, inverted_index=postings
        )
    counts = SparseMatrix(arrays["counts"], matrix.indices, matrix.indptr, matrix.shape)
    bm25_index = InvertedIndex(
        arrays["posting_indptr"],
        arrays["posting_rows"],
        arrays["bm25_weights"],
        matrix.shape[0],
    )
    parameters = saved.metadata["bm25_parameters"]
    bm25 = NumpyBM25Retriever(
        saved.vectorizer,
        counts,
        **common,
        document_lengths=arrays["document_lengths"],
        k1=parameters["k1"],
        b=parameters["b"],
        inverted_index=bm25_index,
    )
    if not np.array_equal(bm25.idf, arrays["bm25_idf"]):
        raise ValueError("BM25 IDF mismatch; rebuild with `document-system build`")
    return RetrievalArtifacts(
        searchers, bm25, str(saved.metadata["corpus_fingerprint"]), saved.metadata
    )


def _metadata_digest(metadata: dict[str, object]) -> str:
    payload = {
        key: value for key, value in metadata.items() if key != "metadata_sha256"
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
