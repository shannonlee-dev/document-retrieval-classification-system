"""Dataset contracts and the 20 Newsgroups loader."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from .constants import DEFAULT_RANDOM_STATE, MINIMUM_DOCUMENTS
from .privacy import (
    PrivacyReport,
    SanitizationResult,
    sanitize_document,
)

EXPECTED_CATEGORY_COUNT = 20
DATASET_SUBSET = "all"
METADATA_FIELDS_TO_REMOVE = ("headers", "footers", "quotes")
MINIMUM_CATEGORY_COUNT = 2


@dataclass(frozen=True)
class DatasetBundle:
    texts: tuple[str, ...]
    labels: np.ndarray
    target_names: tuple[str, ...]
    source_doc_ids: np.ndarray
    privacy_report: PrivacyReport


@dataclass(frozen=True)
class _RetainedDocument:
    source_doc_id: int
    text: str
    label: int


def _validate_dataset(
    texts: Sequence[str],
    labels: Sequence[int | str],
    *,
    source_doc_ids: Sequence[int] | None = None,
    minimum_documents: int = MINIMUM_DOCUMENTS,
) -> None:
    """Validate the portable text-classification input contract."""

    if minimum_documents < 1:
        raise ValueError("minimum_documents must be positive")
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if len(texts) < minimum_documents:
        raise ValueError(f"dataset must contain at least {minimum_documents} documents")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("all texts must be strings; missing values are not allowed")
    if any(not text.strip() for text in texts):
        raise ValueError("all texts must be nonblank")
    if source_doc_ids is not None and len(texts) != len(source_doc_ids):
        raise ValueError("texts and source document IDs must have the same length")
    if len(set(labels)) < MINIMUM_CATEGORY_COUNT:
        raise ValueError("dataset must contain at least two labels")


def validate_full_20_newsgroups(bundle: DatasetBundle) -> None:
    """Validate the fixed production 20 Newsgroups dataset contract."""

    expected_class_ids = set(range(EXPECTED_CATEGORY_COUNT))
    observed_class_ids = set(np.asarray(bundle.labels, dtype=np.int64).tolist())
    privacy_categories = set(bundle.privacy_report.category_counts)
    target_categories = set(bundle.target_names)
    if len(bundle.target_names) != EXPECTED_CATEGORY_COUNT:
        raise ValueError("the full build requires exactly 20 target categories")
    if observed_class_ids != expected_class_ids:
        raise ValueError("the full build requires all 20 categories and class IDs 0..19")
    if privacy_categories != target_categories:
        raise ValueError("privacy report categories must match all target categories")


def load_20newsgroups() -> DatasetBundle:
    """Load the complete, metadata-stripped 20 Newsgroups corpus."""

    dataset = fetch_20newsgroups(
        subset=DATASET_SUBSET,
        remove=METADATA_FIELDS_TO_REMOVE,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )
    retained_documents: list[_RetainedDocument] = []
    privacy_results: list[SanitizationResult] = []
    for source_doc_id, (text, label) in enumerate(zip(dataset.data, dataset.target)):
        sanitization_result = sanitize_document(text)
        privacy_results.append(sanitization_result)
        if sanitization_result.text:
            retained_documents.append(
                _RetainedDocument(
                    source_doc_id=source_doc_id,
                    text=sanitization_result.text,
                    label=int(label),
                )
            )

    raw_labels = dataset.target
    source_doc_ids = np.asarray(
        [document.source_doc_id for document in retained_documents],
        dtype=np.int32,
    )
    texts = tuple(document.text for document in retained_documents)
    labels = np.asarray(
        [document.label for document in retained_documents],
        dtype=np.int32,
    )
    target_names = tuple(dataset.target_names)
    _validate_dataset(texts, labels, source_doc_ids=source_doc_ids)
    bundle = DatasetBundle(
        texts=texts,
        labels=labels,
        target_names=target_names,
        source_doc_ids=source_doc_ids,
        privacy_report=PrivacyReport.from_sanitization_results(
            privacy_results,
            raw_labels,
            target_names,
        ),
    )
    validate_full_20_newsgroups(bundle)
    return bundle
