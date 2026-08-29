"""Dataset contracts and the 20 Newsgroups loader."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from .constants import DEFAULT_RANDOM_STATE, MINIMUM_DOCUMENTS
from .privacy import (
    PrivacyReport,
    sanitize_document,
)

SAFE_CATEGORIES = (
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
)
DATASET_SUBSET = "all"
METADATA_FIELDS_TO_REMOVE = ("headers", "footers", "quotes")
MINIMUM_CATEGORY_COUNT = 2


@dataclass(frozen=True)
class DatasetBundle:
    texts: tuple[str, ...]
    labels: np.ndarray
    target_names: tuple[str, ...]
    source_doc_ids: np.ndarray
    privacy_report: PrivacyReport | None = None


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


def load_20newsgroups() -> DatasetBundle:
    """Load the complete, metadata-stripped 20 Newsgroups corpus."""

    dataset = fetch_20newsgroups(
        subset=DATASET_SUBSET,
        categories=SAFE_CATEGORIES,
        remove=METADATA_FIELDS_TO_REMOVE,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )
    retained = []
    privacy_results = []
    for source_doc_id, (text, label) in enumerate(zip(dataset.data, dataset.target)):
        result = sanitize_document(text)
        privacy_results.append(result)
        if result.text:
            retained.append((source_doc_id, result.text, label))

    raw_labels = dataset.target
    source_doc_ids = np.asarray([row[0] for row in retained], dtype=np.int32)
    texts = tuple(row[1] for row in retained)
    labels = np.asarray([row[2] for row in retained], dtype=np.int32)
    target_names = tuple(dataset.target_names)
    _validate_dataset(texts, labels, source_doc_ids=source_doc_ids)
    return DatasetBundle(
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
