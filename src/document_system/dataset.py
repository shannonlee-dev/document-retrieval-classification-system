"""Dataset contracts and the 20 Newsgroups loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_20newsgroups


@dataclass(frozen=True)
class DatasetBundle:
    texts: tuple[str, ...]
    labels: np.ndarray
    target_names: tuple[str, ...]


def validate_dataset(
    texts: Sequence[str],
    labels: Sequence[int | str],
    *,
    minimum_documents: int = 500,
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
    if len(set(labels)) < 2:
        raise ValueError("dataset must contain at least two labels")


def load_20newsgroups() -> DatasetBundle:
    """Load the complete, metadata-stripped 20 Newsgroups corpus."""

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    texts = tuple(dataset.data)
    labels = np.asarray(dataset.target, dtype=np.int32)
    target_names = tuple(dataset.target_names)
    validate_dataset(texts, labels)
    if len(texts) != 18_846 or len(target_names) != 20:
        raise ValueError(
            "expected the complete 20 Newsgroups dataset "
            f"(18,846 documents, 20 categories), got {len(texts):,} and {len(target_names)}"
        )
    return DatasetBundle(texts=texts, labels=labels, target_names=target_names)
