from types import SimpleNamespace

import numpy as np
import pytest

import document_system.dataset as dataset_module
from document_system.dataset import validate_dataset
from document_system.privacy import is_safe_text


def test_validate_dataset_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_dataset(["one", "two"], [0], minimum_documents=2)


def test_validate_dataset_requires_two_labels() -> None:
    with pytest.raises(ValueError, match="two labels"):
        validate_dataset(["one", "two"], [0, 0], minimum_documents=2)


def test_validate_dataset_rejects_missing_or_non_string_text() -> None:
    with pytest.raises(ValueError, match="strings"):
        validate_dataset(["one", None], [0, 1], minimum_documents=2)  # type: ignore[list-item]


def test_validate_dataset_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        validate_dataset(["space", "   "], [0, 1], minimum_documents=2)


def test_validate_dataset_checks_source_document_ids() -> None:
    with pytest.raises(ValueError, match="document IDs"):
        validate_dataset(
            ["space", "baseball"],
            [0, 1],
            source_doc_ids=[7],
            minimum_documents=2,
        )


def test_loader_returns_only_safe_nonblank_documents(monkeypatch) -> None:
    raw_texts = (
        ["Alice alice@example.com image pixel"] * 200
        + ["Bob 010-1234-5678 baseball pitcher"] * 200
        + ["medical treatment details rocket orbit"] * 200
        + ["Alice alice@example.com treatment"]
    )
    raw_labels = np.array([0] * 200 + [1] * 200 + [2] * 200 + [0])
    fake = SimpleNamespace(
        data=raw_texts,
        target=raw_labels,
        target_names=["comp.graphics", "rec.sport.baseball", "sci.space"],
    )
    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", lambda **_: fake)

    bundle = dataset_module.load_20newsgroups()

    assert len(bundle.texts) == 600
    assert len(bundle.source_doc_ids) == 600
    assert all(is_safe_text(text) for text in bundle.texts)


def test_validate_dataset_accepts_valid_input() -> None:
    validate_dataset(["one", "two"], [0, 1], minimum_documents=2)
