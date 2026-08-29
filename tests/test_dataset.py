from types import SimpleNamespace

import numpy as np
import pytest

import document_system.dataset as dataset_module
from document_system.dataset import _validate_dataset
from document_system.privacy import is_safe_text


def test_validate_dataset_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        _validate_dataset(["one", "two"], [0], minimum_documents=2)


def test_validate_dataset_requires_two_labels() -> None:
    with pytest.raises(ValueError, match="two labels"):
        _validate_dataset(["one", "two"], [0, 0], minimum_documents=2)


def test_validate_dataset_rejects_missing_or_non_string_text() -> None:
    with pytest.raises(ValueError, match="strings"):
        _validate_dataset(["one", None], [0, 1], minimum_documents=2)  # type: ignore[list-item]


def test_validate_dataset_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        _validate_dataset(["space", "   "], [0, 1], minimum_documents=2)


def test_validate_dataset_checks_source_document_ids() -> None:
    with pytest.raises(ValueError, match="document IDs"):
        _validate_dataset(
            ["space", "baseball"],
            [0, 1],
            source_doc_ids=[7],
            minimum_documents=2,
        )


def test_loader_returns_only_safe_nonblank_documents(monkeypatch) -> None:
    raw_texts = (
        ["Alice alice@example.com image pixel"] * 200
        + ["Bob 010-1234-5678 baseball pitcher"] * 200
        + ["Mission control discusses rocket orbit telemetry"] * 200
        + ["Alice alice@example.com patient treatment"]
        + ["alice@example.com"]
    )
    raw_labels = np.array([0] * 200 + [1] * 200 + [2] * 200 + [0, 1])
    fake = SimpleNamespace(
        data=raw_texts,
        target=raw_labels,
        target_names=["comp.graphics", "rec.sport.baseball", "sci.space"],
    )
    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", lambda **_: fake)

    bundle = dataset_module.load_20newsgroups()

    assert len(bundle.texts) == 601
    assert len(bundle.source_doc_ids) == 601
    assert all(is_safe_text(text) for text in bundle.texts)
    assert bundle.privacy_report.raw_document_count == 602
    assert bundle.privacy_report.retained_document_count == 601
    assert bundle.privacy_report.dropped_after_sanitization == 1
    assert bundle.privacy_report.redactions == {
        "email": 202,
        "phone": 200,
        "url": 0,
        "ipv4": 0,
        "ipv6": 0,
    }
    assert bundle.privacy_report.retention_rate == pytest.approx(601 / 602)
    assert bundle.privacy_report.category_counts == {
        "comp.graphics": {"raw": 201, "retained": 201, "excluded": 0},
        "rec.sport.baseball": {"raw": 201, "retained": 200, "excluded": 1},
        "sci.space": {"raw": 200, "retained": 200, "excluded": 0},
    }


def test_validate_dataset_accepts_valid_input() -> None:
    _validate_dataset(["one", "two"], [0, 1], minimum_documents=2)
