import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import document_system.dataset as dataset_module
from document_system.dataset import DatasetBundle, _validate_dataset
from document_system.privacy import PrivacyReport, is_safe_text, sanitize_document


def make_full_category_fake() -> SimpleNamespace:
    target_names = [f"category.{index}" for index in range(20)]
    labels = np.repeat(np.arange(20, dtype=np.int32), 25)
    texts = [f"category token document {index}" for index in range(labels.size)]
    return SimpleNamespace(data=texts, target=labels, target_names=target_names)


@pytest.fixture
def full_bundle() -> DatasetBundle:
    fake = make_full_category_fake()
    return DatasetBundle(
        texts=tuple(fake.data),
        labels=fake.target.copy(),
        target_names=tuple(fake.target_names),
        source_doc_ids=np.arange(len(fake.data), dtype=np.int32),
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in fake.data],
            fake.target,
            tuple(fake.target_names),
        ),
    )


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


def test_loader_requests_all_20_categories(monkeypatch) -> None:
    captured_options = None

    def fake_fetch(**options):
        nonlocal captured_options
        captured_options = options
        return make_full_category_fake()

    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", fake_fetch)

    bundle = dataset_module.load_20newsgroups()

    assert captured_options is not None
    assert "categories" not in captured_options
    assert captured_options["subset"] == "all"
    assert len(bundle.target_names) == 20
    assert set(bundle.labels.tolist()) == set(range(20))


def test_full_dataset_contract_rejects_missing_category(full_bundle) -> None:
    invalid = dataclasses.replace(
        full_bundle,
        labels=np.where(full_bundle.labels == 19, 18, full_bundle.labels),
    )

    with pytest.raises(ValueError, match="all 20 categories"):
        dataset_module.validate_full_20_newsgroups(invalid)


def test_full_dataset_contract_rejects_out_of_range_label(full_bundle) -> None:
    labels = full_bundle.labels.copy()
    labels[0] = 20

    with pytest.raises(ValueError, match="class IDs"):
        dataset_module.validate_full_20_newsgroups(
            dataclasses.replace(full_bundle, labels=labels)
        )


def test_loader_returns_only_safe_nonblank_documents(monkeypatch) -> None:
    raw_texts = (
        ["Alice alice@example.com image pixel"] * 200
        + ["Bob 010-1234-5678 baseball pitcher"] * 200
        + ["Mission control discusses rocket orbit telemetry"] * 200
        + ["Alice alice@example.com patient treatment"]
        + ["alice@example.com"]
    )
    raw_texts.extend(
        text
        for category in range(3, 20)
        for text in [f"category {category} topic document"] * 25
    )
    raw_labels = np.array(
        [0] * 200
        + [1] * 200
        + [2] * 200
        + [0, 1]
        + [category for category in range(3, 20) for _ in range(25)]
    )
    fake = SimpleNamespace(
        data=raw_texts,
        target=raw_labels,
        target_names=["comp.graphics", "rec.sport.baseball", "sci.space"]
        + [f"category.{index}" for index in range(3, 20)],
    )
    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", lambda **_: fake)

    bundle = dataset_module.load_20newsgroups()

    assert len(bundle.texts) == 1026
    assert len(bundle.source_doc_ids) == 1026
    assert all(is_safe_text(text) for text in bundle.texts)
    assert bundle.privacy_report.raw_document_count == 1027
    assert bundle.privacy_report.retained_document_count == 1026
    assert bundle.privacy_report.dropped_after_sanitization == 1
    assert bundle.privacy_report.redactions == {
        "email": 202,
        "phone": 200,
        "url": 0,
        "ipv4": 0,
        "ipv6": 0,
    }
    assert bundle.privacy_report.retention_rate == pytest.approx(1026 / 1027)
    assert {
        name: bundle.privacy_report.category_counts[name]
        for name in ("comp.graphics", "rec.sport.baseball", "sci.space")
    } == {
        "comp.graphics": {"raw": 201, "retained": 201, "excluded": 0},
        "rec.sport.baseball": {"raw": 201, "retained": 200, "excluded": 1},
        "sci.space": {"raw": 200, "retained": 200, "excluded": 0},
    }
    assert set(bundle.privacy_report.category_counts) == set(fake.target_names)
    assert len(bundle.privacy_report.category_counts) == 20


def test_validate_dataset_accepts_valid_input() -> None:
    _validate_dataset(["one", "two"], [0, 1], minimum_documents=2)
