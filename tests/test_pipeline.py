import json
from pathlib import Path

import numpy as np
import pytest

import document_system.build_stages as build_stages_module
import document_system.pipeline as pipeline_module
from document_system.dataset import DatasetBundle
from document_system.pipeline import BuildConfig, build_from_dataset, build_project
from document_system.privacy import (
    SNIPPET_LIMIT,
    PrivacyReport,
    is_safe_text,
    sanitize_document,
)


def test_dataset_bundle_requires_privacy_report() -> None:
    with pytest.raises(TypeError, match="privacy_report"):
        DatasetBundle(
            texts=("space orbit", "baseball game"),
            labels=np.array([0, 1], dtype=np.int32),
            target_names=("space", "baseball"),
            source_doc_ids=np.array([0, 1], dtype=np.int32),
        )


def test_small_pipeline_writes_consistent_report_and_runtime_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_texts = (
        "space shuttle orbit mission",
        "rocket moon planet space",
        "astronaut orbit spacecraft",
        "baseball pitcher inning game",
        "hitter baseball stadium team",
        "pitcher season baseball",
        "space telescope planet",
        "baseball game score",
        "rocket launch orbit",
        "team pitcher hitter",
    )
    texts = tuple(
        f"{text} {' '.join(['telemetry'] * 40)}"
        f"{' heldoutonly' if index >= 8 else ''}"
        for index, text in enumerate(base_texts)
    )
    source_doc_ids = np.arange(100, 110, dtype=np.int32)
    bundle = DatasetBundle(
        texts=texts,
        labels=np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
        source_doc_ids=source_doc_ids,
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in texts],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
            ("space", "baseball"),
        ),
    )
    config = BuildConfig(
        runtime_dir=tmp_path / "runtime",
        reports_dir=tmp_path / "reports",
        batch_size=2,
        epochs=2,
        search_queries=("space orbit",),
    )

    evaluated_document_ids = None
    evaluate_classifier = build_stages_module.evaluate_classifier

    def record_document_ids(*args, **kwargs):
        nonlocal evaluated_document_ids
        evaluated_document_ids = np.asarray(kwargs["document_ids"])
        return evaluate_classifier(*args, **kwargs)

    monkeypatch.setattr(build_stages_module, "evaluate_classifier", record_document_ids)
    monkeypatch.setattr(build_stages_module, "train_test_split", fixed_split)

    report = build_from_dataset(bundle, config)

    assert report.document_count == 10
    assert report.train_count + report.test_count == 10
    assert report.classification_vocabulary_size < report.search_vocabulary_size
    assert report.classification_validation_passed is True
    assert report.search_validation_passed is True
    validation = json.loads(
        (config.reports_dir / "tfidf_validation.json").read_text(encoding="utf-8")
    )
    assert validation["classification"]["max_absolute_error"] <= 1e-6
    assert validation["classification"]["fit_scope"] == "train_split"
    assert validation["classification"]["fit_document_count"] == 8
    assert validation["search"]["fit_scope"] == "full_corpus"
    assert validation["search"]["fit_document_count"] == 10
    assert (config.reports_dir / "confusion_matrix.png").stat().st_size > 0
    assert (config.runtime_dir / "matrix.npz").is_file()
    assert evaluated_document_ids is not None
    assert set(evaluated_document_ids) <= set(source_doc_ids)

    metadata = json.loads(
        (config.runtime_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert "texts" not in metadata
    assert metadata["document_ids"] == source_doc_ids.tolist()
    assert all(is_safe_text(text) for text in metadata["snippets"])
    assert all(len(text) <= SNIPPET_LIMIT for text in metadata["snippets"])
    assert all(text not in metadata["snippets"] for text in texts)
    assert "heldoutonly" in metadata["feature_names"]
    assert metadata["fit_document_count"] == 10
    assert all("heldoutonly" not in snippet for snippet in metadata["snippets"])

    privacy_report = json.loads(
        (config.reports_dir / "privacy_report.json").read_text(encoding="utf-8")
    )
    assert privacy_report["documents_input"] == 10
    assert privacy_report["documents_retained"] == 10
    assert privacy_report["documents_dropped_after_sanitization"] == 0
    assert privacy_report["retention_rate"] == 1.0
    assert privacy_report["category_counts"] == {
        "baseball": {"raw": 5, "retained": 5, "excluded": 0},
        "space": {"raw": 5, "retained": 5, "excluded": 0},
    }
    assert privacy_report["redactions"] == {
        "email": 0,
        "phone": 0,
        "url": 0,
        "ipv4": 0,
        "ipv6": 0,
    }
    assert "residual_privacy_check" not in privacy_report

    search_examples = json.loads(
        (config.reports_dir / "search_examples.json").read_text(encoding="utf-8")
    )
    assert all(
        is_safe_text(result["text_snippet"])
        for example in search_examples
        for result in example["results"]
    )
    misclassifications = json.loads(
        (config.reports_dir / "misclassifications.json").read_text(encoding="utf-8")
    )
    assert all(
        is_safe_text(item["text_snippet"]) for item in misclassifications
    )


def test_build_project_accepts_sanitized_full_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    target_names = tuple(f"category.{index}" for index in range(20))
    labels = np.repeat(np.arange(20, dtype=np.int32), 25)
    category_texts = (
        "image pixel graphics",
        "baseball pitcher game",
        "space rocket orbit",
    ) + tuple(f"category {index} topic" for index in range(3, 20))
    texts = tuple(
        f"{category_texts[label]} document {index}"
        for index, label in enumerate(labels)
    )
    bundle = DatasetBundle(
        texts=texts,
        labels=labels,
        target_names=target_names,
        source_doc_ids=np.arange(len(texts), dtype=np.int32),
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in texts],
            labels,
            target_names,
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_20newsgroups", lambda: bundle)

    report = build_project(
        BuildConfig(
            runtime_dir=tmp_path / "runtime",
            reports_dir=tmp_path / "reports",
            batch_size=32,
            epochs=1,
        )
    )

    assert report.document_count == 500
    assert report.category_count == 20


def fixed_split(*_args, **_kwargs):
    return np.arange(8), np.arange(8, 10)
