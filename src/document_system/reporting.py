"""Persistence for generated build reports."""

from __future__ import annotations

import json
from pathlib import Path

from .build_stages import ClassificationStageResult, SearchStageResult
from .classification.visualization import save_confusion_matrix
from .constants import DEFAULT_TEST_SIZE
from .dataset import DatasetBundle
from .validation import ValidationResult

MAX_REPORTED_MISCLASSIFICATIONS = 20


def write_build_reports(
    reports_dir: Path,
    bundle: DatasetBundle,
    classification: ClassificationStageResult,
    search: SearchStageResult,
    *,
    epochs: int,
    random_state: int,
) -> None:
    """Persist build metrics, examples, validation, and visualization outputs."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    privacy_report = bundle.privacy_report
    if privacy_report.retained_document_count != len(bundle.texts):
        raise ValueError("privacy report retained count must match the dataset")
    _write_json(
        reports_dir / "dataset_sanitization_report.json",
        privacy_report.to_dict(),
    )
    _write_json(
        reports_dir / "tfidf_sklearn_validation.json",
        {
            "classification": _validation_payload(
                classification.validation,
                fit_scope="train_split",
                fit_document_count=classification.train_count,
            ),
            "search": _validation_payload(
                search.validation,
                fit_scope="full_corpus",
                fit_document_count=len(bundle.texts),
            ),
        },
    )
    matrix_stats = search.matrix.memory_stats()
    matrix_stats.update(
        {
            "document_count": len(bundle.texts),
            "category_count": len(bundle.target_names),
            "fit_scope": "full_corpus",
            "fit_document_count": len(bundle.texts),
            "search_vocabulary_size": search.vocabulary_size,
            "vocabulary_size": search.vocabulary_size,
            "representation": "NumPy CSR-like data/indices/indptr",
        }
    )
    _write_json(reports_dir / "search_index_statistics.json", matrix_stats)
    metrics = classification.report.metrics_dict()
    metrics.update(
        {
            "document_count": len(bundle.texts),
            "category_count": len(bundle.target_names),
            "train_count": classification.train_count,
            "test_count": classification.test_count,
            "test_size": DEFAULT_TEST_SIZE,
            "stratified": True,
            "epochs": epochs,
            "random_state": random_state,
            "classification_vocabulary_size": classification.vocabulary_size,
        }
    )
    _write_json(reports_dir / "classification_metrics.json", metrics)
    stage_payload = dict(classification.stage_example)
    stage_payload.update(
        {
            "fit_scope": "train_split",
            "fit_document_count": classification.train_count,
        }
    )
    _write_json(
        reports_dir / "tfidf_transformation_example.json",
        stage_payload,
    )
    _write_json(
        reports_dir / "classification_error_examples.json",
        classification.report.misclassifications[:MAX_REPORTED_MISCLASSIFICATIONS],
    )
    _write_json(
        reports_dir / "search_result_examples.json",
        search.search_examples,
    )
    save_confusion_matrix(
        classification.report,
        bundle.target_names,
        reports_dir / "classification_confusion_matrix.png",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validation_payload(
    validation: ValidationResult,
    *,
    fit_scope: str,
    fit_document_count: int,
) -> dict[str, object]:
    payload = validation.to_dict()
    payload.update(
        {"fit_scope": fit_scope, "fit_document_count": fit_document_count}
    )
    return payload
