"""End-to-end build pipeline for reports and the reusable search index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import save_search_artifacts
from .build_stages import run_classification_stage, run_search_stage
from .classification import save_confusion_matrix
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REPORTS_DIR,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_TEST_SIZE,
    MINIMUM_DOCUMENTS,
)
from .dataset import (
    MINIMUM_CATEGORY_COUNT,
    DatasetBundle,
    load_20newsgroups,
    validate_full_20_newsgroups,
)
from .privacy import make_safe_snippet
from .validation import ValidationResult

DEFAULT_SEARCH_QUERIES = (
    "space shuttle orbit",
    "baseball pitcher season",
    "computer graphics image",
)
MAX_REPORTED_MISCLASSIFICATIONS = 20


@dataclass(frozen=True)
class BuildConfig:
    runtime_dir: Path = DEFAULT_RUNTIME_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    random_state: int = DEFAULT_RANDOM_STATE
    search_queries: tuple[str, ...] = DEFAULT_SEARCH_QUERIES


@dataclass(frozen=True)
class BuildReport:
    document_count: int
    category_count: int
    train_count: int
    test_count: int
    classification_vocabulary_size: int
    search_vocabulary_size: int
    classification_validation_passed: bool
    search_validation_passed: bool
    classification_max_absolute_error: float
    search_max_absolute_error: float
    accuracy: float
    macro_f1: float


def build_project(config: BuildConfig | None = None) -> BuildReport:
    bundle = load_20newsgroups()
    validate_full_20_newsgroups(bundle)
    if (
        len(bundle.texts) < MINIMUM_DOCUMENTS
        or len(bundle.target_names) < MINIMUM_CATEGORY_COUNT
    ):
        raise ValueError(
            "the full build requires at least 500 documents and two categories"
        )
    return build_from_dataset(bundle, config or BuildConfig())


def build_from_dataset(bundle: DatasetBundle, config: BuildConfig) -> BuildReport:
    if config.batch_size < 1 or config.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    snippets = tuple(make_safe_snippet(text) for text in bundle.texts)
    classification = run_classification_stage(
        bundle,
        snippets,
        batch_size=config.batch_size,
        epochs=config.epochs,
        random_state=config.random_state,
    )
    search = run_search_stage(
        bundle,
        snippets,
        config.search_queries,
    )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    privacy_report = bundle.privacy_report
    if privacy_report.retained_document_count != len(bundle.texts):
        raise ValueError("privacy report retained count must match the dataset")
    _write_json(config.reports_dir / "privacy_report.json", privacy_report.to_dict())
    _write_json(
        config.reports_dir / "tfidf_validation.json",
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
    _write_json(config.reports_dir / "matrix_stats.json", matrix_stats)
    metrics = classification.report.metrics_dict()
    metrics.update(
        {
            "document_count": len(bundle.texts),
            "category_count": len(bundle.target_names),
            "train_count": classification.train_count,
            "test_count": classification.test_count,
            "test_size": DEFAULT_TEST_SIZE,
            "stratified": True,
            "epochs": config.epochs,
            "random_state": config.random_state,
            "classification_vocabulary_size": classification.vocabulary_size,
        }
    )
    _write_json(config.reports_dir / "metrics.json", metrics)
    stage_payload = dict(classification.stage_example)
    stage_payload.update(
        {
            "fit_scope": "train_split",
            "fit_document_count": classification.train_count,
        }
    )
    _write_json(
        config.reports_dir / "stage_example.json",
        stage_payload,
    )
    _write_json(
        config.reports_dir / "misclassifications.json",
        classification.report.misclassifications[:MAX_REPORTED_MISCLASSIFICATIONS],
    )
    _write_json(config.reports_dir / "search_examples.json", search.search_examples)
    save_confusion_matrix(
        classification.report,
        bundle.target_names,
        config.reports_dir / "confusion_matrix.png",
    )
    save_search_artifacts(
        config.runtime_dir,
        search.vectorizer,
        search.matrix,
        snippets,
        bundle.labels,
        bundle.target_names,
        bundle.source_doc_ids,
    )
    return BuildReport(
        document_count=len(bundle.texts),
        category_count=len(bundle.target_names),
        train_count=classification.train_count,
        test_count=classification.test_count,
        classification_vocabulary_size=classification.vocabulary_size,
        search_vocabulary_size=search.vocabulary_size,
        classification_validation_passed=classification.validation.passed,
        search_validation_passed=search.validation.passed,
        classification_max_absolute_error=classification.validation.max_absolute_error,
        search_max_absolute_error=search.validation.max_absolute_error,
        accuracy=classification.report.accuracy,
        macro_f1=classification.report.macro_f1,
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
