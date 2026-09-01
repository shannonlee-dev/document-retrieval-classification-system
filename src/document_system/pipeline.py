"""End-to-end build pipeline for reports and the reusable search index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import save_search_artifacts
from .build_stages import run_classification_stage, run_search_stage
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REPORTS_DIR,
    DEFAULT_RUNTIME_DIR,
    MINIMUM_DOCUMENTS,
)
from .dataset import (
    MINIMUM_CATEGORY_COUNT,
    DatasetBundle,
    load_20newsgroups,
    validate_full_20_newsgroups,
)
from .privacy import make_safe_snippet
from .reporting import write_build_reports

DEFAULT_SEARCH_QUERIES = (
    "space shuttle orbit",
    "baseball pitcher season",
    "computer graphics image",
)
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

    write_build_reports(
        config.reports_dir,
        bundle,
        classification,
        search,
        epochs=config.epochs,
        random_state=config.random_state,
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
