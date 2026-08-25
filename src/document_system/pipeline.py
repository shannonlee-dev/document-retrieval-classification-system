"""End-to-end build pipeline for reports and the reusable search index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .artifacts import save_search_artifacts
from .classification import (
    evaluate_classifier,
    save_confusion_matrix,
    train_linear_svm,
)
from .dataset import DatasetBundle, load_20newsgroups
from .preprocessing import EnglishPreprocessor
from .search import DocumentSearch
from .tfidf import NumpyTfidfVectorizer
from .validation import stage_example, validate_against_sklearn


@dataclass(frozen=True)
class BuildConfig:
    runtime_dir: Path = Path("artifacts/runtime")
    reports_dir: Path = Path("artifacts/reports")
    batch_size: int = 128
    epochs: int = 6
    random_state: int = 42
    search_queries: tuple[str, ...] = (
        "space shuttle orbit",
        "baseball pitcher season",
        "computer graphics image",
    )


@dataclass(frozen=True)
class BuildReport:
    document_count: int
    category_count: int
    train_count: int
    test_count: int
    vocabulary_size: int
    validation_passed: bool
    max_absolute_error: float
    accuracy: float
    macro_f1: float


def build_project(config: BuildConfig | None = None) -> BuildReport:
    bundle = load_20newsgroups()
    if len(bundle.texts) < 500 or len(bundle.target_names) < 2:
        raise ValueError("the full build requires at least 500 documents and two categories")
    return build_from_dataset(bundle, config or BuildConfig())


def build_from_dataset(bundle: DatasetBundle, config: BuildConfig) -> BuildReport:
    if config.batch_size < 1 or config.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    document_ids = np.arange(len(bundle.texts))
    train_ids, test_ids = train_test_split(
        document_ids,
        test_size=0.2,
        stratify=bundle.labels,
        random_state=config.random_state,
    )
    train_texts = [bundle.texts[int(index)] for index in train_ids]
    test_texts = [bundle.texts[int(index)] for index in test_ids]
    train_labels = bundle.labels[train_ids]
    test_labels = bundle.labels[test_ids]

    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor())
    stages = vectorizer.fit_transform_with_stages(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    validation = validate_against_sklearn(train_texts, vectorizer, stages.tfidf)
    if not validation.passed:
        raise RuntimeError(
            f"TF-IDF validation failed: max error {validation.max_absolute_error}"
        )

    model = train_linear_svm(
        stages.tfidf,
        train_labels,
        batch_size=config.batch_size,
        epochs=config.epochs,
        random_state=config.random_state,
    )
    classification = evaluate_classifier(
        model,
        test_matrix,
        test_labels,
        test_texts,
        bundle.target_names,
        batch_size=config.batch_size,
        document_ids=test_ids,
    )

    full_matrix = vectorizer.transform(bundle.texts)
    searcher = DocumentSearch(
        vectorizer=vectorizer,
        matrix=full_matrix,
        snippets=bundle.texts,
        labels=bundle.labels,
        target_names=bundle.target_names,
        document_ids=bundle.source_doc_ids,
    )
    search_examples = [
        {
            "query": query,
            "results": [
                result.to_dict()
                for result in searcher.search(query, topk=min(5, len(bundle.texts)))
            ],
        }
        for query in config.search_queries
    ]

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.reports_dir / "tfidf_validation.json", validation.to_dict())
    matrix_stats = full_matrix.memory_stats()
    matrix_stats.update(
        {
            "document_count": len(bundle.texts),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "representation": "NumPy CSR-like data/indices/indptr",
        }
    )
    _write_json(config.reports_dir / "matrix_stats.json", matrix_stats)
    metrics = classification.metrics_dict()
    metrics.update(
        {
            "document_count": len(bundle.texts),
            "category_count": len(bundle.target_names),
            "train_count": len(train_ids),
            "test_count": len(test_ids),
            "test_size": 0.2,
            "stratified": True,
            "epochs": config.epochs,
            "random_state": config.random_state,
        }
    )
    _write_json(config.reports_dir / "metrics.json", metrics)
    _write_json(
        config.reports_dir / "stage_example.json",
        stage_example(stages, vectorizer),
    )
    _write_json(
        config.reports_dir / "misclassifications.json",
        classification.misclassifications[:20],
    )
    _write_json(config.reports_dir / "search_examples.json", search_examples)
    save_confusion_matrix(
        classification,
        bundle.target_names,
        config.reports_dir / "confusion_matrix.png",
    )
    save_search_artifacts(
        config.runtime_dir,
        vectorizer,
        full_matrix,
        bundle.texts,
        bundle.labels,
        bundle.target_names,
        bundle.source_doc_ids,
    )
    return BuildReport(
        document_count=len(bundle.texts),
        category_count=len(bundle.target_names),
        train_count=len(train_ids),
        test_count=len(test_ids),
        vocabulary_size=len(vectorizer.vocabulary_),
        validation_passed=validation.passed,
        max_absolute_error=validation.max_absolute_error,
        accuracy=classification.accuracy,
        macro_f1=classification.macro_f1,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
