"""Independent classification and full-corpus search build stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .classification.evaluation import ClassificationReport, evaluate_classifier
from .classification.model import train_linear_svm
from .constants import DEFAULT_TEST_SIZE, DEFAULT_TOP_K
from .dataset import DatasetBundle
from .preprocessing import EnglishPreprocessor
from .search import DocumentSearch
from .sparse_matrix import SparseMatrix
from .tfidf import NumpyTfidfVectorizer
from .validation import ValidationResult, stage_example, validate_against_sklearn


@dataclass(frozen=True)
class ClassificationStageResult:
    report: ClassificationReport
    validation: ValidationResult
    stage_example: dict[str, object]
    train_count: int
    test_count: int
    vocabulary_size: int


@dataclass(frozen=True)
class SearchStageResult:
    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    validation: ValidationResult
    search_examples: list[dict[str, object]]
    vocabulary_size: int


def run_classification_stage(
    bundle: DatasetBundle,
    snippets: Sequence[str],
    *,
    batch_size: int,
    epochs: int,
    random_state: int,
) -> ClassificationStageResult:
    """Fit and evaluate classification using only the train split vocabulary."""

    row_ids = np.arange(len(bundle.texts))
    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=DEFAULT_TEST_SIZE,
        stratify=bundle.labels,
        random_state=random_state,
    )
    train_texts = [bundle.texts[int(row_id)] for row_id in train_ids]
    test_texts = [bundle.texts[int(row_id)] for row_id in test_ids]
    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor())
    stages = vectorizer.fit_transform_with_stages(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    validation = validate_against_sklearn(train_texts, vectorizer, stages.tfidf)
    if not validation.passed:
        raise RuntimeError(
            "classification TF-IDF validation failed: "
            f"{validation.max_absolute_error}"
        )
    model = train_linear_svm(
        stages.tfidf,
        bundle.labels[train_ids],
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state,
    )
    report = evaluate_classifier(
        model,
        test_matrix,
        bundle.labels[test_ids],
        [snippets[int(row_id)] for row_id in test_ids],
        bundle.target_names,
        batch_size=batch_size,
        document_ids=bundle.source_doc_ids[test_ids],
    )
    return ClassificationStageResult(
        report=report,
        validation=validation,
        stage_example=stage_example(stages, vectorizer),
        train_count=len(train_ids),
        test_count=len(test_ids),
        vocabulary_size=len(vectorizer.vocabulary_),
    )


def run_search_stage(
    bundle: DatasetBundle,
    snippets: Sequence[str],
    queries: Sequence[str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> SearchStageResult:
    """Fit the reusable search index from the complete corpus."""

    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor())
    matrix = vectorizer.fit_transform(bundle.texts)
    validation = validate_against_sklearn(bundle.texts, vectorizer, matrix)
    if not validation.passed:
        raise RuntimeError(
            f"search TF-IDF validation failed: {validation.max_absolute_error}"
        )
    searcher = DocumentSearch(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=snippets,
        labels=bundle.labels,
        target_names=bundle.target_names,
        document_ids=bundle.source_doc_ids,
    )
    examples = [
        {
            "query": query,
            "results": [
                result.to_dict()
                for result in searcher.search(query, min(top_k, len(bundle.texts)))
            ],
        }
        for query in queries
    ]
    return SearchStageResult(
        vectorizer=vectorizer,
        matrix=matrix,
        validation=validation,
        search_examples=examples,
        vocabulary_size=len(vectorizer.vocabulary_),
    )
