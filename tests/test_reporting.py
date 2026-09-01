import json
from pathlib import Path

import numpy as np

from document_system.build_stages import (
    ClassificationStageResult,
    SearchStageResult,
)
from document_system.classification.evaluation import ClassificationReport
from document_system.dataset import DatasetBundle
from document_system.preprocessing import EnglishPreprocessor
from document_system.privacy import PrivacyReport, sanitize_document
from document_system.reporting import write_build_reports
from document_system.tfidf import NumpyTfidfVectorizer
from document_system.validation import ValidationResult


def test_write_build_reports_persists_all_report_outputs(tmp_path: Path) -> None:
    texts = ("space orbit", "baseball pitcher")
    labels = np.array([0, 1], dtype=np.int32)
    target_names = ("space", "baseball")
    bundle = DatasetBundle(
        texts=texts,
        labels=labels,
        target_names=target_names,
        source_doc_ids=np.array([10, 20], dtype=np.int32),
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in texts],
            labels,
            target_names,
        ),
    )
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    matrix = vectorizer.fit_transform(texts)
    validation = ValidationResult(
        shape=matrix.shape,
        max_absolute_error=0.0,
        mean_absolute_error=0.0,
        tolerance=1e-6,
        passed=True,
        settings={"norm": "l2"},
    )
    report = ClassificationReport(
        accuracy=1.0,
        macro_f1=1.0,
        confusion_matrix=np.eye(2, dtype=np.int32),
        predictions=labels.copy(),
        misclassifications=[],
        model_settings={
            "loss": "hinge",
            "batch_size": 2,
            "random_state": 42,
            "input_representation": "NumPy dense batches",
        },
    )
    classification = ClassificationStageResult(
        report=report,
        validation=validation,
        stage_example={"document_id": 0, "terms": []},
        train_count=1,
        test_count=1,
        vocabulary_size=4,
    )
    search = SearchStageResult(
        vectorizer=vectorizer,
        matrix=matrix,
        validation=validation,
        search_examples=[{"query": "space", "results": []}],
        vocabulary_size=4,
    )

    write_build_reports(
        tmp_path,
        bundle,
        classification,
        search,
        epochs=2,
        random_state=42,
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "classification_confusion_matrix.png",
        "classification_error_examples.json",
        "classification_metrics.json",
        "dataset_sanitization_report.json",
        "search_index_statistics.json",
        "search_result_examples.json",
        "tfidf_sklearn_validation.json",
        "tfidf_transformation_example.json",
    }
    assert (tmp_path / "classification_confusion_matrix.png").stat().st_size > 0
    metrics = json.loads(
        (tmp_path / "classification_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["epochs"] == 2
    assert metrics["random_state"] == 42
    validation_payload = json.loads(
        (tmp_path / "tfidf_sklearn_validation.json").read_text(encoding="utf-8")
    )
    assert validation_payload["classification"]["fit_scope"] == "train_split"
    assert validation_payload["search"]["fit_scope"] == "full_corpus"
