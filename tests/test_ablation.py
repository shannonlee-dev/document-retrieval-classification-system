import numpy as np
import pytest

from document_system.ablation import evaluate_ranked_labels, run_stop_word_ablation
from document_system.dataset import DatasetBundle
from document_system.privacy import PrivacyReport, sanitize_document


def test_evaluate_ranked_labels_reports_precision_and_map_at_k() -> None:
    metrics = evaluate_ranked_labels(
        query_labels=[0, 1],
        ranked_labels=[[0, 1, 0], [0, 0, 1]],
        corpus_labels=[0, 0, 1],
        k=2,
    )

    assert metrics.precision_at_k == pytest.approx(0.25)
    assert metrics.map_at_k == pytest.approx(0.25)


def test_evaluate_ranked_labels_rejects_rankings_shorter_than_k() -> None:
    with pytest.raises(ValueError, match="at least k"):
        evaluate_ranked_labels(
            query_labels=[0],
            ranked_labels=[[0]],
            corpus_labels=[0],
            k=2,
        )


def test_stop_word_ablation_reports_full_20_category_dataset_description() -> None:
    texts = tuple(
        [f"alpha category document {index}" for index in range(10)]
        + [f"beta category document {index}" for index in range(10)]
    )
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    target_names = ("alpha", "beta")
    bundle = DatasetBundle(
        texts=texts,
        labels=labels,
        target_names=target_names,
        source_doc_ids=np.arange(20, dtype=np.int32),
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in texts], labels, target_names
        ),
    )

    report = run_stop_word_ablation(bundle)

    assert report["dataset"] == "20 Newsgroups: all 20 categories"
    assert report["split"]["train_documents"] == 16
    assert report["split"]["test_queries"] == 4
