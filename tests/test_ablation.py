import pytest

from document_system.ablation import evaluate_ranked_labels


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
