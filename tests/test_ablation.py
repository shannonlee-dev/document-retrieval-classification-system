import pytest

from document_system.ablation import (
    AblationVariant,
    evaluate_ranked_labels,
    summarize_variants,
)


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


def test_summarize_variants_reports_mean_and_sample_standard_deviation() -> None:
    """Catch an ablation report that hides variation across random seeds."""

    variants = [
        AblationVariant("default_stop_words", 2, 100, 0.8, 0.7, 0.6, 0.5),
        AblationVariant("default_stop_words", 2, 120, 1.0, 0.9, 0.8, 0.7),
    ]

    summary = summarize_variants(variants)

    assert summary == {
        "runs": 2,
        "stop_word_count": 2,
        "vocabulary_size": {"mean": 110.0, "std": pytest.approx(14.1421356237)},
        "accuracy": {"mean": 0.9, "std": pytest.approx(0.1414213562)},
        "macro_f1": {"mean": 0.8, "std": pytest.approx(0.1414213562)},
        "precision_at_10": {"mean": 0.7, "std": pytest.approx(0.1414213562)},
        "map_at_10": {"mean": 0.6, "std": pytest.approx(0.1414213562)},
    }
