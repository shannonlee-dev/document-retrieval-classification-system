import json
from pathlib import Path

import pytest

import document_system.experiments.ablation as ablation_module
import document_system.experiments.reporting as reporting_module
from document_system.experiments.ablation import (
    AblationVariant,
    run_stop_word_ablation,
    summarize_variants,
)
from document_system.experiments.metrics import evaluate_ranked_labels
from document_system.experiments.reporting import write_stop_word_ablation_report


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


def test_stop_word_ablation_reports_full_20_category_dataset_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = [
        AblationVariant("default_stop_words", 2, 100, 0.8, 0.7, 0.6, 0.5),
        AblationVariant("no_stop_words", 0, 120, 0.7, 0.6, 0.5, 0.4),
    ]
    monkeypatch.setattr(
        ablation_module,
        "_run_stop_word_ablation_seed",
        lambda _dataset, _seed: variants,
    )

    report = run_stop_word_ablation(object())  # type: ignore[arg-type]

    assert report["dataset"] == "20 Newsgroups: all 20 categories"
    assert report["split"]["seeds"] == list(range(10))


def test_write_stop_word_ablation_report_persists_runner_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = {"dataset": "small", "variants": []}
    monkeypatch.setattr(
        reporting_module,
        "run_stop_word_ablation",
        lambda: expected,
    )

    output = tmp_path / "ablation.json"
    result = write_stop_word_ablation_report(output)

    assert result == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
