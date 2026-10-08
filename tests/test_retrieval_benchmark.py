import csv
import json

import numpy as np
import pytest

from document_system.dataset import DatasetBundle
from document_system.experiments.retrieval_benchmark import (
    latency_statistics,
    run_benchmark,
)
from document_system.pipeline import BuildConfig, build_from_dataset
from document_system.privacy import PrivacyReport, sanitize_document


def fixture_bundle():
    texts = tuple(
        ("space orbit rocket" if i % 2 == 0 else "baseball pitcher game")
        + (" orbit orbit" if i % 3 == 0 else " pitcher")
        for i in range(40)
    )
    labels = np.array([i % 2 for i in range(40)])
    return DatasetBundle(
        texts,
        labels,
        ("space", "sport"),
        np.arange(100, 140),
        PrivacyReport.from_sanitization_results(
            [sanitize_document(t) for t in texts], labels, ("space", "sport")
        ),
    )


def test_latency_statistics_use_real_samples_and_empty_is_not_measured():
    result = latency_statistics([0.001, 0.002, 0.003, 0.004])
    assert result == {
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "samples": 4,
    }
    assert latency_statistics([]) == {
        "mean_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "samples": 0,
    }


def test_fixture_benchmark_has_shared_split_real_metrics_and_reports(tmp_path):
    bundle = fixture_bundle()
    runtime, reports = tmp_path / "runtime", tmp_path / "reports"
    build_from_dataset(
        bundle,
        BuildConfig(
            runtime_dir=runtime,
            reports_dir=reports,
            batch_size=8,
            epochs=1,
            search_queries=("space",),
        ),
    )
    result = run_benchmark(
        bundle,
        runtime,
        reports,
        query_count=4,
        repeats=2,
        epochs=1,
        batch_size=8,
        sizes=(10, 40),
        queries=("space orbit", "baseball pitcher", "rocket"),
    )
    train, test = (
        result["split"]["train_document_ids"],
        result["split"]["test_document_ids"],
    )
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(bundle.source_doc_ids)
    assert set(result["split"]["query_document_ids"]) <= set(test)
    assert len(result["tf_variants"]) == 3
    assert all(0 <= row["accuracy"] <= 1 for row in result["tf_variants"])
    assert all(row["search_consistency"] for row in result["inverted_index"])
    assert len(result["top5_comparisons"]) == 3
    assert result["validation"]["raw"]["passed"]
    assert result["validation"]["log"]["passed"]
    assert result["validation"]["double"]["reference"] == "hand calculation"
    assert result["metadata"]["query_count"] == 4
    assert len(result["tf_variants"][0]["retrieval_queries"]) == 4
    assert len(result["bm25_comparison"][1]["retrieval_queries"]) == 4
    assert "quality_difference_cases" in result
    assert (
        sum(
            result["quality_difference_cases"][key]
            for key in ("tfidf_better_count", "bm25_better_count", "equal_count")
        )
        == 4
    )
    for name in [
        "tf_variants_comparison.csv",
        "bm25_comparison.csv",
        "inverted_index_benchmark.csv",
    ]:
        with (reports / name).open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows
    assert json.loads((reports / "retrieval_benchmark.json").read_text())["metadata"][
        "corpus_fingerprint"
    ]
    assert (reports / "experiment_summary.md").exists()
    again = run_benchmark(
        bundle,
        runtime,
        tmp_path / "again",
        query_count=4,
        repeats=1,
        epochs=1,
        batch_size=8,
        sizes=(10,),
        queries=("space",),
    )
    assert result["split"] == again["split"]
    for first, second in zip(result["tf_variants"], again["tf_variants"]):
        for key in ["accuracy", "macro_f1", "precision_at_10", "map_at_10"]:
            assert first[key] == second[key]


def test_benchmark_rejects_dataset_runtime_mismatch(tmp_path):
    bundle = fixture_bundle()
    runtime = tmp_path / "runtime"
    build_from_dataset(
        bundle,
        BuildConfig(
            runtime_dir=runtime,
            reports_dir=tmp_path / "reports",
            epochs=1,
            search_queries=("space",),
        ),
    )
    changed = DatasetBundle(
        tuple(t + " extra" for t in bundle.texts),
        bundle.labels,
        bundle.target_names,
        bundle.source_doc_ids,
        bundle.privacy_report,
    )
    with pytest.raises(ValueError, match="corpus"):
        run_benchmark(changed, runtime, tmp_path / "wrong", query_count=4, repeats=1)


def test_zero_score_queries_have_fixed_k_metrics_and_external_id_ties():
    from document_system.experiments.metrics import evaluate_ranked_labels, rank_scores

    assert rank_scores({}, np.array([90, 7, 3]), 2) == [2, 1]
    assert rank_scores({0: 0.2, 1: 0.2}, np.array([90, 7, 3]), 2) == [1, 0]
    result = evaluate_ranked_labels(
        query_labels=[0], ranked_labels=[[1, 0]], corpus_labels=[0, 1, 0], k=2
    )
    assert result.precision_at_k == 0.5
    assert result.map_at_k == 0.25


def test_tf_experiment_never_fits_test_vocabulary_and_keeps_oov_queries():
    from document_system.experiments.tf_variants import run_tf_variants

    bundle = fixture_bundle()
    changed_texts = list(bundle.texts)
    changed_texts[-2] = "heldoutuniquetermzz"
    changed_texts[-1] = "heldoutuniquetermzz"
    bundle = DatasetBundle(
        tuple(changed_texts),
        bundle.labels,
        bundle.target_names,
        bundle.source_doc_ids,
        bundle.privacy_report,
    )
    rows, searchers, validation, counts = run_tf_variants(
        bundle,
        np.arange(38),
        np.array([38, 39]),
        np.array([38, 39]),
        k=10,
        epochs=1,
        batch_size=8,
        random_state=42,
    )
    assert validation["count_statistics"]["empty_query_count"] == 2
    for searcher in searchers.values():
        assert "heldoutuniquetermzz" not in searcher.vectorizer.vocabulary_
        assert searcher.matrix.shape[0] == 38
    assert len(rows) == 3
    assert counts.shape == searchers["raw"].matrix.shape


def test_evaluation_can_align_ties_to_external_ids():
    from document_system.experiments.metrics import evaluate_label_retrieval
    from document_system.tfidf import NumpyTfidfVectorizer

    vectorizer = NumpyTfidfVectorizer()
    corpus = vectorizer.fit_transform(["space"] * 20)
    queries = vectorizer.transform(["unknownzz"])
    labels = np.array([0] * 10 + [1] * 10)
    legacy = evaluate_label_retrieval(queries, [0], corpus, labels)
    aligned = evaluate_label_retrieval(
        queries, [0], corpus, labels, corpus_document_ids=np.arange(20)[::-1]
    )
    assert legacy.precision_at_k == 1
    assert aligned.precision_at_k == 0
