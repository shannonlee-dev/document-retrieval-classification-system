"""Quality (held-out labels) and speed (short queries) measured separately."""

from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ..artifacts import load_retrievers
from ..bm25 import NumpyBM25Retriever
from ..constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_QUERIES,
    DEFAULT_TEST_SIZE,
    DEFAULT_TOP_K,
    MILLISECONDS_PER_SECOND,
    SEARCH_SCORE_TOLERANCE,
)
from ..dataset import DatasetBundle, dataset_fingerprint
from ..inverted_index import InvertedIndex
from ..privacy import make_safe_snippet
from ..search import DocumentSearch
from ..tfidf import NumpyTfidfVectorizer
from .ablation import run_stop_word_ablation, select_query_positions
from .metrics import (
    RETRIEVAL_TOP_K,
    evaluate_ranked_labels,
    query_metrics_record,
    rank_scores,
)
from .tf_variants import run_tf_variants

DEFAULT_BENCHMARK_QUERY_COUNT = 100
DEFAULT_BENCHMARK_REPEATS = 5
DEFAULT_BENCHMARK_SIZES = (1000, 5000, 10000)


def run_benchmark(
    bundle: DatasetBundle,
    runtime_dir: str | Path,
    reports_dir: str | Path,
    *,
    query_count: int = DEFAULT_BENCHMARK_QUERY_COUNT,
    repeats: int = DEFAULT_BENCHMARK_REPEATS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    sizes: Sequence[int] = DEFAULT_BENCHMARK_SIZES,
    queries: Sequence[str] = DEFAULT_SEARCH_QUERIES,
) -> dict[str, object]:
    """Run paired quality and speed experiments and persist reports for this corpus."""
    if (
        query_count < 1
        or repeats < 1
        or epochs < 1
        or batch_size < 1
        or any(size < 1 for size in sizes)
    ):
        raise ValueError(
            "query count, repeats, epochs, batch size and sizes must be positive"
        )
    fingerprint = dataset_fingerprint(bundle)
    start = time.perf_counter()
    runtime = load_retrievers(runtime_dir)
    load_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
    if runtime.corpus_fingerprint != fingerprint:
        raise ValueError(
            "benchmark corpus does not match runtime; run python -m document_system build"
        )
    train_ids, test_ids = train_test_split(
        np.arange(len(bundle.texts)),
        test_size=DEFAULT_TEST_SIZE,
        stratify=bundle.labels,
        random_state=random_state,
    )
    query_ids = test_ids[
        select_query_positions(bundle.labels[test_ids], query_count, random_state)
    ]
    k = min(RETRIEVAL_TOP_K, len(train_ids))
    print(
        f"E1/E2: {len(train_ids):,} train / {len(test_ids):,} test; {len(query_ids)} shared queries",
        flush=True,
    )
    tf_rows, searchers, validations, counts = run_tf_variants(
        bundle,
        train_ids,
        test_ids,
        query_ids,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    raw = searchers["raw"]
    train_texts = [bundle.texts[int(i)] for i in train_ids]
    lengths = np.array(
        [len(raw.vectorizer.preprocessor.tokenize(t)) for t in train_texts]
    )
    bm25 = NumpyBM25Retriever(
        raw.vectorizer,
        counts,
        raw.snippets,
        raw.labels,
        raw.target_names,
        raw.document_ids,
        lengths,
    )
    query_texts = [bundle.texts[int(i)] for i in query_ids]
    bm_rankings = []
    bm_query_records = []
    for query_id, query in zip(query_ids, query_texts, strict=True):
        terms = sorted(
            {
                raw.vectorizer.vocabulary_[t]
                for t in raw.vectorizer.preprocessor.tokenize(query)
                if t in raw.vectorizer.vocabulary_
            }
        )
        scores = bm25.inverted_index.score(
            np.array(terms, dtype=np.int32), np.ones(len(terms))
        )
        ranked_rows = rank_scores(scores, raw.document_ids, k)
        bm_rankings.append(raw.labels[ranked_rows].tolist())
        bm_query_records.append(
            query_metrics_record(
                bundle.source_doc_ids[query_id],
                bundle.labels[query_id],
                ranked_rows,
                raw.labels,
                raw.document_ids,
                k,
            )
        )
    bm_quality = evaluate_ranked_labels(
        query_labels=bundle.labels[query_ids],
        ranked_labels=bm_rankings,
        corpus_labels=raw.labels,
        k=k,
    )
    # 두 검색 방식의 지연 시간은 동일한 유효 평가 질의로 비교한다.
    latency_queries = [
        q
        for q in query_texts
        if any(
            t in raw.vectorizer.vocabulary_
            for t in raw.vectorizer.preprocessor.tokenize(q)
        )
    ]
    print("E3: TF-IDF/BM25 repeated held-out query latency", flush=True)
    raw_latency = _timed_search(
        lambda q, topk: raw.search(q, topk, strategy="inverted"),
        latency_queries,
        repeats,
        k,
    )
    bm_latency = _timed_search(bm25.search, latency_queries, repeats, k)
    algorithm_rows = [
        {
            "method": "tfidf",
            "tf_mode": "raw",
            "precision_at_10": tf_rows[0]["precision_at_10"],
            "map_at_10": tf_rows[0]["map_at_10"],
            **raw_latency,
            "retrieval_queries": tf_rows[0]["retrieval_queries"],
        },
        {
            "method": "bm25",
            "tf_mode": None,
            "precision_at_10": bm_quality.precision_at_k,
            "map_at_10": bm_quality.map_at_k,
            **bm_latency,
            "retrieval_queries": bm_query_records,
        },
    ]
    print(
        "E4: Scan/Inverted by corpus size (warm-up then repeated short queries)",
        flush=True,
    )
    speed_rows = _benchmark_inverted(bundle, sizes, queries, repeats)
    print("E5: paired stopword ablation, same query budget and seed", flush=True)
    ablation = run_stop_word_ablation(
        bundle,
        seeds=(random_state,),
        query_count=len(query_ids),
        epochs=epochs,
        batch_size=batch_size,
        external_id_ties=True,
    )
    comparisons = []
    for query in queries:
        if not any(
            term in runtime.tfidf["raw"].vectorizer.vocabulary_
            for term in runtime.tfidf["raw"].vectorizer.preprocessor.tokenize(query)
        ):
            comparisons.append(
                {
                    "query": query,
                    "status": "not measured",
                    "reason": "no query terms in demo vocabulary",
                    "tfidf": [],
                    "bm25": [],
                    "top5_overlap": None,
                }
            )
            continue
        raw_results = runtime.tfidf["raw"].search(
            query, min(DEFAULT_TOP_K, len(bundle.texts)), strategy="inverted"
        )
        bm_results = runtime.bm25.search(query, min(DEFAULT_TOP_K, len(bundle.texts)))
        lengths_by_id = dict(
            zip(
                runtime.bm25.documents.document_ids.tolist(),
                runtime.bm25.document_lengths.tolist(),
                strict=True,
            )
        )
        comparisons.append(
            {
                "query": query,
                "tfidf": [
                    {**r.to_dict(), "token_length": lengths_by_id[r.doc_id]}
                    for r in raw_results
                ],
                "bm25": [
                    {**r.to_dict(), "token_length": lengths_by_id[r.doc_id]}
                    for r in bm_results
                ],
                "top5_overlap": len(
                    {r.doc_id for r in raw_results} & {r.doc_id for r in bm_results}
                ),
            }
        )
    validations["bm25"] = _bm25_numerical_validation()
    result = {
        "schema_version": 1,
        "metadata": {
            "dataset": "20 Newsgroups",
            "document_count": len(bundle.texts),
            "category_count": len(bundle.target_names),
            "corpus_fingerprint": fingerprint,
            "runtime_artifact_sha256": runtime.metadata["arrays_sha256"],
            "query_count": len(query_ids),
            "requested_query_count": query_count,
            "latency_query_count": len(latency_queries),
            "repeats": repeats,
            "random_state": random_state,
            "epochs": epochs,
            "batch_size": batch_size,
            "topk": k,
            "bm25": {"k1": bm25.k1, "b": bm25.b, "query_terms": "unique"},
            "preprocessing": "existing EnglishPreprocessor, default stopwords, existing sanitization",
            "runtime_load_ms": load_ms,
            "quality_corpus": "train split",
            "demo_corpus": "full corpus",
            "relevance": "same category proxy",
            "zero_score_policy": "fill by ascending document ID to k; empty queries retained in quality, omitted in latency",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
                "packages": {
                    name: version(name)
                    for name in ("numpy", "scikit-learn", "matplotlib")
                },
            },
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "split": {
            "test_size": DEFAULT_TEST_SIZE,
            "stratified": True,
            "random_state": random_state,
            "train_document_ids": bundle.source_doc_ids[train_ids].tolist(),
            "test_document_ids": bundle.source_doc_ids[test_ids].tolist(),
            "query_document_ids": bundle.source_doc_ids[query_ids].tolist(),
        },
        "validation": validations,
        "tf_variants": tf_rows,
        "bm25_comparison": algorithm_rows,
        "quality_difference_cases": _quality_difference_cases(
            tf_rows[0]["retrieval_queries"], bm_query_records
        ),
        "inverted_index": speed_rows,
        "stop_word_ablation": ablation,
        "top5_comparisons": comparisons,
    }
    from .reporting import write_benchmark_reports

    write_benchmark_reports(Path(reports_dir), result, bundle.target_names)
    return result


def latency_statistics(seconds: Sequence[float]) -> dict[str, object]:
    """Summarize finite latency samples in milliseconds; leave empty sets unmeasured."""
    if not seconds:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "samples": 0}
    values = np.asarray(seconds, dtype=np.float64) * MILLISECONDS_PER_SECOND
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "samples": len(values),
    }


def _benchmark_inverted(
    bundle: DatasetBundle,
    sizes: Sequence[int],
    queries: Sequence[str],
    repeats: int,
) -> list[dict[str, object]]:
    rows = []
    for count in sorted(
        {min(size, len(bundle.texts)) for size in sizes} | {len(bundle.texts)}
    ):
        texts = bundle.texts[:count]
        vectorizer = NumpyTfidfVectorizer()
        start = time.perf_counter()
        matrix = vectorizer.fit_transform(texts)
        tfidf_build_seconds = time.perf_counter() - start
        start = time.perf_counter()
        postings = InvertedIndex.from_matrix(matrix)
        posting_build_seconds = time.perf_counter() - start
        searcher = DocumentSearch(
            vectorizer,
            matrix,
            [make_safe_snippet(t) for t in texts],
            bundle.labels[:count],
            bundle.target_names,
            bundle.source_doc_ids[:count],
            postings,
        )
        valid_queries = [
            q
            for q in queries
            if any(
                t in vectorizer.vocabulary_ for t in vectorizer.preprocessor.tokenize(q)
            )
        ]
        topk = min(RETRIEVAL_TOP_K, count)
        for query in valid_queries:
            _assert_consistent(
                searcher.search(query, topk, strategy="scan"),
                searcher.search(query, topk, strategy="inverted"),
            )
        scan = _timed_search(
            lambda q, k: searcher.search(q, k, strategy="scan"),
            valid_queries,
            repeats,
            topk,
        )
        inverted = _timed_search(
            lambda q, k: searcher.search(q, k, strategy="inverted"),
            valid_queries,
            repeats,
            topk,
        )
        rows.append(
            {
                "document_count": count,
                "queries": valid_queries,
                "skipped_oov_queries": [q for q in queries if q not in valid_queries],
                "query_count": len(valid_queries),
                "repeats": repeats,
                "topk": topk,
                "tfidf_build_ms": tfidf_build_seconds * MILLISECONDS_PER_SECOND,
                "inverted_build_ms": posting_build_seconds * MILLISECONDS_PER_SECOND,
                "scan_index_bytes": matrix.memory_stats()["sparse_bytes"],
                "inverted_index_bytes": postings.memory_bytes,
                "memory_definition": "resident NumPy array payload; inverted is additional, excludes Python overhead",
                "scan_mean_ms": scan["mean_ms"],
                "scan_p50_ms": scan["p50_ms"],
                "scan_p95_ms": scan["p95_ms"],
                "inverted_mean_ms": inverted["mean_ms"],
                "inverted_p50_ms": inverted["p50_ms"],
                "inverted_p95_ms": inverted["p95_ms"],
                "samples_per_method": scan["samples"],
                "speedup": scan["mean_ms"] / inverted["mean_ms"]
                if inverted["mean_ms"]
                else None,
                "search_consistency": bool(valid_queries),
                "score_tolerance": SEARCH_SCORE_TOLERANCE,
            }
        )
    return rows


def _timed_search(
    search, queries: Sequence[str], repeats: int, topk: int
) -> dict[str, object]:
    for query in queries:
        search(query, topk)  # 준비 실행과 인덱스 로드 시간은 측정에서 제외한다.
    samples = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter()
            search(query, topk)
            samples.append(time.perf_counter() - start)
    return latency_statistics(samples)


def _assert_consistent(scan, inverted) -> None:
    if [r.doc_id for r in scan] != [r.doc_id for r in inverted]:
        raise RuntimeError("scan/inverted ranks differ")
    if not np.allclose(
        [r.score for r in scan],
        [r.score for r in inverted],
        atol=SEARCH_SCORE_TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError("scan/inverted scores differ beyond 1e-12")


def _bm25_numerical_validation() -> dict[str, object]:
    model = NumpyBM25Retriever.from_texts(
        ["alpha alpha beta", "beta", "the"],
        snippets=["alpha alpha beta", "beta", "the"],
        labels=np.array([0, 1, 0]),
        target_names=("first", "second"),
        document_ids=np.array([9, 3, 1]),
    )
    expected = np.log(1 + 2.5 / 1.5) * 5 / (2 + 1.5 * (0.25 + 0.75 * 3 / (4 / 3)))
    error = abs(model.search("alpha", 1)[0].score - expected)
    return {
        "reference": "hand calculation",
        "max_absolute_error": float(error),
        "tolerance": SEARCH_SCORE_TOLERANCE,
        "passed": bool(error <= SEARCH_SCORE_TOLERANCE),
    }


def _quality_difference_cases(
    tfidf_queries: Sequence[dict], bm25_queries: Sequence[dict]
) -> dict[str, object]:
    cases = []
    for tfidf, bm25 in zip(tfidf_queries, bm25_queries, strict=True):
        if tfidf["query_document_id"] != bm25["query_document_id"]:
            raise ValueError("quality comparison requires identical query IDs")
        delta = bm25["ap_at_10"] - tfidf["ap_at_10"]
        cases.append(
            {
                "query_document_id": tfidf["query_document_id"],
                "query_category_id": tfidf["query_category_id"],
                "bm25_minus_tfidf_ap": delta,
                "tfidf_precision_at_10": tfidf["precision_at_10"],
                "bm25_precision_at_10": bm25["precision_at_10"],
                "tfidf_ap_at_10": tfidf["ap_at_10"],
                "bm25_ap_at_10": bm25["ap_at_10"],
            }
        )
    bm_better = sorted(
        (c for c in cases if c["bm25_minus_tfidf_ap"] > SEARCH_SCORE_TOLERANCE),
        key=lambda c: -c["bm25_minus_tfidf_ap"],
    )
    tf_better = sorted(
        (c for c in cases if c["bm25_minus_tfidf_ap"] < -SEARCH_SCORE_TOLERANCE),
        key=lambda c: c["bm25_minus_tfidf_ap"],
    )
    return {
        "criterion": "per-query AP@10, absolute tie tolerance 1e-12",
        "bm25_better_count": len(bm_better),
        "tfidf_better_count": len(tf_better),
        "equal_count": len(cases) - len(bm_better) - len(tf_better),
        "bm25_better_examples": bm_better[:3],
        "tfidf_better_examples": tf_better[:3],
    }
