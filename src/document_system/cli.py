"""Build/benchmark preparation and offline presentation/search commands."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .artifacts import load_retrievers, load_search_artifacts
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_REPORTS_DIR,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_TOP_K,
    TF_MODES,
)
from .inverted_index import InvertedIndex
from .presentation import (
    interactive,
    load_benchmark,
    run_demo,
    show_algorithm_comparison,
    show_search,
    show_speed_comparison,
    show_tf_comparison,
    title,
)
from .search import DocumentSearch


def _run_build(arguments: Sequence[str]) -> int:
    from .pipeline import BuildConfig, build_project

    parser = _parser(
        "Build TF-IDF/BM25/posting indexes and classification reports", reports=True
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    options = parser.parse_args(arguments)
    _use_local_dataset_cache()
    report = build_project(
        BuildConfig(
            runtime_dir=options.artifacts,
            reports_dir=options.reports,
            batch_size=options.batch_size,
            epochs=options.epochs,
        )
    )
    print(
        f"build complete: documents={report.document_count:,}, "
        f"classification_vocabulary={report.classification_vocabulary_size:,}, "
        f"search_vocabulary={report.search_vocabulary_size:,}, "
        f"classification_max_error={report.classification_max_absolute_error:.3e}, "
        f"search_max_error={report.search_max_absolute_error:.3e}, "
        f"accuracy={report.accuracy:.4f}, macro_f1={report.macro_f1:.4f}"
    )
    print(
        "Retrieval indexes: Raw / Log / Double / BM25 / Inverted\nNext: python -m document_system benchmark"
    )
    return 0


def _run_benchmark(arguments: Sequence[str]) -> int:
    from .experiments.retrieval_benchmark import (
        DEFAULT_BENCHMARK_QUERY_COUNT,
        DEFAULT_BENCHMARK_REPEATS,
    )

    parser = _parser("Run E1-E6, save measured CSV/JSON/Markdown reports", reports=True)
    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_BENCHMARK_QUERY_COUNT,
        help="held-out query budget (default 100; fixed shared subset)",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_BENCHMARK_REPEATS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    options = parser.parse_args(arguments)
    # 저장 인덱스가 없으면 데이터에 접근하기 전에 종료한다.
    load_retrievers(options.artifacts)
    _use_local_dataset_cache()
    from .dataset import load_20newsgroups
    from .experiments.retrieval_benchmark import run_benchmark

    run_benchmark(
        load_20newsgroups(),
        options.artifacts,
        options.reports,
        query_count=options.queries,
        repeats=options.repeats,
        epochs=options.epochs,
        batch_size=options.batch_size,
    )
    print(
        f"Benchmark complete. Reports: {options.reports}\nNext: python -m document_system demo"
    )
    return 0


def _run_search(arguments: Sequence[str]) -> int:
    parser = _parser(
        "Search the 20 Newsgroups corpus. Commands: build, benchmark, demo, search, compare, interactive"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--method", choices=("tfidf", "bm25"), default="tfidf")
    parser.add_argument("--tf-mode", choices=TF_MODES, default="raw")
    parser.add_argument("--strategy", choices=("scan", "inverted"), default="scan")
    options = parser.parse_args(arguments)
    if options.method == "tfidf" and options.tf_mode == "raw":
        saved = load_search_artifacts(options.artifacts)
        if saved.vectorizer.tf_mode != "raw":
            raise ValueError(
                "raw artifact is unavailable; rebuild with python -m document_system build"
            )
        postings = None
        arrays = saved.extra_arrays
        if options.strategy == "inverted" and "posting_order" in arrays:
            postings = InvertedIndex(
                arrays["posting_indptr"],
                arrays["posting_rows"],
                saved.matrix.data[arrays["posting_order"]],
                saved.matrix.shape[0],
            )
        searcher = DocumentSearch(
            saved.vectorizer,
            saved.matrix,
            saved.snippets,
            saved.labels,
            saved.target_names,
            saved.document_ids,
            postings,
        )
    else:
        runtime = load_retrievers(options.artifacts)
        searcher = (
            runtime.bm25 if options.method == "bm25" else runtime.tfidf[options.tf_mode]
        )
    title("DOCUMENT SEARCH")
    show_search(
        searcher,
        options.query,
        options.topk,
        method=options.method,
        mode=options.tf_mode,
        strategy=options.strategy,
    )
    return 0


def _run_presentation(command: str, arguments: Sequence[str]) -> int:
    parser = _parser(
        f"Offline {command} using saved indexes and benchmark results", reports=True
    )
    if command == "compare":
        parser.add_argument("--query", required=True)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    options = parser.parse_args(arguments)
    runtime = load_retrievers(options.artifacts)
    report = load_benchmark(options.reports, runtime, required=command == "demo")
    if command == "demo":
        return 0 if run_demo(runtime, report, options.reports) else 1
    if command == "interactive":
        _use_local_dataset_cache()
        interactive(runtime, report, options.reports)
        return 0
    runtime.tfidf["raw"].validate_topk(options.topk)
    title("TF WEIGHTING COMPARISON")
    show_tf_comparison(runtime, options.query, report)
    title("TF-IDF VS BM25")
    show_algorithm_comparison(runtime, options.query, options.topk)
    title("SCAN VS INVERTED INDEX")
    return 0 if show_speed_comparison(runtime, options.query, report) else 1


def _parser(description: str, *, reports: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_RUNTIME_DIR)
    if reports:
        parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser


def _use_local_dataset_cache() -> None:
    # 사용자가 지정한 데이터 캐시는 유지한다.
    os.environ.setdefault("SCIKIT_LEARN_DATA", str(Path("scikit_learn_data").resolve()))


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI commands and return an exit code for expected user errors."""
    arguments = list(argv) if argv is not None else sys.argv[1:]
    try:
        if arguments and arguments[0] == "build":
            return _run_build(arguments[1:])
        if arguments and arguments[0] == "benchmark":
            return _run_benchmark(arguments[1:])
        if arguments and arguments[0] in ("demo", "compare", "interactive"):
            return _run_presentation(arguments[0], arguments[1:])
        if arguments and arguments[0] == "search":
            arguments = arguments[1:]
        return _run_search(arguments)
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        if isinstance(error, FileNotFoundError) and "Benchmark" not in str(error):
            print(
                "\nSearch artifacts are not available.\n\nRun:\n  python -m document_system build\nThen:\n  python -m document_system benchmark\n  python -m document_system demo",
                file=sys.stderr,
            )
        elif "artifact" in str(error).lower():
            print("Rebuild with: python -m document_system build", file=sys.stderr)
        return 2
