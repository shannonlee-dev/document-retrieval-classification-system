"""Offline presentation output using the standard library only."""

from __future__ import annotations

import json
import math
import shutil
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import RetrievalArtifacts
from .constants import (
    DEFAULT_TOP_K,
    MILLISECONDS_PER_SECOND,
    SEARCH_SCORE_TOLERANCE,
    TF_MODES,
)
from .experiments.metrics import RETRIEVAL_TOP_K

if TYPE_CHECKING:
    from .dataset import DatasetBundle

MODE_NAMES = {"raw": "Raw", "log": "Log", "double": "Double"}
DEMO_QUERY = "space shuttle orbit"
_MINIMUM_TERMINAL_WIDTH = 40
_MAXIMUM_OUTPUT_WIDTH = 76
_FALLBACK_TERMINAL_SIZE = (_MAXIMUM_OUTPUT_WIDTH, 24)
_MINIMUM_COLUMN_WIDTH = 7
_COLUMN_SEPARATOR_WIDTH = 3
_PREVIEW_INDENT = "  "
_MAX_MISCLASSIFICATION_PREVIEWS = 3


def load_benchmark(
    reports: Path, runtime: RetrievalArtifacts, *, required: bool
) -> dict | None:
    """Load matching benchmark evidence; allow missing reports only when optional."""
    path = reports / "retrieval_benchmark.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                "Benchmark results are not available.\n\nRun:\n  python -m document_system benchmark\nThen:\n  python -m document_system demo"
            )
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(
            "Benchmark version mismatch; run python -m document_system benchmark"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(
            "Benchmark metadata is invalid; run python -m document_system benchmark"
        )
    if metadata.get("corpus_fingerprint") != runtime.corpus_fingerprint:
        raise ValueError(
            "Benchmark corpus mismatch; run python -m document_system benchmark"
        )
    if metadata.get("runtime_artifact_sha256") != runtime.metadata["arrays_sha256"]:
        raise ValueError(
            "Benchmark index configuration mismatch; run python -m document_system benchmark"
        )
    _validate_benchmark_schema(payload, runtime)
    return payload


def run_demo(runtime: RetrievalArtifacts, report: dict, reports_dir: Path) -> bool:
    """Present six offline stages and return whether all validation evidence passes."""
    title("DOCUMENT RETRIEVAL & CLASSIFICATION SYSTEM")
    print(
        f"Dataset       : {runtime.metadata['dataset']}\nDocuments     : {runtime.metadata['fit_document_count']:,}\nCategories    : {runtime.metadata['category_count']}\nVectorization : NumPy TF-IDF\nClassifiers   : Linear SVM"
    )
    print(
        "\nAvailable Retrieval Methods\n  - TF-IDF (Raw / Log / Double)\n  - BM25\n  - Inverted Index Search"
    )
    query = DEMO_QUERY
    topk = min(DEFAULT_TOP_K, int(runtime.metadata["fit_document_count"]))
    title("[1/6] TF-IDF DOCUMENT SEARCH")
    show_search(runtime.tfidf["raw"], query, topk)
    title("[2/6] TF WEIGHTING COMPARISON")
    show_tf_comparison(runtime, query, report)
    title("[3/6] TF-IDF VS BM25")
    show_algorithm_comparison(runtime, query, topk)
    title("[4/6] SEARCH PERFORMANCE COMPARISON")
    consistent = show_speed_comparison(runtime, query, report)
    title("[5/6] DOCUMENT CLASSIFICATION")
    show_classification(report, reports_dir)
    title("[6/6] EXPERIMENT SUMMARY")
    validations = report["validation"]
    statuses = {
        "TF-IDF Numerical Validation": all(validations[m]["passed"] for m in TF_MODES),
        "BM25 Numerical Validation": validations["bm25"]["passed"],
        "Inverted Index": all(
            r["search_consistency"] for r in report["inverted_index"]
        ),
        "Search Consistency": consistent,
    }
    print(f"TF Variants Implemented: {len(runtime.tfidf)}")
    for name, passed in statuses.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(
        f"\nReports: {reports_dir}/\nValidation PASS means implementation checks passed; quality is shown separately."
    )
    print("Demo Completed." if all(statuses.values()) else "Demo validation failed.")
    return all(statuses.values())


def interactive(
    runtime: RetrievalArtifacts, report: dict | None, reports_dir: Path
) -> None:
    """Run the offline menu until exit or EOF, recovering from invalid input."""
    documents = None
    while True:
        title("DOCUMENT RETRIEVAL & CLASSIFICATION SYSTEM")
        print(
            "[1] Search Documents\n[2] Compare TF-IDF Variants\n[3] Compare TF-IDF vs BM25\n[4] Compare Scan vs Inverted Index\n[5] Show Classification Results\n[6] Run Full Demo\n[7] Show Full Document Text\n[0] Exit\nSelect an option:"
        )
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return
        if choice in ("0", "q"):
            return
        if choice not in ("1", "2", "3", "4", "5", "6", "7"):
            print("Select a number from 0 to 7, or q to exit.")
            continue
        try:
            if choice in ("1", "2", "3", "4"):
                query = input("Query: ")
                if not query.strip():
                    raise ValueError(
                        "query must not be blank; please enter search terms"
                    )
                topk = min(DEFAULT_TOP_K, int(runtime.metadata["fit_document_count"]))
                if choice == "1":
                    show_search(runtime.tfidf["raw"], query, topk, strategy="inverted")
                elif choice == "2":
                    show_tf_comparison(runtime, query, report)
                elif choice == "3":
                    show_algorithm_comparison(runtime, query, topk)
                else:
                    show_speed_comparison(runtime, query, report)
            elif choice == "5":
                show_classification(report, reports_dir)
            elif choice == "7":
                document_id_text = input("Document ID: ").strip()
                try:
                    document_id = int(document_id_text)
                except ValueError:
                    raise ValueError("Document ID must be an integer.") from None
                if document_id not in runtime.tfidf["raw"].document_ids:
                    raise ValueError(
                        f"Document ID {document_id} is not in the search index."
                    )
                if documents is None:
                    documents = _load_full_documents(runtime)
                show_document(documents, document_id)
            elif report:
                run_demo(runtime, report, reports_dir)
            else:
                print(
                    "Benchmark results are not available. Run python -m document_system benchmark."
                )
        except ValueError as error:
            print(f"Input error: {error}")
        except EOFError:
            return


def _load_full_documents(runtime: RetrievalArtifacts) -> DatasetBundle:
    """Load matching sanitized full texts once, without downloading or fitting."""
    from .dataset import dataset_fingerprint, load_20newsgroups

    try:
        documents = load_20newsgroups(download_if_missing=False)
    except OSError as error:
        raise ValueError(
            "Full text requires the local 20 Newsgroups dataset cache. "
            "Restore the cache or run make prepare to download it."
        ) from error
    if dataset_fingerprint(documents) != runtime.corpus_fingerprint:
        raise ValueError(
            "Dataset cache/index corpus mismatch; use the matching cache "
            "or run make prepare to rebuild the index."
        )
    return documents


def show_document(documents: DatasetBundle, document_id: int) -> None:
    """Display a sanitized full document using its original public ID."""
    try:
        row = documents.source_doc_ids.tolist().index(document_id)
    except ValueError:
        raise ValueError(f"Document ID {document_id} is not in the dataset.") from None
    title("DOCUMENT FULL TEXT")
    print(f"Document ID: {document_id}")
    print(f"Category: {documents.target_names[int(documents.labels[row])]}")
    print("\nFull text (sanitized):")
    print(textwrap.fill(documents.texts[row], width=_output_width()))


def title(text: str) -> None:
    width = _output_width()
    print(f"\n{text}\n{'-' * width}")


def show_search(
    searcher, query: str, topk: int, *, method="tfidf", mode="raw", strategy="scan"
) -> None:
    """Print ranked results, a safe preview and latency excluding index loading."""
    call = (
        (lambda q, k: searcher.search(q, k, strategy=strategy))
        if method == "tfidf"
        else searcher.search
    )
    results, latency = _timed_results(call, query, topk)
    print(
        f"Query      : {query}\nMethod     : {'TF-IDF' if method == 'tfidf' else 'BM25'}"
    )
    if method == "tfidf":
        print(
            f"TF Mode    : {MODE_NAMES[mode]}\nIndex      : {'Full Scan' if strategy == 'scan' else 'Inverted Index'}"
        )
    else:
        print(
            f"Parameters : k1={searcher.k1}, b={searcher.b}; unique query terms\nIndex      : Inverted Index"
        )
    print(f"Top-K      : {topk}\nReturned   : {len(results)}")
    _table(
        ["Rank", "Score", "Document ID", "Category"],
        [
            [rank, f"{r.score:.6f}", r.doc_id, r.label]
            for rank, r in enumerate(results, 1)
        ],
    )
    if results:
        print("\nTop Result Preview")
        first = results[0]
        # 기존 CLI 출력 소비자가 사용하는 필드 이름을 유지한다.
        print(
            f"1. score={first.score:.6f} doc_id={first.doc_id} category={first.label}"
        )
        _preview(first.text_snippet)
    else:
        print("No documents have a positive score.")
    print(f"Search Latency: {latency:.3f} ms (current query; excludes load)")


def show_tf_comparison(
    runtime: RetrievalArtifacts, query: str, report: dict | None
) -> None:
    """Compare TF rankings and show saved quality metrics when available."""
    topk = min(DEFAULT_TOP_K, int(runtime.metadata["fit_document_count"]))
    results = {
        mode: searcher.search(query, topk, strategy="inverted")
        for mode, searcher in runtime.tfidf.items()
    }
    print(f"Query: {query}")
    _table(
        ["TF Mode", "Top-1 Doc", "Top-1 Score"],
        [
            [
                MODE_NAMES[mode],
                ranked[0].doc_id if ranked else "none",
                f"{ranked[0].score:.6f}" if ranked else "none",
            ]
            for mode, ranked in results.items()
        ],
    )
    baseline_ranks = {r.doc_id: rank for rank, r in enumerate(results["raw"], 1)}
    for mode, ranked in results.items():
        print(
            f"{MODE_NAMES[mode]} Top-5 IDs: {', '.join(str(r.doc_id) for r in ranked)}"
        )
        shifts = [
            f"{r.doc_id}: {baseline_ranks.get(r.doc_id, 'outside top5')} → {rank}"
            for rank, r in enumerate(ranked, 1)
            if baseline_ranks.get(r.doc_id) != rank
        ]
        if mode != "raw":
            print("  Rank changes vs Raw: " + ("; ".join(shifts) if shifts else "none"))
    if report:
        print("\nSaved paired quality benchmark (train corpus / held-out queries)")
        _table(
            ["TF", "Accuracy", "Macro F1", "P@10", "MAP@10"],
            [
                [
                    MODE_NAMES[r["tf_mode"]],
                    *[
                        _number(r[key])
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "precision_at_10",
                            "map_at_10",
                        )
                    ],
                ]
                for r in report["tf_variants"]
            ],
        )
    else:
        print(
            "Quality metrics: not measured here. Run python -m document_system benchmark."
        )


def show_algorithm_comparison(
    runtime: RetrievalArtifacts, query: str, topk: int
) -> None:
    """Compare TF-IDF and BM25 ranks, document lengths and current latency."""
    raw, raw_ms = _timed_results(
        lambda q, k: runtime.tfidf["raw"].search(q, k, strategy="inverted"), query, topk
    )
    bm25, bm_ms = _timed_results(runtime.bm25.search, query, topk)
    print(f"Query: {query}")
    _table(
        ["Rank", "TF-IDF Doc ID", "BM25 Doc ID"],
        [
            [
                i + 1,
                raw[i].doc_id if i < len(raw) else "—",
                bm25[i].doc_id if i < len(bm25) else "—",
            ]
            for i in range(max(len(raw), len(bm25)))
        ],
    )
    overlap = {r.doc_id for r in raw} & {r.doc_id for r in bm25}
    print(f"Top-{topk} common documents: {len(overlap)}")
    raw_ranks = {r.doc_id: i for i, r in enumerate(raw, 1)}
    print(
        "Rank differences: "
        + (
            "; ".join(
                f"{r.doc_id}: {raw_ranks.get(r.doc_id, 'outside topk')} → {i}"
                for i, r in enumerate(bm25, 1)
                if raw_ranks.get(r.doc_id) != i
            )
            or "none"
        )
    )
    lengths = dict(
        zip(
            runtime.bm25.documents.document_ids,
            runtime.bm25.document_lengths,
            strict=True,
        )
    )
    for name, results in (("TF-IDF", raw), ("BM25", bm25)):
        if results:
            print(
                f"{name} top preview (ID {results[0].doc_id}, tokens {lengths[results[0].doc_id]:.0f})"
            )
            _preview(results[0].text_snippet)
    print(
        f"Current latency: TF-IDF {raw_ms:.3f} ms; BM25 {bm_ms:.3f} ms (load excluded)"
    )
    print(
        "Scores have different scales. BM25 saturates TF and normalizes token length."
    )


def show_speed_comparison(
    runtime: RetrievalArtifacts, query: str, report: dict | None
) -> bool:
    """Print scan/posting timing evidence and return whether current results agree."""
    topk = min(RETRIEVAL_TOP_K, int(runtime.metadata["fit_document_count"]))
    raw = runtime.tfidf["raw"]
    scan, scan_ms = _timed_results(
        lambda q, k: raw.search(q, k, strategy="scan"), query, topk
    )
    inverted, inverted_ms = _timed_results(
        lambda q, k: raw.search(q, k, strategy="inverted"), query, topk
    )
    consistent = [r.doc_id for r in scan] == [r.doc_id for r in inverted] and all(
        abs(a.score - b.score) <= SEARCH_SCORE_TOLERANCE
        for a, b in zip(scan, inverted, strict=True)
    )
    print(
        f"Query: {query}\nResult Consistency: {'PASS' if consistent else 'FAIL'} (absolute tolerance 1e-12)"
    )
    print(
        f"Current single query: Scan {scan_ms:.3f} ms / Inverted {inverted_ms:.3f} ms"
    )
    if report:
        full_rows = [
            r
            for r in report["inverted_index"]
            if r["document_count"] == runtime.metadata["fit_document_count"]
        ]
        if full_rows:
            stats = full_rows[0]
            print(
                f"\nSaved repeated benchmark: {stats['query_count']} short queries × {stats['repeats']} repeats; warm-up excluded"
            )
            _table(
                ["Method", "Mean(ms)", "P50(ms)", "P95(ms)"],
                [
                    [
                        "Full Scan",
                        *[
                            _number(stats[f"scan_{key}_ms"], 3)
                            for key in ("mean", "p50", "p95")
                        ],
                    ],
                    [
                        "Inverted Index",
                        *[
                            _number(stats[f"inverted_{key}_ms"], 3)
                            for key in ("mean", "p50", "p95")
                        ],
                    ],
                ],
            )
            print(f"Speedup: {_number(stats['speedup'], 2)}x")
    else:
        print(
            "Repeated latency/P95: not measured here. Run python -m document_system benchmark."
        )
    return consistent


def show_classification(report: dict | None, reports_dir: Path) -> None:
    """Show saved classification metrics and bounded misclassification previews."""
    if report is None:
        print(
            "Classification results are not available. Run python -m document_system benchmark."
        )
        return
    metadata = report["metadata"]
    print(
        f"Model       : Linear SVM (SGD hinge)\nTrain/Test  : 80 / 20 (seed {metadata['random_state']})\nCategories  : {metadata['category_count']}"
    )
    _table(
        ["TF Mode", "Accuracy", "Macro F1"],
        [
            [MODE_NAMES[r["tf_mode"]], _number(r["accuracy"]), _number(r["macro_f1"])]
            for r in report["tf_variants"]
        ],
    )
    raw = next(r for r in report["tf_variants"] if r["tf_mode"] == "raw")
    print(f"Confusion Matrix: {reports_dir / 'tf_raw_confusion_matrix.png'}")
    print("Misclassification previews:")
    for error in raw["misclassifications"][:_MAX_MISCLASSIFICATION_PREVIEWS]:
        print(f"  ID {error['doc_id']}: {error['actual']} → {error['predicted']}")
        _preview(error["text_snippet"])
    if not raw["misclassifications"]:
        print("  No misclassifications in this evaluation.")


def _validate_benchmark_schema(payload: dict, runtime: RetrievalArtifacts) -> None:
    """Reject incomplete persisted evidence before formatting any demo output."""

    def number(value, *, optional=False):
        return (optional and value is None) or (
            type(value) in (int, float) and math.isfinite(value) and value >= 0
        )

    try:
        metadata = payload["metadata"]
        valid = (
            metadata["document_count"] == runtime.metadata["fit_document_count"]
            and metadata["category_count"] == runtime.metadata["category_count"]
        )
        valid &= type(metadata["random_state"]) is int
        rows = payload["tf_variants"]
        valid &= isinstance(rows, list) and len(rows) == len(TF_MODES)
        valid &= {row["tf_mode"] for row in rows} == set(TF_MODES)
        for row in rows:
            valid &= all(
                number(row[key]) and row[key] <= 1
                for key in ("accuracy", "macro_f1", "precision_at_10", "map_at_10")
            )
            errors = row["misclassifications"]
            valid &= isinstance(errors, list)
            for error in errors:
                valid &= type(error["doc_id"]) is int and all(
                    isinstance(error[key], str)
                    for key in ("actual", "predicted", "text_snippet")
                )
        for method in (*TF_MODES, "bm25"):
            validation = payload["validation"][method]
            valid &= type(validation["passed"]) is bool
            valid &= number(validation["max_absolute_error"]) and number(
                validation["tolerance"]
            )
            valid &= validation["passed"] == (
                validation["max_absolute_error"] <= validation["tolerance"]
            )
        speed_rows = payload["inverted_index"]
        valid &= isinstance(speed_rows, list) and bool(speed_rows)
        valid &= any(
            row["document_count"] == metadata["document_count"] for row in speed_rows
        )
        for row in speed_rows:
            valid &= all(
                type(row[key]) is int and row[key] >= 0
                for key in ("document_count", "query_count", "repeats")
            )
            valid &= type(row["search_consistency"]) is bool
            valid &= all(
                number(row[key], optional=True)
                for key in (
                    "scan_mean_ms",
                    "scan_p50_ms",
                    "scan_p95_ms",
                    "inverted_mean_ms",
                    "inverted_p50_ms",
                    "inverted_p95_ms",
                    "speedup",
                )
            )
        split = payload["split"]
        valid &= all(
            isinstance(split[key], list)
            for key in ("train_document_ids", "test_document_ids", "query_document_ids")
        )
        valid &= not set(split["train_document_ids"]) & set(split["test_document_ids"])
        valid &= set(split["query_document_ids"]) <= set(split["test_document_ids"])
    except (KeyError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ValueError(
            "Benchmark report is incomplete or invalid; run python -m document_system benchmark"
        )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    cells = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in cells)) if cells else len(header)
        for i, header in enumerate(headers)
    ]
    available = _terminal_width() - _COLUMN_SEPARATOR_WIDTH * (len(headers) - 1)
    # 좁은 터미널에서는 가장 넓은 열부터 줄여 카테고리와 ID 목록을 표시한다.
    while sum(widths) > available and max(widths) > _MINIMUM_COLUMN_WIDTH:
        widest = max(range(len(widths)), key=widths.__getitem__)
        widths[widest] -= 1

    def render(row):
        return " | ".join(
            (value[: width - 1] + "…" if len(value) > width else value).ljust(width)
            for value, width in zip(row, widths, strict=True)
        ).rstrip()

    print(render(list(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in cells:
        print(render(row))


def _preview(text: str) -> None:
    width = _output_width() - len(_PREVIEW_INDENT)
    print(
        textwrap.fill(
            text,
            width=width,
            initial_indent=_PREVIEW_INDENT,
            subsequent_indent=_PREVIEW_INDENT,
        )
    )


def _number(value, digits=4) -> str:
    return f"{value:.{digits}f}" if value is not None else "not measured"


def _timed_results(search, query: str, topk: int):
    start = time.perf_counter()
    results = search(query, topk)
    return results, (time.perf_counter() - start) * MILLISECONDS_PER_SECOND


def _terminal_width() -> int:
    return max(
        _MINIMUM_TERMINAL_WIDTH,
        shutil.get_terminal_size(_FALLBACK_TERMINAL_SIZE).columns,
    )


def _output_width() -> int:
    return min(_MAXIMUM_OUTPUT_WIDTH, _terminal_width())
