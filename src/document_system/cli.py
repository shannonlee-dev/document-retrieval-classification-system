"""Command-line interface for building and searching the corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .artifacts import load_search_artifacts
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_REPORTS_DIR,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_TOP_K,
)
from .search import DocumentSearch


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "build":
        return _run_build(arguments[1:])
    return _run_search(arguments)


def _run_build(arguments: Sequence[str]) -> int:
    from .pipeline import BuildConfig, build_project

    parser = argparse.ArgumentParser(description="Build TF-IDF reports and search index")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    options = parser.parse_args(arguments)
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
        f"vocabulary={report.vocabulary_size:,}, "
        f"max_error={report.max_absolute_error:.3e}, "
        f"accuracy={report.accuracy:.4f}, macro_f1={report.macro_f1:.4f}"
    )
    return 0


def _run_search(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Search the 20 Newsgroups corpus")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    options = parser.parse_args(arguments)
    try:
        artifacts = load_search_artifacts(options.artifacts)
        searcher = DocumentSearch(
            vectorizer=artifacts.vectorizer,
            matrix=artifacts.matrix,
            snippets=artifacts.snippets,
            labels=artifacts.labels,
            target_names=artifacts.target_names,
            document_ids=artifacts.document_ids,
        )
        results = searcher.search(options.query, options.topk)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f'Query: "{options.query}"')
    print(f"Top {options.topk} results:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result.score:.6f} doc_id={result.doc_id} "
            f"category={result.label}\n   {result.text_snippet}"
        )
    return 0
