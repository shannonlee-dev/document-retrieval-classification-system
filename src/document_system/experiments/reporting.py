"""Persistence for reproducible experiment reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .ablation import run_stop_word_ablation


def write_stop_word_ablation_report(path: str | Path) -> dict[str, object]:
    """Run the stop-word ablation and persist its JSON report."""

    report = run_stop_word_ablation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def write_benchmark_reports(
    directory: Path, report: dict[str, object], target_names: tuple[str, ...]
) -> None:
    """Write new reports without overwriting the historical build/ablation results."""
    import numpy as np

    from ..classification.evaluation import ClassificationReport
    from ..classification.visualization import save_confusion_matrix

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "retrieval_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tables = {
        "tf_variants_comparison.csv": (
            report["tf_variants"],
            [
                "tf_mode",
                "accuracy",
                "macro_f1",
                "precision_at_10",
                "map_at_10",
                "vocabulary_size",
            ],
        ),
        "bm25_comparison.csv": (
            report["bm25_comparison"],
            [
                "method",
                "tf_mode",
                "precision_at_10",
                "map_at_10",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "samples",
            ],
        ),
        "inverted_index_benchmark.csv": (
            report["inverted_index"],
            [
                "document_count",
                "query_count",
                "repeats",
                "samples_per_method",
                "scan_mean_ms",
                "scan_p50_ms",
                "scan_p95_ms",
                "inverted_mean_ms",
                "inverted_p50_ms",
                "inverted_p95_ms",
                "speedup",
                "tfidf_build_ms",
                "inverted_build_ms",
                "scan_index_bytes",
                "inverted_index_bytes",
                "search_consistency",
            ],
        ),
    }
    for name, (rows, columns) in tables.items():
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    for row in report["tf_variants"]:
        mode = row["tf_mode"]
        classification = ClassificationReport(
            row["accuracy"],
            row["macro_f1"],
            np.asarray(row["confusion_matrix"]),
            np.array([], dtype=int),
            row["misclassifications"],
            row["model_settings"],
        )
        save_confusion_matrix(
            classification, target_names, directory / f"tf_{mode}_confusion_matrix.png"
        )
        per_class_rows = [
            {"category": category, **row["per_class"][category]}
            for category in target_names
        ]
        with (directory / f"tf_{mode}_per_class.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["category", "precision", "recall", "f1-score", "support"],
            )
            writer.writeheader()
            writer.writerows(per_class_rows)
    (directory / "experiment_summary.md").write_text(
        _render_experiment_summary(report), encoding="utf-8"
    )


def _render_experiment_summary(report: dict[str, object]) -> str:
    """Render measured results and reproduction settings without recomputing them."""
    metadata = report["metadata"]
    lines = [
        "# Experiment summary",
        "",
        "## 구현 요약",
        "",
        "NumPy Raw/Log/Double TF-IDF, direct Okapi BM25 and sparse term postings. Existing SVM/loader/redaction preserved.",
        "",
        "## 검증 및 실험 결과",
        "",
        f"Dataset: {metadata['dataset']}, {metadata['document_count']:,} documents, {metadata['category_count']} categories.",
        f"Train/test: {len(report['split']['train_document_ids']):,}/{len(report['split']['test_document_ids']):,}, stratified 80/20; seed {metadata['random_state']}.",
        f"Quality: {metadata['query_count']} held-out document queries, k={metadata['topk']}. Latency: warm-up + {metadata['repeats']} repeats.",
        f"SVM: hinge, epochs={metadata['epochs']}, batch_size={metadata['batch_size']}, train-only fit.",
        f"Runtime load (excluded from query latency): {metadata['runtime_load_ms']:.3f} ms.",
        "",
        "| TF | Accuracy | Macro F1 | P@10 | MAP@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["tf_variants"]:
        lines.append(
            f"| {row['tf_mode']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['precision_at_10']:.6f} | {row['map_at_10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Method (train corpus) | P@10 | MAP@10 | Mean ms | P95 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["bm25_comparison"]:
        mean = f"{row['mean_ms']:.3f}" if row["mean_ms"] is not None else "not measured"
        p95 = f"{row['p95_ms']:.3f}" if row["p95_ms"] is not None else "not measured"
        lines.append(
            f"| {row['method']} | {row['precision_at_10']:.6f} | {row['map_at_10']:.6f} | {mean} | {p95} |"
        )
    lines.extend(
        [
            "",
            "Speed uses short presentation queries, separate from held-out quality queries.",
            "",
            "| Documents | Scan mean ms | Inverted mean ms | Scan P95 ms | Inverted P95 ms | Speedup | Extra posting MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["inverted_index"]:
        if row["speedup"] is None:
            lines.append(
                f"| {row['document_count']} | not measured | not measured | not measured | not measured | not measured | {row['inverted_index_bytes'] / 2**20:.3f} |"
            )
        else:
            lines.append(
                f"| {row['document_count']} | {row['scan_mean_ms']:.3f} | {row['inverted_mean_ms']:.3f} | {row['scan_p95_ms']:.3f} | {row['inverted_p95_ms']:.3f} | {row['speedup']:.2f}x | {row['inverted_index_bytes'] / 2**20:.3f} |"
            )
    cases = report.get("quality_difference_cases")
    if cases:
        lines.extend(
            [
                "",
                "### Actual per-query differences (held-out quality)",
                "",
                f"AP@10: BM25 better on {cases['bm25_better_count']} queries; TF-IDF better on {cases['tfidf_better_count']}; tied on {cases['equal_count']}.",
                "",
                "| Query document ID | TF-IDF P@10 | BM25 P@10 | TF-IDF AP@10 | BM25 AP@10 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for case in cases["bm25_better_examples"] + cases["tfidf_better_examples"]:
            lines.append(
                f"| {case['query_document_id']} | {case['tfidf_precision_at_10']:.4f} | {case['bm25_precision_at_10']:.4f} | {case['tfidf_ap_at_10']:.4f} | {case['bm25_ap_at_10']:.4f} |"
            )
    lines.extend(["", "Numerical validation:"])
    for name in ("raw", "log", "double", "bm25"):
        validation = report["validation"][name]
        lines.append(
            f"- {name}: {'PASS' if validation['passed'] else 'FAIL'}, maximum absolute error={validation['max_absolute_error']:.3e}; {validation.get('settings', validation.get('reference'))}."
        )
    lines.extend(
        [
            "",
            "Stopwords paired experiment (one seed, separate from historical ten-seed report):",
            "",
            "| Condition | Accuracy | Macro F1 | P@10 | MAP@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["stop_word_ablation"]["variants"]:
        lines.append(
            f"| {row['name']} | {row['accuracy']['mean']:.6f} | {row['macro_f1']['mean']:.6f} | {row['precision_at_10']['mean']:.6f} | {row['map_at_10']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Top-5 (full demo corpus)",
            "",
            "BM25 and cosine scores have different scales; only ranks/labels and token lengths are compared.",
        ]
    )
    for example in report["top5_comparisons"]:
        if example.get("status") == "not measured":
            lines.extend(
                ["", f"Query `{example['query']}`: not measured ({example['reason']})."]
            )
            continue
        lines.extend(
            [
                "",
                f"Query: `{example['query']}`; common documents: {example['top5_overlap']}.",
                "",
                "| Rank | TF-IDF ID / tokens / category | BM25 ID / tokens / category |",
                "|---:|---|---|",
            ]
        )
        for rank in range(max(len(example["tfidf"]), len(example["bm25"]))):
            cells = []
            for method in ("tfidf", "bm25"):
                result = example[method][rank] if rank < len(example[method]) else None
                cells.append(
                    f"{result['doc_id']} / {result['token_length']:.0f} / {result['label']}"
                    if result
                    else "—"
                )
            lines.append(f"| {rank + 1} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "### Misclassification examples", ""])
    for row in report["tf_variants"]:
        lines.append(
            f"{row['tf_mode']} confusion matrix: `tf_{row['tf_mode']}_confusion_matrix.png`; per-class precision/recall/F1: `tf_{row['tf_mode']}_per_class.csv`."
        )
        for error in row["misclassifications"][:5]:
            lines.append(
                f"- ID {error['doc_id']}: {error['actual']} → {error['predicted']}: {error['text_snippet']}"
            )
    lines.extend(
        [
            "",
            "## 한계와 개선 방향",
            "",
            "Same-category relevance is a proxy, not a semantic judgment. The recorded query subset limits generalization; this is one split and one timing run. Empty/OOV quality queries stay in the denominator; zero-score ranks fill by ID to maintain fixed-k metric definitions. Timing excludes load and compares identical queries per pair. Array memory excludes Python overhead and transient allocations.",
            "",
            "BoW discards word order and cannot resolve polysemy/context. Tokenization, stopwords and existing redaction lose information. Raw TF overweights repetition; log limits it; double stresses presence. BM25 saturates TF and normalizes length, which may help concise documents but needs tuning. Postings add construction/memory costs and frequent/long queries still touch many documents. Overlapping topic vocabularies and short/ambiguous documents cause SVM errors. Follow-up: repeated splits, judged queries, parameter studies and linguistic features.",
            "",
            "### Environment and reproduction",
            "",
            f"Executed: {metadata['executed_at_utc']}",
            f"Corpus fingerprint: `{metadata['corpus_fingerprint']}`",
            "",
            "```json",
            json.dumps(metadata["environment"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"Run `python main.py benchmark --queries {metadata['requested_query_count']} --repeats {metadata['repeats']} --epochs {metadata['epochs']} --batch-size {metadata['batch_size']}` after build. Full configuration and split/query IDs are in retrieval_benchmark.json.",
            "",
        ]
    )
    return "\n".join(lines)
