"""Reproducible stop-word ablation for classification and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from ..classification.evaluation import evaluate_classifier
from ..classification.model import train_linear_svm
from ..constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_TEST_SIZE
from ..dataset import DatasetBundle, load_20newsgroups
from ..preprocessing import DEFAULT_STOP_WORDS, EnglishPreprocessor
from ..privacy import make_safe_snippet
from ..tfidf import NumpyTfidfVectorizer
from .metrics import evaluate_label_retrieval

ABLATION_SEEDS = tuple(range(10))


@dataclass(frozen=True)
class AblationVariant:
    name: str
    stop_word_count: int
    vocabulary_size: int
    accuracy: float
    macro_f1: float
    precision_at_10: float
    map_at_10: float


def summarize_variants(
    variants: Sequence[AblationVariant],
) -> dict[str, object]:
    """Summarize one ablation condition over independent random seeds."""

    if not variants:
        raise ValueError("at least one variant is required")
    stop_word_counts = {variant.stop_word_count for variant in variants}
    if len(stop_word_counts) != 1:
        raise ValueError("all variants must use the same stop-word count")

    def summary(metric: str) -> dict[str, float]:
        values = np.asarray([getattr(variant, metric) for variant in variants])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return {
        "runs": len(variants),
        "stop_word_count": stop_word_counts.pop(),
        "vocabulary_size": summary("vocabulary_size"),
        "accuracy": summary("accuracy"),
        "macro_f1": summary("macro_f1"),
        "precision_at_10": summary("precision_at_10"),
        "map_at_10": summary("map_at_10"),
    }


def run_stop_word_ablation(
    bundle: DatasetBundle | None = None,
) -> dict[str, object]:
    """Compare fixed default stop words with no stop-word removal over 10 seeds."""

    dataset = bundle or load_20newsgroups()
    results_by_variant: dict[str, list[AblationVariant]] = {
        "default_stop_words": [],
        "no_stop_words": [],
    }
    seed_results: list[dict[str, object]] = []
    for seed in ABLATION_SEEDS:
        variants = _run_stop_word_ablation_seed(dataset, seed)
        seed_results.append(
            {"seed": seed, "variants": [asdict(item) for item in variants]}
        )
        for variant in variants:
            results_by_variant[variant.name].append(variant)

    default_variants = results_by_variant["default_stop_words"]
    no_stop_words_variants = results_by_variant["no_stop_words"]
    deltas = [
        AblationVariant(
            name="default_minus_none",
            stop_word_count=0,
            vocabulary_size=default.vocabulary_size - no_stop.vocabulary_size,
            accuracy=default.accuracy - no_stop.accuracy,
            macro_f1=default.macro_f1 - no_stop.macro_f1,
            precision_at_10=default.precision_at_10 - no_stop.precision_at_10,
            map_at_10=default.map_at_10 - no_stop.map_at_10,
        )
        for default, no_stop in zip(
            default_variants, no_stop_words_variants, strict=True
        )
    ]
    return {
        "dataset": "20 Newsgroups: all 20 categories",
        "split": {
            "seeds": list(ABLATION_SEEDS),
            "test_size": DEFAULT_TEST_SIZE,
        },
        "retrieval_definition": {
            "corpus": "training documents",
            "queries": "held-out test documents",
            "relevance": "same category label",
            "metrics": "Precision@10 and MAP@10",
        },
        "variants": [
            {"name": name, **summarize_variants(variants)}
            for name, variants in results_by_variant.items()
        ],
        "delta_default_minus_none": summarize_variants(deltas),
        "seed_results": seed_results,
    }


def _run_stop_word_ablation_seed(
    dataset: DatasetBundle, seed: int
) -> list[AblationVariant]:
    """Evaluate both stop-word conditions on a single paired random seed."""

    dataset_row_ids = np.arange(len(dataset.texts))
    train_row_ids, test_row_ids = train_test_split(
        dataset_row_ids,
        test_size=DEFAULT_TEST_SIZE,
        stratify=dataset.labels,
        random_state=seed,
    )
    train_texts = [dataset.texts[int(row_id)] for row_id in train_row_ids]
    test_texts = [dataset.texts[int(row_id)] for row_id in test_row_ids]
    train_labels = dataset.labels[train_row_ids]
    test_labels = dataset.labels[test_row_ids]
    test_snippets = [make_safe_snippet(text) for text in test_texts]
    variants: list[AblationVariant] = []
    for variant_name, variant_stop_words in (
        ("default_stop_words", DEFAULT_STOP_WORDS),
        ("no_stop_words", frozenset()),
    ):
        vectorizer = NumpyTfidfVectorizer(
            EnglishPreprocessor(stop_words=variant_stop_words)
        )
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        model = train_linear_svm(
            train_matrix,
            train_labels,
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            random_state=seed,
        )
        classification_report = evaluate_classifier(
            model,
            test_matrix,
            test_labels,
            test_snippets,
            dataset.target_names,
            batch_size=DEFAULT_BATCH_SIZE,
            document_ids=dataset.source_doc_ids[test_row_ids],
        )
        retrieval_metrics = evaluate_label_retrieval(
            test_matrix,
            test_labels,
            train_matrix,
            train_labels,
        )
        variants.append(
            AblationVariant(
                name=variant_name,
                stop_word_count=len(variant_stop_words),
                vocabulary_size=len(vectorizer.vocabulary_),
                accuracy=classification_report.accuracy,
                macro_f1=classification_report.macro_f1,
                precision_at_10=retrieval_metrics.precision_at_k,
                map_at_10=retrieval_metrics.map_at_k,
            )
        )
    return variants
