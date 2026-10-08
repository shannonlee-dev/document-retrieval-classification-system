"""Paired TF experiments in one train-only vocabulary/IDF space."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report

from ..classification.evaluation import evaluate_classifier
from ..classification.model import train_linear_svm
from ..dataset import DatasetBundle
from ..inverted_index import InvertedIndex
from ..privacy import make_safe_snippet
from ..search import DocumentSearch
from ..sparse_matrix import SparseMatrix
from ..tfidf import NumpyTfidfVectorizer
from ..validation import validate_against_sklearn
from .metrics import evaluate_ranked_labels, query_metrics_record, rank_scores


def double_tf_validation() -> dict[str, object]:
    """Independent small numerical reference; sklearn has no double-TF switch."""
    vectorizer = NumpyTfidfVectorizer(tf_mode="double")
    stages = vectorizer.fit_transform_with_stages(["alpha alpha beta", "beta", ""])
    reference = np.array([1 + np.log(2), 0.75 * (1 + np.log(4 / 3))])
    reference /= np.linalg.norm(reference)
    error = float(np.max(np.abs(reference - stages.tfidf.get_sparse_row(0)[1])))
    return {
        "reference": "hand calculation",
        "max_absolute_error": error,
        "tolerance": 1e-6,
        "passed": error <= 1e-6,
        "sklearn_equivalent": False,
    }


def run_tf_variants(
    bundle: DatasetBundle,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    k: int,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> tuple[
    list[dict[str, object]], dict[str, DocumentSearch], dict[str, object], SparseMatrix
]:
    train_texts = [bundle.texts[int(i)] for i in train_ids]
    test_texts = [bundle.texts[int(i)] for i in test_ids]
    query_texts = [bundle.texts[int(i)] for i in query_ids]
    raw = NumpyTfidfVectorizer()
    stages = raw.fit_transform_with_stages(train_texts)
    # Shared count rows for all three methods, including held-out transforms.
    test_counts = raw._count_matrix(test_texts)
    query_counts = raw._count_matrix(query_texts)
    train_snippets = tuple(make_safe_snippet(t) for t in train_texts)
    test_snippets = tuple(make_safe_snippet(t) for t in test_texts)
    searchers: dict[str, DocumentSearch] = {}
    validations: dict[str, object] = {}
    rows = []
    for mode in ("raw", "log", "double"):
        vectorizer = NumpyTfidfVectorizer(raw.preprocessor, tf_mode=mode)
        vectorizer.vocabulary_, vectorizer.feature_names_, vectorizer.idf_ = (
            raw.vocabulary_,
            raw.feature_names_,
            raw.idf_,
        )
        train_matrix = (
            stages.tfidf if mode == "raw" else vectorizer.weight_counts(stages.counts)
        )
        test_matrix = vectorizer.weight_counts(test_counts)
        query_matrix = vectorizer.weight_counts(query_counts)
        validation = (
            double_tf_validation()
            if mode == "double"
            else validate_against_sklearn(
                train_texts, vectorizer, train_matrix
            ).to_dict()
        )
        validations[mode] = validation
        if not validation["passed"]:
            raise RuntimeError(f"{mode} TF numerical validation failed")
        model = train_linear_svm(
            train_matrix,
            bundle.labels[train_ids],
            batch_size=batch_size,
            epochs=epochs,
            random_state=random_state,
        )
        classification = evaluate_classifier(
            model,
            test_matrix,
            bundle.labels[test_ids],
            test_snippets,
            bundle.target_names,
            batch_size=batch_size,
            document_ids=bundle.source_doc_ids[test_ids],
        )
        postings = InvertedIndex.from_matrix(train_matrix)
        searcher = DocumentSearch(
            vectorizer,
            train_matrix,
            train_snippets,
            bundle.labels[train_ids],
            bundle.target_names,
            bundle.source_doc_ids[train_ids],
            postings,
        )
        rankings = []
        query_records = []
        for query_row in range(query_matrix.shape[0]):
            terms, weights = query_matrix.get_sparse_row(query_row)
            scores = postings.score(terms, weights)
            ranked_rows = rank_scores(scores, bundle.source_doc_ids[train_ids], k)
            rankings.append(bundle.labels[train_ids][ranked_rows].tolist())
            query_records.append(
                query_metrics_record(
                    bundle.source_doc_ids[query_ids[query_row]],
                    bundle.labels[query_ids[query_row]],
                    ranked_rows,
                    bundle.labels[train_ids],
                    bundle.source_doc_ids[train_ids],
                    k,
                )
            )
        retrieval = evaluate_ranked_labels(
            query_labels=bundle.labels[query_ids],
            ranked_labels=rankings,
            corpus_labels=bundle.labels[train_ids],
            k=k,
        )
        per_class = classification_report(
            bundle.labels[test_ids],
            classification.predictions,
            labels=np.arange(len(bundle.target_names)),
            target_names=bundle.target_names,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "tf_mode": mode,
                "accuracy": classification.accuracy,
                "macro_f1": classification.macro_f1,
                "precision_at_10": retrieval.precision_at_k,
                "map_at_10": retrieval.map_at_k,
                "retrieval_queries": query_records,
                "per_class": per_class,
                "confusion_matrix": classification.confusion_matrix.tolist(),
                "misclassifications": classification.misclassifications[:20],
                "model_settings": {
                    **classification.model_settings,
                    "epochs": epochs,
                    "alpha": model.alpha,
                    "learning_rate": model.learning_rate,
                    "shuffle": False,
                },
                "vocabulary_size": len(vectorizer.vocabulary_),
            }
        )
        searchers[mode] = searcher
        print(
            f"  {mode}: accuracy={classification.accuracy:.4f}, macro_f1={classification.macro_f1:.4f}, P@10={retrieval.precision_at_k:.4f}",
            flush=True,
        )
    validations["count_statistics"] = {
        "train_nnz": stages.counts._nnz,
        "empty_query_count": int(np.count_nonzero(np.diff(query_counts.indptr) == 0)),
    }
    return rows, searchers, validations, stages.counts
