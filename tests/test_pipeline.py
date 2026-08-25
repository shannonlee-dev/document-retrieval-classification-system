import json
from pathlib import Path

import numpy as np

import document_system.pipeline as pipeline_module
from document_system.dataset import DatasetBundle
from document_system.pipeline import BuildConfig, build_from_dataset, build_project
from document_system.privacy import is_safe_text


def test_small_pipeline_writes_consistent_report_and_runtime_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    texts = (
        "space shuttle orbit mission",
        "rocket moon planet space",
        "astronaut orbit spacecraft",
        "baseball pitcher inning game",
        "hitter baseball stadium team",
        "pitcher season baseball",
        "space telescope planet",
        "baseball game score",
        "rocket launch orbit",
        "team pitcher hitter",
    )
    source_doc_ids = np.arange(100, 110, dtype=np.int32)
    bundle = DatasetBundle(
        texts=texts,
        labels=np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
        source_doc_ids=source_doc_ids,
    )
    config = BuildConfig(
        runtime_dir=tmp_path / "runtime",
        reports_dir=tmp_path / "reports",
        batch_size=2,
        epochs=2,
        search_queries=("space orbit",),
    )

    evaluated_document_ids = None
    evaluate_classifier = pipeline_module.evaluate_classifier

    def record_document_ids(*args, **kwargs):
        nonlocal evaluated_document_ids
        evaluated_document_ids = np.asarray(kwargs["document_ids"])
        return evaluate_classifier(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "evaluate_classifier", record_document_ids)

    report = build_from_dataset(bundle, config)

    assert report.document_count == 10
    assert report.train_count + report.test_count == 10
    assert report.validation_passed is True
    validation = json.loads(
        (config.reports_dir / "tfidf_validation.json").read_text(encoding="utf-8")
    )
    assert validation["max_absolute_error"] <= 1e-6
    assert (config.reports_dir / "confusion_matrix.png").stat().st_size > 0
    assert (config.runtime_dir / "matrix.npz").is_file()
    assert evaluated_document_ids is not None
    assert set(evaluated_document_ids) <= set(source_doc_ids)

    metadata = json.loads(
        (config.runtime_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert "texts" not in metadata
    assert metadata["document_ids"] == source_doc_ids.tolist()
    assert all(is_safe_text(text) for text in metadata["snippets"])

    search_examples = json.loads(
        (config.reports_dir / "search_examples.json").read_text(encoding="utf-8")
    )
    assert all(
        is_safe_text(result["text_snippet"])
        for example in search_examples
        for result in example["results"]
    )
    misclassifications = json.loads(
        (config.reports_dir / "misclassifications.json").read_text(encoding="utf-8")
    )
    assert all(
        is_safe_text(item["text_snippet"]) for item in misclassifications
    )


def test_build_project_accepts_sanitized_dataset(monkeypatch, tmp_path: Path) -> None:
    texts = tuple(
        ["image pixel graphics"] * 200
        + ["baseball pitcher game"] * 200
        + ["space rocket orbit"] * 200
    )
    bundle = DatasetBundle(
        texts=texts,
        labels=np.array([0] * 200 + [1] * 200 + [2] * 200, dtype=np.int32),
        target_names=("comp.graphics", "rec.sport.baseball", "sci.space"),
        source_doc_ids=np.arange(len(texts), dtype=np.int32),
    )
    monkeypatch.setattr(pipeline_module, "load_20newsgroups", lambda: bundle)

    report = build_project(
        BuildConfig(
            runtime_dir=tmp_path / "runtime",
            reports_dir=tmp_path / "reports",
            batch_size=32,
            epochs=1,
        )
    )

    assert report.document_count == 600
    assert report.category_count == 3
