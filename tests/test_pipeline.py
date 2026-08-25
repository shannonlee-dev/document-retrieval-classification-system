import json
from pathlib import Path

import numpy as np

import document_system.pipeline as pipeline_module
from document_system.dataset import DatasetBundle
from document_system.pipeline import BuildConfig, build_from_dataset, build_project


def test_small_pipeline_writes_consistent_report_and_runtime_artifacts(
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
    bundle = DatasetBundle(
        texts=texts,
        labels=np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
        source_doc_ids=np.arange(len(texts), dtype=np.int32),
    )
    config = BuildConfig(
        runtime_dir=tmp_path / "runtime",
        reports_dir=tmp_path / "reports",
        batch_size=2,
        epochs=2,
        search_queries=("space orbit",),
    )

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
