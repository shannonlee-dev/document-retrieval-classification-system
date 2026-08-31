import json
from pathlib import Path

import numpy as np
import pytest

from document_system.artifacts import load_search_artifacts, save_search_artifacts
from document_system.preprocessing import EnglishPreprocessor
from document_system.privacy import SNIPPET_LIMIT
from document_system.tfidf import NumpyTfidfVectorizer


def make_search_data():
    snippets = ["space shuttle orbit", "baseball pitcher game"]
    document_ids = np.array([42, 99], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int32)
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    matrix = vectorizer.fit_transform(snippets)
    return vectorizer, matrix, snippets, labels, ("space", "baseball"), document_ids


def test_search_artifacts_round_trip(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()

    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    restored = load_search_artifacts(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert restored.matrix.shape == matrix.shape
    np.testing.assert_array_equal(restored.matrix.data, matrix.data)
    assert restored.vectorizer.vocabulary_ == vectorizer.vocabulary_
    np.testing.assert_array_equal(restored.vectorizer.idf_, vectorizer.idf_)
    assert restored.snippets == tuple(snippets)
    np.testing.assert_array_equal(restored.document_ids, document_ids)
    np.testing.assert_array_equal(restored.labels, labels)
    assert restored.target_names == target_names
    assert metadata["artifact_version"] == 2
    assert metadata["fit_scope"] == "full_corpus"
    assert metadata["fit_document_count"] == matrix.shape[0]
    assert metadata["category_count"] == len(target_names)


def test_search_artifacts_reject_unknown_version(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_version"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_version", 1),
        ("fit_scope", "train_split"),
        ("fit_document_count", 999),
        ("category_count", 999),
    ],
)
def test_search_artifacts_reject_incompatible_build_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


def test_search_artifacts_reject_labels_outside_target_names(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["labels"][0] = len(target_names)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


def test_search_artifacts_reject_save_row_count_mismatch(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()

    with pytest.raises(ValueError, match="row"):
        save_search_artifacts(
            tmp_path,
            vectorizer,
            matrix,
            snippets[:-1],
            labels,
            target_names,
            document_ids,
        )


def test_search_artifacts_store_sanitized_search_fields(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()

    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "texts" not in metadata
    assert metadata["snippets"] == list(snippets)
    assert metadata["document_ids"] == [42, 99]
    assert metadata["privacy_policy"] == "structured-pii-redaction-v3"


def test_search_artifacts_reject_unsafe_or_blank_snippets(tmp_path: Path) -> None:
    vectorizer, matrix, _, labels, target_names, document_ids = make_search_data()

    for snippets in (["space shuttle orbit", "Alice"], ["space shuttle orbit", " "]):
        with pytest.raises(ValueError, match="safe"):
            save_search_artifacts(
                tmp_path,
                vectorizer,
                matrix,
                snippets,
                labels,
                target_names,
                document_ids,
            )


def test_search_artifacts_reject_snippets_over_limit(tmp_path: Path) -> None:
    vectorizer, matrix, _, labels, target_names, document_ids = make_search_data()
    long_snippet = "x" * (SNIPPET_LIMIT + 1)

    with pytest.raises(ValueError, match="snippet"):
        save_search_artifacts(
            tmp_path,
            vectorizer,
            matrix,
            ["space shuttle orbit", long_snippet],
            labels,
            target_names,
            document_ids,
        )


def test_search_artifacts_reject_legacy_metadata(tmp_path: Path) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["privacy_policy"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


@pytest.mark.parametrize(
    "old_policy",
    ["safe-topic-terms-v1", "redaction-and-risk-screen-v2"],
)
def test_search_artifacts_reject_old_privacy_policy(
    tmp_path: Path,
    old_policy: str,
) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["privacy_policy"] = old_policy
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


def test_search_artifacts_require_all_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python main.py build"):
        load_search_artifacts(tmp_path)
