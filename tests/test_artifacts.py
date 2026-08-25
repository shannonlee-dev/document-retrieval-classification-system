import json
from pathlib import Path

import numpy as np
import pytest

from document_system.artifacts import load_search_artifacts, save_search_artifacts
from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer


def make_search_data():
    texts = ["space shuttle orbit", "baseball pitcher game"]
    labels = np.array([0, 1], dtype=np.int32)
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, texts, labels, ("space", "baseball")


def test_search_artifacts_round_trip(tmp_path: Path) -> None:
    vectorizer, matrix, texts, labels, target_names = make_search_data()

    save_search_artifacts(
        tmp_path, vectorizer, matrix, texts, labels, target_names
    )
    restored = load_search_artifacts(tmp_path)

    assert restored.matrix.shape == matrix.shape
    np.testing.assert_array_equal(restored.matrix.data, matrix.data)
    assert restored.vectorizer.vocabulary_ == vectorizer.vocabulary_
    np.testing.assert_array_equal(restored.vectorizer.idf_, vectorizer.idf_)
    assert restored.texts == tuple(texts)
    np.testing.assert_array_equal(restored.labels, labels)
    assert restored.target_names == target_names


def test_search_artifacts_reject_unknown_version(tmp_path: Path) -> None:
    vectorizer, matrix, texts, labels, target_names = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, texts, labels, target_names
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_version"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)


def test_search_artifacts_require_all_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python main.py build"):
        load_search_artifacts(tmp_path)
