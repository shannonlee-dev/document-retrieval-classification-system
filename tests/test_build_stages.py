import numpy as np
import pytest

from document_system import build_stages
from document_system.dataset import DatasetBundle
from document_system.privacy import PrivacyReport, make_safe_snippet, sanitize_document
from document_system.tfidf import NumpyTfidfVectorizer


@pytest.fixture
def small_bundle() -> DatasetBundle:
    texts = (
        "space orbit rocket",
        "space shuttle mission",
        "space planet telescope",
        "space astronaut launch",
        "baseball pitcher inning",
        "baseball hitter stadium",
        "baseball game season",
        "baseball team score",
        "space heldoutonly satellite",
        "baseball heldoutonly catcher",
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int32)
    target_names = ("space", "baseball")
    return DatasetBundle(
        texts=texts,
        labels=labels,
        target_names=target_names,
        source_doc_ids=np.arange(10, dtype=np.int32),
        privacy_report=PrivacyReport.from_sanitization_results(
            [sanitize_document(text) for text in texts], labels, target_names
        ),
    )


def fixed_split(*_args, **_kwargs):
    return np.arange(8), np.arange(8, 10)


def test_classification_and_search_use_independent_fit_scopes(
    monkeypatch, small_bundle
) -> None:
    created_vectorizers = []

    class RecordingVectorizer(NumpyTfidfVectorizer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created_vectorizers.append(self)

    monkeypatch.setattr(build_stages, "train_test_split", fixed_split)
    monkeypatch.setattr(
        build_stages, "NumpyTfidfVectorizer", RecordingVectorizer
    )
    snippets = tuple(make_safe_snippet(text) for text in small_bundle.texts)

    classification = build_stages.run_classification_stage(
        small_bundle,
        snippets,
        batch_size=2,
        epochs=1,
        random_state=42,
    )
    search = build_stages.run_search_stage(
        small_bundle,
        snippets,
        ("space orbit",),
    )

    classification_vectorizer, search_vectorizer = created_vectorizers
    assert classification_vectorizer is not search_vectorizer
    assert "heldoutonly" not in classification_vectorizer.vocabulary_
    assert "heldoutonly" in search_vectorizer.vocabulary_
    assert classification.validation.shape[0] == 8
    assert search.validation.shape[0] == 10
    assert search.matrix.shape[0] == 10
