import json

import numpy as np
import pytest

from document_system.artifacts import (
    load_retrievers,
    load_search_artifacts,
    save_search_artifacts,
)
from document_system.privacy import make_safe_snippet
from document_system.tfidf import NumpyTfidfVectorizer


def _save_full(path):
    texts = ["space space orbit", "baseball pitcher", "space rocket"]
    vectorizer = NumpyTfidfVectorizer()
    stages = vectorizer.fit_transform_with_stages(texts)
    save_search_artifacts(
        path,
        vectorizer,
        stages.tfidf,
        [make_safe_snippet(t) for t in texts],
        np.array([0, 1, 0]),
        ("space", "sport"),
        np.array([9, 7, 3]),
        counts=stages.counts,
        document_lengths=np.array([3, 2, 2]),
    )
    return texts


def test_full_index_restores_all_methods_without_fit(tmp_path, monkeypatch):
    texts = _save_full(tmp_path)
    expected = {
        mode: NumpyTfidfVectorizer(tf_mode=mode).fit_transform(texts).data
        for mode in ["raw", "log", "double"]
    }
    monkeypatch.setattr(
        NumpyTfidfVectorizer, "fit_transform", lambda *_: pytest.fail("refit")
    )
    restored = load_retrievers(tmp_path)
    for mode, searcher in restored.tfidf.items():
        assert searcher.vectorizer.tf_mode == mode
        np.testing.assert_allclose(searcher.matrix.data, expected[mode], atol=1e-12)
        assert searcher.search("space", 3, strategy="scan") == searcher.search(
            "space", 3, strategy="inverted"
        )
    assert restored.bm25.search("baseball", 1)[0].doc_id == 7
    again = load_retrievers(tmp_path)
    assert again.bm25.search("space", 3) == restored.bm25.search("space", 3)
    assert restored.corpus_fingerprint


def test_single_variant_restores_tf_mode(tmp_path):
    texts = ["space space orbit", "space rocket"]
    vectorizer = NumpyTfidfVectorizer(tf_mode="log")
    matrix = vectorizer.fit_transform(texts)
    save_search_artifacts(
        tmp_path,
        vectorizer,
        matrix,
        texts,
        np.array([0, 0]),
        ("space",),
        np.array([42, 3]),
    )
    restored = load_search_artifacts(tmp_path)
    assert restored.vectorizer.tf_mode == "log"
    np.testing.assert_allclose(
        restored.vectorizer.transform([texts[0]]).data, matrix.get_sparse_row(0)[1]
    )


def test_rejects_swapped_arrays_and_edited_corpus_metadata(tmp_path):
    _save_full(tmp_path)
    arrays = tmp_path / "search_index_arrays.npz"
    contents = bytearray(arrays.read_bytes())
    contents[100] ^= 1
    arrays.write_bytes(contents)
    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)
    _save_full(tmp_path)
    metadata_path = tmp_path / "search_index_data.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["document_ids"][0] = 1234
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rebuild"):
        load_retrievers(tmp_path)


def test_bm25_roundtrip_restores_non_default_parameters_and_scores(tmp_path):
    from document_system.bm25 import NumpyBM25Retriever

    texts = ["alpha alpha beta", "beta gamma", "gamma"]
    labels, ids, names = np.array([0, 1, 1]), np.array([90, 3, 42]), ("first", "second")
    vectorizer = NumpyTfidfVectorizer()
    stages = vectorizer.fit_transform_with_stages(texts)
    lengths = np.array([3, 2, 1])
    original = NumpyBM25Retriever(
        vectorizer, stages.counts, texts, labels, names, ids, lengths, k1=2.2, b=0.3
    )
    save_search_artifacts(
        tmp_path,
        vectorizer,
        stages.tfidf,
        texts,
        labels,
        names,
        ids,
        counts=stages.counts,
        document_lengths=lengths,
        k1=2.2,
        b=0.3,
    )
    restored = load_retrievers(tmp_path).bm25
    assert restored.k1 == 2.2 and restored.b == 0.3
    for query in ["alpha", "gamma beta", "beta beta gamma"]:
        assert restored.search(query, 3) == original.search(query, 3)
