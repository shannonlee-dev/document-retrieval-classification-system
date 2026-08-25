from pathlib import Path

import numpy as np

from document_system.classification import (
    evaluate_classifier,
    predict_sparse,
    save_confusion_matrix,
    train_linear_svm,
)
from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer


def make_training_data():
    texts = [
        "space orbit shuttle",
        "planet rocket orbit",
        "spacecraft moon mission",
        "baseball pitcher inning",
        "baseball hitter game",
        "pitcher team stadium",
    ]
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    return texts, labels, vectorizer.fit_transform(texts)


def test_linear_svm_trains_from_sparse_batches_reproducibly() -> None:
    _, labels, matrix = make_training_data()

    first = train_linear_svm(
        matrix, labels, batch_size=2, epochs=6, random_state=42
    )
    second = train_linear_svm(
        matrix, labels, batch_size=2, epochs=6, random_state=42
    )

    np.testing.assert_array_equal(
        predict_sparse(first, matrix, batch_size=2),
        predict_sparse(second, matrix, batch_size=2),
    )


def test_evaluation_contains_metrics_confusion_and_error_records() -> None:
    texts, labels, matrix = make_training_data()
    model = train_linear_svm(matrix, labels, batch_size=2, epochs=6)

    report = evaluate_classifier(
        model,
        matrix,
        labels,
        texts,
        ("space", "baseball"),
        batch_size=2,
    )

    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion_matrix.shape == (2, 2)
    assert report.predictions.shape == labels.shape
    assert all(
        {"doc_id", "actual", "predicted", "text_snippet"} <= item.keys()
        for item in report.misclassifications
    )


def test_save_confusion_matrix_creates_nonempty_png(tmp_path: Path) -> None:
    texts, labels, matrix = make_training_data()
    model = train_linear_svm(matrix, labels, batch_size=2, epochs=2)
    report = evaluate_classifier(
        model, matrix, labels, texts, ("space", "baseball"), batch_size=2
    )
    output = tmp_path / "confusion.png"

    save_confusion_matrix(report, ("space", "baseball"), output)

    assert output.read_bytes().startswith(b"\x89PNG")


def test_training_rejects_invalid_batch_configuration() -> None:
    _, labels, matrix = make_training_data()

    for batch_size, epochs in [(0, 1), (2, 0)]:
        try:
            train_linear_svm(matrix, labels, batch_size=batch_size, epochs=epochs)
        except ValueError as error:
            assert "positive" in str(error)
        else:
            raise AssertionError("invalid training configuration was accepted")
