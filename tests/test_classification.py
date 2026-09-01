from pathlib import Path

import numpy as np

import document_system.classification.model as model_module
from document_system.classification.evaluation import evaluate_classifier
from document_system.classification.model import predict_sparse, train_linear_svm
from document_system.classification.visualization import save_confusion_matrix
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


def test_training_and_prediction_use_bounded_numpy_batches(monkeypatch) -> None:
    class RecordingClassifier:
        def __init__(self, **settings) -> None:
            self.random_state = settings["random_state"]
            self.training_batches: list[np.ndarray] = []
            self.prediction_batches: list[np.ndarray] = []

        def partial_fit(self, features, labels, classes=None):
            self.training_batches.append(features)
            return self

        def predict(self, features):
            self.prediction_batches.append(features)
            return np.zeros(features.shape[0], dtype=np.int32)

    monkeypatch.setattr(model_module, "SGDClassifier", RecordingClassifier)
    _, labels, matrix = make_training_data()

    model = train_linear_svm(matrix, labels, batch_size=2, epochs=1)
    predict_sparse(model, matrix, batch_size=2)

    batches = model.training_batches + model.prediction_batches
    assert batches
    assert all(isinstance(batch, np.ndarray) for batch in batches)
    assert all(batch.shape[0] <= 2 for batch in batches)


def test_classification_has_no_direct_scipy_dependency() -> None:
    package = Path("src/document_system/classification")
    source = "\n".join(path.read_text() for path in package.glob("*.py"))

    assert "scipy" not in source
    assert "csr_matrix" not in source


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
    assert report.model_settings["input_representation"] == "NumPy dense batches"
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
