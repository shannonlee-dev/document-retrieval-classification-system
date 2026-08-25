import pytest

from document_system.preprocessing import EnglishPreprocessor


def test_tokenize_normalizes_and_filters() -> None:
    processor = EnglishPreprocessor(stop_words=frozenset({"the", "and"}))

    assert processor.tokenize("The QUICK, brown fox 123 and x!") == [
        "quick",
        "brown",
        "fox",
    ]


def test_tokenize_rejects_non_string_text() -> None:
    processor = EnglishPreprocessor()

    with pytest.raises(TypeError, match="text must be a string"):
        processor.tokenize(None)  # type: ignore[arg-type]
