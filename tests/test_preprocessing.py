import pytest

from document_system.preprocessing import EnglishPreprocessor


def test_tokenize_normalizes_and_filters() -> None:
    processor = EnglishPreprocessor(stop_words=frozenset({"the", "and"}))

    assert processor.tokenize("The QUICK, brown fox 123 and x!") == [
        "quick",
        "brown",
        "fox",
    ]


def test_whitespace_tokenizer_cleans_then_splits() -> None:
    processor = EnglishPreprocessor(stop_words=frozenset({"and"}))

    assert processor.tokenize("Image, PIXEL and orbit-rocket!") == [
        "image",
        "pixel",
        "orbit",
        "rocket",
    ]


def test_tokenize_rejects_non_string_text() -> None:
    processor = EnglishPreprocessor()

    with pytest.raises(TypeError, match="text must be a string"):
        processor.tokenize(None)  # type: ignore[arg-type]
