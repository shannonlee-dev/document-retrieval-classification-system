import pytest

from document_system.dataset import validate_dataset


def test_validate_dataset_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_dataset(["one", "two"], [0], minimum_documents=2)


def test_validate_dataset_requires_two_labels() -> None:
    with pytest.raises(ValueError, match="two labels"):
        validate_dataset(["one", "two"], [0, 0], minimum_documents=2)


def test_validate_dataset_rejects_missing_or_non_string_text() -> None:
    with pytest.raises(ValueError, match="strings"):
        validate_dataset(["one", None], [0, 1], minimum_documents=2)  # type: ignore[list-item]


def test_validate_dataset_accepts_valid_input() -> None:
    validate_dataset(["one", ""], [0, 1], minimum_documents=2)
