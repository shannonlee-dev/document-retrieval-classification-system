from document_system.privacy import is_safe_text, sanitize_text


def test_sanitize_text_keeps_only_safe_generic_terms() -> None:
    text = "Alice alice@example.com called 010-1234-5678 about treatment. Rocket orbit!"

    assert sanitize_text(text) == "rocket orbit"
    assert is_safe_text("rocket orbit")
    assert not is_safe_text("rocket alice")
