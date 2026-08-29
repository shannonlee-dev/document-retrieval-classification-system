import document_system.privacy as privacy_module
from document_system.privacy import (
    SNIPPET_LIMIT,
    PrivacyReport,
    SanitizationResult,
    is_safe_text,
    make_safe_snippet,
    sanitize_document,
)


def test_privacy_policy_has_no_fixed_topic_allowlist() -> None:
    assert not hasattr(privacy_module, "SAFE_TERMS")


def test_sanitize_document_preserves_general_vocabulary() -> None:
    assert sanitize_document("framework shading polygon rocket orbit").text == (
        "framework shading polygon rocket orbit"
    )


def test_sanitize_document_does_not_treat_proper_nouns_as_person_names() -> None:
    assert sanitize_document(
        "Open Graphics Consortium released Polygon Framework"
    ).text == (
        "open graphics consortium released polygon framework"
    )


def test_sanitize_document_redacts_explicit_identifiers_and_counts_them() -> None:
    text = (
        "shared alice@example.com at https://example.com from 192.168.1.9 "
        "or 2001:db8::1 or 010-1234-5678 with 123-45-6789 and version "
        "999.999.999.999 about polygon rendering"
    )

    result = sanitize_document(text)

    assert result.text == "shared at from or or with and version about polygon rendering"
    assert result.excluded_reason is None
    assert result.redactions == {
        "email": 1,
        "phone": 1,
        "url": 1,
        "ipv4": 1,
        "ipv6": 1,
    }


def test_sanitize_document_does_not_guess_free_form_names_or_addresses() -> None:
    name_result = sanitize_document("from: Alice Smith about polygon rendering")
    signature_result = sanitize_document("Thanks,\nAlice Smith")
    address_result = sanitize_document("send graphics notes to 123 Main Street")

    assert name_result.text == "from alice smith about polygon rendering"
    assert signature_result.text == "thanks alice smith"
    assert address_result.text == "send graphics notes to main street"
    assert name_result.excluded_reason is None
    assert signature_result.excluded_reason is None
    assert address_result.excluded_reason is None


def test_sanitize_document_does_not_guess_free_form_health_information() -> None:
    assert sanitize_document("rocket orbit patient treatment").text == (
        "rocket orbit patient treatment"
    )


def test_is_safe_text_accepts_only_sanitized_non_sensitive_text() -> None:
    assert is_safe_text("framework shading polygon")
    assert not is_safe_text("")
    assert not is_safe_text("alice@example.com")
    assert is_safe_text("patient treatment")
    assert not is_safe_text("Alice shared polygon rendering")
    assert is_safe_text("alice shared polygon rendering")


def test_make_safe_snippet_is_limited_without_cutting_a_word() -> None:
    text = " ".join(["graphics"] * 40)

    snippet = make_safe_snippet(text)

    assert len(snippet) <= SNIPPET_LIMIT
    assert snippet == snippet.strip()
    assert snippet.split()[-1] == "graphics"
    assert is_safe_text(snippet)


def test_make_safe_snippet_keeps_a_word_ending_at_the_limit() -> None:
    assert make_safe_snippet("hello test more", limit=10) == "hello test"


def test_make_safe_snippet_reapplies_structured_redaction() -> None:
    assert make_safe_snippet("graphics alice@example.com rendering") == (
        "graphics rendering"
    )


def test_privacy_report_aggregates_sanitization_results() -> None:
    report = PrivacyReport.from_sanitization_results(
        [
            SanitizationResult(
                text="space orbit",
                redactions={"email": 1, "phone": 0, "url": 0, "ipv4": 0, "ipv6": 0},
            ),
            SanitizationResult(
                text="",
                redactions={"email": 0, "phone": 1, "url": 0, "ipv4": 0, "ipv6": 0},
                excluded_reason="empty_after_sanitization",
            ),
        ],
        labels=[0, 1],
        target_names=("space", "baseball"),
    )

    assert report.to_dict() == {
        "documents_input": 2,
        "documents_retained": 1,
        "documents_dropped_after_sanitization": 1,
        "retention_rate": 0.5,
        "redactions": {"email": 1, "phone": 1, "url": 0, "ipv4": 0, "ipv6": 0},
        "category_counts": {
            "space": {"raw": 1, "retained": 1, "excluded": 0},
            "baseball": {"raw": 1, "retained": 0, "excluded": 1},
        },
    }
