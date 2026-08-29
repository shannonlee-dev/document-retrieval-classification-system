"""Privacy-safe text boundary for bundled documents."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .constants import SNIPPET_LIMIT

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_CANDIDATE_RE = re.compile(
    r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
    r"(?![0-9A-Fa-f:])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})"
    r"[ .-]\d{3,4}[ .-]\d{4}|\d{3}[-.]\d{4})(?!\w)"
)

REDACTION_KINDS = (
    "email",
    "phone",
    "url",
    "ipv4",
    "ipv6",
)
REDACTION_REPLACEMENT = " "
MINIMUM_SNIPPET_LIMIT = 1


@dataclass(frozen=True)
class SanitizationResult:
    text: str
    redactions: dict[str, int]
    excluded_reason: str | None = None


@dataclass(frozen=True)
class PrivacyReport:
    raw_document_count: int
    retained_document_count: int
    redactions: dict[str, int]
    category_counts: dict[str, dict[str, int]]

    @property
    def dropped_after_sanitization(self) -> int:
        return self.raw_document_count - self.retained_document_count

    @property
    def retention_rate(self) -> float:
        if self.raw_document_count == 0:
            return 0.0
        return self.retained_document_count / self.raw_document_count

    @classmethod
    def from_sanitization_results(
        cls,
        results: Sequence[SanitizationResult],
        labels: Sequence[int],
        target_names: tuple[str, ...],
    ) -> PrivacyReport:
        """Build a report from per-document sanitization results."""

        if len(results) != len(labels):
            raise ValueError("privacy results and labels must have the same length")
        redactions = {kind: 0 for kind in REDACTION_KINDS}
        category_counts = {
            name: {"raw": 0, "retained": 0, "excluded": 0} for name in target_names
        }
        retained_count = 0
        for result, label in zip(results, labels, strict=True):
            category = target_names[int(label)]
            category_counts[category]["raw"] += 1
            for kind, count in result.redactions.items():
                redactions[kind] += count
            if result.excluded_reason is None:
                retained_count += 1
                category_counts[category]["retained"] += 1
                continue
            category_counts[category]["excluded"] += 1
        return cls(
            raw_document_count=len(results),
            retained_document_count=retained_count,
            redactions=redactions,
            category_counts=category_counts,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "documents_input": self.raw_document_count,
            "documents_retained": self.retained_document_count,
            "documents_dropped_after_sanitization": self.dropped_after_sanitization,
            "retention_rate": self.retention_rate,
            "redactions": dict(self.redactions),
            "category_counts": self.category_counts,
        }


def sanitize_document(text: str) -> SanitizationResult:
    """Redact deterministic structured identifiers and normalize text."""

    redactions = {kind: 0 for kind in REDACTION_KINDS}
    sanitized = text
    for kind, pattern in (
        ("email", _EMAIL_RE),
        ("url", _URL_RE),
    ):
        sanitized, redactions[kind] = pattern.subn(REDACTION_REPLACEMENT, sanitized)
    sanitized, redactions["ipv4"] = _redact_ipv4(sanitized)
    sanitized, redactions["ipv6"] = _redact_ipv6(sanitized)
    sanitized, redactions["phone"] = _PHONE_RE.subn(REDACTION_REPLACEMENT, sanitized)

    normalized = " ".join(re.findall(r"[a-z]+", sanitized.lower()))
    if not normalized:
        return SanitizationResult(
            text="",
            redactions=redactions,
            excluded_reason="empty_after_sanitization",
        )
    return SanitizationResult(text=normalized, redactions=redactions)


def is_safe_text(text: str) -> bool:
    """Return whether text matches the canonical sanitized-text contract."""

    if not isinstance(text, str):
        return False
    if any(
        pattern.search(text)
        for pattern in (
            _EMAIL_RE,
            _URL_RE,
            _IPV4_RE,
            _PHONE_RE,
        )
    ):
        return False
    if _contains_ipv6(text):
        return False
    tokens = re.findall(r"[a-z]+", text)
    return bool(tokens) and text == " ".join(tokens)


def make_safe_snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Re-sanitize text and create an artifact-safe word-boundary snippet."""

    if limit < MINIMUM_SNIPPET_LIMIT:
        raise ValueError("snippet limit must be positive")
    sanitized = sanitize_document(text).text
    if not sanitized:
        raise ValueError("snippet source must contain text after sanitization")
    if len(sanitized) <= limit:
        return sanitized
    prefix = sanitized[:limit]
    if sanitized[limit] == " ":
        return prefix
    return prefix.rsplit(" ", maxsplit=1)[0] or prefix


def _redact_ipv6(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        try:
            ipaddress.IPv6Address(match.group())
        except ValueError:
            return match.group()
        count += 1
        return REDACTION_REPLACEMENT

    return _IPV6_CANDIDATE_RE.sub(replace, text), count


def _redact_ipv4(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        try:
            ipaddress.IPv4Address(match.group())
        except ValueError:
            return match.group()
        count += 1
        return REDACTION_REPLACEMENT

    return _IPV4_RE.sub(replace, text), count


def _contains_ipv6(text: str) -> bool:
    _, count = _redact_ipv6(text)
    return count > 0
