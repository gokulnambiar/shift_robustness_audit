from __future__ import annotations

import math
import re
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")
MIN_TEXT_LENGTH = 20


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def join_review_fields(summary: Any, review_text: Any) -> str:
    parts: list[str] = []
    for value in (summary, review_text):
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            parts.append(text)
    return normalize_text(" ".join(parts))


def derive_label(rating: Any) -> int | None:
    if rating is None:
        return None

    try:
        numeric_rating = float(rating)
    except (TypeError, ValueError):
        return None

    if math.isnan(numeric_rating):
        return None
    if numeric_rating >= 4.0:
        return 1
    if numeric_rating <= 2.0:
        return 0
    return None


def is_usable_text(text: str) -> bool:
    return len(normalize_text(text)) >= MIN_TEXT_LENGTH
