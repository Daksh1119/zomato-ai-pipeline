"""Deterministic handlers for review-corpus questions that do not need RAG."""

from __future__ import annotations

import re

import pandas as pd


def requests_city_catalog(question: str) -> bool:
    """Return whether a question asks for the corpus-wide city list or count."""
    if not isinstance(question, str):
        return False

    normalized = " ".join(question.casefold().split())
    if not re.search(r"\bcities\b", normalized):
        return False

    return bool(
        re.search(r"\b(list|show|display|enumerate)\b.*\bcities\b", normalized)
        or re.search(r"\b(all|every)\s+cities\b", normalized)
        or re.search(r"\b(number|count|total)\s+(?:of\s+)?cities\b", normalized)
        or re.search(
            r"\bcities\s+(?:are\s+)?(?:covered|there|in the (?:data|dataset|corpus))\b",
            normalized,
        )
    )


def build_city_catalog(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create a complete, deterministic city catalogue with review counts."""
    if "city" not in reviews.columns:
        raise ValueError("Review data does not contain a city column.")

    cities = reviews["city"].dropna().astype(str).str.strip()
    cities = cities[cities.ne("")]
    catalog = cities.value_counts().rename_axis("city").reset_index(name="review_count")
    return catalog.sort_values("city", kind="stable").reset_index(drop=True)
