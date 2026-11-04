"""Input-level safety checks shared by the dashboard's conversational tools."""

from __future__ import annotations

import re


READ_ONLY_MESSAGE = (
    "This dashboard is read-only. It cannot create, modify, delete, or grant access "
    "to data. Please ask an analytical question about the available Zomato data instead."
)

INSTRUCTION_OVERRIDE_MESSAGE = (
    "This request cannot be processed because it attempts to override application "
    "instructions. Please ask a question about the available Zomato data."
)

_SQL_WRITE_PATTERN = re.compile(
    r"\b(?:delete\s+from|drop\s+(?:table|database|schema)|truncate\s+table|"
    r"insert\s+into|update\s+\w+\s+set|alter\s+(?:table|database|schema)|"
    r"create\s+(?:table|database|schema)|grant\s+|revoke\s+)\b",
    re.IGNORECASE,
)
_MUTATION_INTENT_PATTERN = re.compile(
    r"\b(?:delete|remove|drop|truncate|erase|purge|destroy|insert|update|alter|"
    r"modify|overwrite|replace|grant|revoke)\b"
    r"(?:\s+\w+){0,5}\s+\b(?:record|records|row|rows|data|table|tables|database|"
    r"schema|review|reviews|order|orders|customer|customers|city|cities|user|users|"
    r"permission|permissions|access)\b",
    re.IGNORECASE,
)
_INSTRUCTION_OVERRIDE_PATTERN = re.compile(
    r"\b(?:ignore|disregard|override|forget|bypass)\b"
    r"(?:\s+\w+){0,4}\s+\b(?:instruction|instructions|rule|rules|guardrail|guardrails|"
    r"system\s+prompt|policy|policies)\b|"
    r"\b(?:reveal|show|print|repeat)\b(?:\s+\w+){0,3}\s+\b(?:system\s+prompt|"
    r"hidden\s+instruction|internal\s+instruction)\b",
    re.IGNORECASE,
)


def blocked_input_message(question: str) -> str | None:
    """Return an explanation when a prompt should not be sent to an LLM."""
    if not isinstance(question, str):
        return READ_ONLY_MESSAGE

    normalized = " ".join(question.split())
    if _INSTRUCTION_OVERRIDE_PATTERN.search(normalized):
        return INSTRUCTION_OVERRIDE_MESSAGE
    if _SQL_WRITE_PATTERN.search(normalized) or _MUTATION_INTENT_PATTERN.search(normalized):
        return READ_ONLY_MESSAGE
    return None
