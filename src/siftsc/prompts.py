"""Prompt templates used in the published small-model experiments."""

from __future__ import annotations

_EXAMPLE = (
    "Question: Natalia sold clips to 48 friends in April, then half as many in May. "
    "How many clips did she sell altogether?\n"
    "Let's think step by step. April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.\n"
    "The answer is 72.\n\n"
)


def math_prompt(question: str) -> str:
    """Format a math question with the paper's compact CoT cue."""

    return _EXAMPLE + f"Question: {question.strip()}\nLet's think step by step.\n"
