"""Answer normalization and plurality voting."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

from .types import Generation

AnswerParser = Callable[[str], str | None]

_ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}"),
    re.compile(
        r"(?:the\s+answer\s+is|final\s+answer\s*[:=]|answer\s*[:=])\s*\$?\s*([^\n$]+)",
        re.I,
    ),
    re.compile(r"####\s*([^\n]+)"),
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def parse_answer(text: str) -> str | None:
    """Extract a short final answer, with a generic text fallback."""

    cleaned = text.strip()
    if not cleaned:
        return None
    for pattern in _ANSWER_PATTERNS:
        matches = list(pattern.finditer(cleaned))
        if matches:
            return str(matches[-1].group(1)).strip().rstrip(". ")
    numbers = _NUMBER.findall(cleaned)
    if numbers:
        return cast(str, numbers[-1])
    final_line = next((line.strip() for line in reversed(cleaned.splitlines()) if line.strip()), "")
    return final_line[:256] or None


def plurality_vote(
    generations: Sequence[Generation], parser: AnswerParser = parse_answer
) -> tuple[Generation, str | None, dict[str, int]]:
    """Vote on parsed answers; break ties with the best sequence log-probability."""

    if not generations:
        raise ValueError("plurality_vote requires at least one generation")
    parsed = [parser(item.text) for item in generations]
    counts = Counter(answer for answer in parsed if answer is not None)
    if not counts:
        return generations[0], None, {}
    winning_count = max(counts.values())
    candidates = {answer for answer, count in counts.items() if count == winning_count}
    winner_index = max(
        (index for index, answer in enumerate(parsed) if answer in candidates),
        key=lambda index: generations[index].sum_logprob,
    )
    return generations[winner_index], parsed[winner_index], dict(counts)
