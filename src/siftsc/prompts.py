"""Prompt templates and the question router used by the CLI.

The paper's compact chain-of-thought prompt is kept byte-for-byte for reasoning
questions, because the bundled gates were calibrated on it. Questions that do not
look like reasoning problems are answered once through the model's own chat
template instead, so a 0.5B model is not forced to invent numbers.
"""

from __future__ import annotations

import re

_EXAMPLE = (
    "Question: Natalia sold clips to 48 friends in April, then half as many in May. "
    "How many clips did she sell altogether?\n"
    "Let's think step by step. April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.\n"
    "The answer is 72.\n\n"
)

GENERAL_SYSTEM_PROMPT = (
    "You are Sifty, a small local assistant. Answer briefly and directly, "
    "in the same language as the question."
)

# Signals that a question expects a checkable, usually numeric, answer.
_NUMBER_WORDS = frozenset(
    {
        "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred", "thousand", "million", "billion",
        "half", "quarter", "double", "twice", "triple", "dozen",
    }
)  # fmt: skip
_MATH_PHRASES = (
    "how many",
    "how much",
    "how far",
    "how long",
    "how old",
    "how fast",
    "total",
    "sum of",
    "average",
    "ratio",
    "percent",
    "probability",
    "calculate",
    "compute",
    "solve",
    "equation",
    "remainder",
    "altogether",
    "in all",
    "left over",
    "times as",
)
_MATH_PHRASES_CJK = (
    "多少",
    "几个",
    "几只",
    "几人",
    "几天",
    "几小时",
    "几分钟",
    "几倍",
    "几岁",
    "计算",
    "求解",
    "求出",
    "一共",
    "总共",
    "平均",
    "百分",
    "比例",
    "概率",
    "面积",
    "周长",
    "体积",
    "利润",
    "折扣",
    "等于",
    "余数",
    "方程",
    "算一下",
    "算出",
    "还剩",
)
# Operators between numbers, or plus, multiplication, division, equals, percent,
# power, and root signs anywhere.
_MATH_SYMBOLS = re.compile("[+\u00d7\u00f7=%^\u221a]|\\d\\s*[-*/]\\s*\\d")
_DIGIT = re.compile(r"\d")
_WORD = re.compile(r"[a-z]+")


def math_prompt(question: str) -> str:
    """Format a math question with the paper's compact CoT cue."""

    return _EXAMPLE + f"Question: {question.strip()}\nLet's think step by step.\n"


def looks_like_math(question: str) -> bool:
    """Return whether a question should take the paper's reasoning route."""

    text = question.strip().lower()
    if not text:
        return False
    if _DIGIT.search(text) or _MATH_SYMBOLS.search(text):
        return True
    if set(_WORD.findall(text)) & _NUMBER_WORDS:
        return True
    if any(phrase in text for phrase in _MATH_PHRASES):
        return True
    return any(phrase in question for phrase in _MATH_PHRASES_CJK)


def fallback_chat_prompt(question: str, system: str = GENERAL_SYSTEM_PROMPT) -> str:
    """Plain chat prompt for tokenizers without a chat template."""

    return f"{system}\n\nUser: {question.strip()}\nAssistant:"
