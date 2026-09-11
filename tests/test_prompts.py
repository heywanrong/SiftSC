from __future__ import annotations

import pytest

from siftsc.prompts import fallback_chat_prompt, looks_like_math, math_prompt


@pytest.mark.parametrize(
    "question",
    [
        "Henry made two stops during his 60-mile bike trip. How many miles between the stops?",
        "If 3 notebooks cost £4 each, what is the total?",
        "A shop sold eighteen books on Monday and twice as many on Tuesday. How many in total?",
        "What is half of a dozen?",
        "How many legs does a spider have?",
        "小明有三个苹果，吃了一个，还剩几个？",
        "一共有多少人参加了会议？",
        "What is 7 + 5?",
    ],
)
def test_reasoning_questions_take_the_math_route(question: str) -> None:
    assert looks_like_math(question)


@pytest.mark.parametrize(
    "question",
    [
        "中国的首都在哪",
        "Who are you?",
        "What is the capital of France?",
        "什么是机器学习？",
        "Which one is better, tea or coffee?",
        "Tell me a fun fact about penguins.",
        "",
        "   ",
    ],
)
def test_general_questions_take_the_chat_route(question: str) -> None:
    assert not looks_like_math(question)


def test_math_prompt_keeps_the_paper_example() -> None:
    assert "Natalia" in math_prompt("What is 7 + 5?")


def test_fallback_chat_prompt_names_the_assistant_and_question() -> None:
    prompt = fallback_chat_prompt("中国的首都在哪")

    assert "Sifty" in prompt
    assert prompt.endswith("User: 中国的首都在哪\nAssistant:")
