from __future__ import annotations

from siftsc.types import Generation
from siftsc.voting import parse_answer, plurality_vote


def test_parse_answer_cascade() -> None:
    assert parse_answer("work\n\\boxed{42}") == "42"
    assert parse_answer("Therefore, the answer is 7.5.") == "7.5"
    assert parse_answer("No marker here, but 19") == "19"


def test_plurality_vote_uses_logprob_for_tie() -> None:
    generations = (
        Generation("The answer is 2.", sum_logprob=-8.0),
        Generation("The answer is 3.", sum_logprob=-2.0),
    )
    winner, answer, counts = plurality_vote(generations)
    assert winner is generations[1]
    assert answer == "3"
    assert counts == {"2": 1, "3": 1}
