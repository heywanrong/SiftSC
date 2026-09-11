from siftsc.backends import _clean_completion_text, _completion_stop_position


def test_completion_stops_before_a_new_model_turn() -> None:
    text = "The answer is 25.<|endoftext|>Human: another question"
    position = _completion_stop_position(text, ("<|endoftext|>",))

    assert position is not None
    assert text[:position] == "The answer is 25."


def test_completion_cleans_text_after_a_finished_answer_without_early_stop() -> None:
    text = "Reasoning complete. Answer: 25.\n\nAnswer: 25."

    assert _completion_stop_position(text, ()) is None
    assert _clean_completion_text(text) == "Reasoning complete. Answer: 25."


def test_completion_does_not_stop_during_reasoning() -> None:
    assert _completion_stop_position("First calculate 60 - 15.", ()) is None
