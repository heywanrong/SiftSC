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


def test_metered_backend_records_seconds_and_tokens() -> None:
    from siftsc.backends import MeteredBackend
    from siftsc.types import Generation

    class Fake:
        def generate(self, prompt: str, **kwargs: object) -> Generation:
            return Generation(text="The answer is 4.", token_ids=(1, 2, 3))

    metered = MeteredBackend(Fake())
    generation = metered.generate(
        "2 + 2", greedy=True, seed=0, max_tokens=8, temperature=0.0, top_p=1.0
    )

    assert generation.text == "The answer is 4."
    costs = metered.take()
    assert len(costs) == 1
    assert costs[0].tokens == 3
    assert costs[0].seconds >= 0.0
    assert metered.take() == []


def test_render_chat_prompt_prefers_the_tokenizer_template() -> None:
    from siftsc.backends import render_chat_prompt

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # type: ignore[no-untyped-def]
            assert messages[0]["role"] == "system"
            assert messages[-1] == {"role": "user", "content": "Who are you?"}
            assert tokenize is False and add_generation_prompt is True
            return "<templated>"

    assert render_chat_prompt(Tokenizer(), "Who are you?") == "<templated>"


def test_render_chat_prompt_falls_back_without_a_template() -> None:
    from siftsc.backends import render_chat_prompt

    prompt = render_chat_prompt(object(), "Who are you?")

    assert prompt.endswith("User: Who are you?\nAssistant:")


def test_completion_stops_at_the_chat_turn_end_marker() -> None:
    from siftsc.backends import MLXBackend

    text = "中国的首都是北京。<|im_end|>"
    position = _completion_stop_position(text, MLXBackend("x").stop_sequences)

    assert position is not None
    assert text[:position] == "中国的首都是北京。"
