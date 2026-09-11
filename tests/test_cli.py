from __future__ import annotations

import json
from dataclasses import dataclass, field

import siftsc.cli as cli
from siftsc.cli import main
from siftsc.types import Generation


@dataclass
class FakeMLXBackend:
    model_path: str
    use_chat_template: bool = False
    revision: str | None = None
    answers: list[str] = field(
        default_factory=lambda: [
            "15",
            "20",
            "15",
            "25",
            "25",
            "30",
            "109",
            "109",
            "99",
            "98",
            "100",
            "100",
            "17",
        ]
    )
    calls: int = 0

    def load(self) -> None:
        return None

    def generate(self, prompt: str, **kwargs: object) -> Generation:
        logits = ((0.0, 0.0),) if self.calls < 6 else ((10.0, 0.0),)
        answer = self.answers[self.calls]
        self.calls += 1
        return Generation(
            text=f"The answer is {answer}.",
            token_ids=(1,),
            top_logits=logits,
            entropies=(0.5,),
            sum_logprob=-float(self.calls),
        )


def test_profiles_command_lists_bundled_profiles(capsys) -> None:
    assert main(["profiles"]) == 0
    output = capsys.readouterr().out
    assert "qwen05b-q4-confidence" in output


def test_demo_reproduces_plain_wrong_siftsc_right(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "MLXBackend", FakeMLXBackend)
    assert main(["demo", "--model", "fake-model"]) == 0
    captured = capsys.readouterr()
    assert "PLAIN    15  ✗" in captured.out
    assert "SIFTSC   25  ✓" in captured.out
    assert "20 · 15 · 25 · 25 · 30" in captured.out
    assert "ALWAYS-SC   100  ✗" in captured.out
    assert "SIFTSC      109  ✓" in captured.out
    assert "79.2% less compute" in captured.out


def test_plain_json_uses_default_hugging_face_model(monkeypatch, capsys) -> None:
    created: list[FakeMLXBackend] = []

    def factory(
        model_path: str, use_chat_template: bool = False, revision: str | None = None
    ) -> FakeMLXBackend:
        backend = FakeMLXBackend(model_path, use_chat_template, revision, answers=["12"])
        created.append(backend)
        return backend

    monkeypatch.setattr(cli, "MLXBackend", factory)
    assert main(["ask", "What is 7 + 5?", "--mode", "plain", "--json"]) == 0
    captured = capsys.readouterr()
    assert created[0].model_path == cli.DEFAULT_MODEL
    payload = json.loads(captured.out)
    assert payload["parsed_answer"] == "12"
    assert payload["compute_comparison"] == {
        "actual_passes": 1,
        "always_sc_passes": 5,
        "passes_saved": 4,
        "compute_reduction": 0.8,
    }
    assert "Loading mlx-community" in captured.err


def test_chat_menu_and_mode_switch(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"])
    replies = iter(["1", "What is 7 + 5?", "/siftsc", "/exit"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    assert main(["chat", "--model", "fake-model"]) == 0
    output = capsys.readouterr().out
    assert "Choose how the model should answer" in output
    assert "[plain] passes=1" in output
    assert "saved=4 (80.0%)" in output
    assert "[siftsc] mode=siftsc" in output
