from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import siftsc.cli as cli
from siftsc.cli import main
from siftsc.types import Generation

DEMO_ANSWERS = ["15", "20", "15", "25", "25", "30", "109", "109", "99", "98", "100", "100", "17"]


@dataclass
class FakeMLXBackend:
    model_path: str
    use_chat_template: bool = False
    revision: str | None = None
    answers: list[str] = field(default_factory=lambda: list(DEMO_ANSWERS))
    fallback: str = "42"
    confident_after: int = 6
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def load(self) -> None:
        return None

    def generate(self, prompt: str, **kwargs: object) -> Generation:
        self.prompts.append(prompt)
        logits = ((0.0, 0.0),) if self.calls < self.confident_after else ((10.0, 0.0),)
        answer = self.answers[self.calls] if self.calls < len(self.answers) else self.fallback
        self.calls += 1
        return Generation(
            text=f"The answer is {answer}.",
            token_ids=(1, 2),
            top_logits=logits,
            entropies=(0.5,),
            sum_logprob=-float(self.calls),
        )


@dataclass
class FakeDownloadBackend(FakeMLXBackend):
    cached: bool = False
    downloads: int = 0

    def is_cached(self) -> bool:
        return self.cached

    def ensure_downloaded(self, progress: Callable[[int, int | None], None] | None = None) -> None:
        self.downloads += 1
        if progress is not None:
            progress(1_000_000, 2_000_000)


@dataclass
class InterruptingBackend(FakeMLXBackend):
    interrupted: bool = False

    def generate(self, prompt: str, **kwargs: object) -> Generation:
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return super().generate(prompt, **kwargs)


def _scripted_input(monkeypatch, replies: list[str]) -> None:
    iterator = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt: next(iterator))


def test_version_flag_reports_the_package_version(capsys) -> None:
    import pytest

    import siftsc

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"siftsc {siftsc.__version__}"


def test_profiles_command_lists_bundled_profiles(capsys) -> None:
    assert main(["profiles"]) == 0
    output = capsys.readouterr().out
    assert "qwen05b-q4-confidence" in output


def test_demo_reproduces_plain_wrong_siftsc_right(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "MLXBackend", FakeMLXBackend)
    assert main(["demo", "--model", "fake-model"]) == 0
    captured = capsys.readouterr()
    assert "PLAIN      15  ✗" in captured.out
    assert "SIFTSC     25  ✓" in captured.out
    assert "20 · 15 · 25 · 25 · 30" in captured.out
    assert "ALWAYS-SC  100  ✗" in captured.out
    assert "SIFTSC     109  ✓" in captured.out
    assert "2,000 passes" in captured.out
    assert "415 passes" in captured.out
    assert "79.2% less compute" in captured.out
    assert "Reproduced" in captured.out


def test_demo_warns_but_continues_to_chat_when_drifted(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["15", "15", "15", "15", "15", "15"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    _scripted_input(monkeypatch, ["/exit"])

    assert main(["demo", "--model", "fake-model"]) == 0
    captured = capsys.readouterr()
    assert "did not reproduce" in captured.out
    assert "YOUR TURN" in captured.out


def test_demo_fails_loudly_when_drifted_and_not_interactive(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["15", "15", "15", "15", "15", "15"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert main(["demo", "--model", "fake-model", "--no-chat"]) == 2
    captured = capsys.readouterr()
    assert "did not reproduce" in captured.out + captured.err


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
    assert payload["cost"]["tokens"] == 2
    assert "Waking up Sifty" in captured.err


def test_ask_prints_answer_and_compute_chart(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert main(["ask", "What is 7 + 5?", "--model", "fake-model"]) == 0
    output = capsys.readouterr().out
    assert "🎯 Answer: 12" in output
    assert "⚡ plain" in output
    assert "👥 always-SC" in output
    assert "🧭 siftsc" in output
    assert "saved 4 (80.0%)" in output


def test_ask_compare_mode_runs_all_three_policies(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], fallback="12", confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert main(["ask", "What is 7 + 5?", "--model", "fake-model", "--mode", "compare"]) == 0
    output = capsys.readouterr().out
    assert "🔬 COMPARE" in output
    assert backend.calls == 6


def test_chat_menu_and_mode_switch(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["1", "What is 7 + 5?", "/siftsc", "/exit"])

    assert main(["chat", "--model", "fake-model"]) == 0
    output = capsys.readouterr().out
    assert "Choose a reasoning mode" in output
    assert "🎯 Answer: 12" in output
    assert "⚡ plain" in output and "1 pass" in output
    assert "saved 4 (80.0%)" in output
    assert "Mode switched" in output and "siftsc" in output


def test_chat_shows_final_answer_route_and_session_line(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["15", "20", "15", "25", "25", "30"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(
        monkeypatch,
        ["Henry rode 60 miles and stopped twice; how many miles between the stops?", "/exit"],
    )

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert "🎯 Answer: 25" in output
    assert "vote called" in output
    assert "20 · 15 · 25 · 25 · 30" in output
    assert "the draft said 15" in output
    assert "extra 1 (+20.0%)" in output
    assert "📊 session · 1 question · 6 vs 5 passes" in output


def test_demo_flows_into_chat_and_answers_the_first_question(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model")
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    _scripted_input(monkeypatch, ["What is 7 + 5?", "/exit"])

    assert main(["demo", "--model", "fake-model"]) == 0
    output = capsys.readouterr().out
    assert "YOUR TURN" in output
    assert "🎯 Answer: 42" in output
    assert "Thanks for trying SiftSC" in output


def test_chat_compare_mode_reuses_voters_when_a_vote_was_called(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["15", "20", "15", "25", "25", "30"])
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(
        monkeypatch,
        ["Henry rode 60 miles and stopped twice; how many miles between the stops?", "/exit"],
    )

    assert main(["chat", "--model", "fake-model", "--mode", "compare"]) == 0
    output = capsys.readouterr().out
    assert "🔬 COMPARE" in output
    assert "→ 15" in output
    assert "→ 25" in output
    assert "vote called" in output
    assert "disagree" in output
    assert backend.calls == 6


def test_chat_compare_mode_draws_fresh_voters_when_vote_was_skipped(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], fallback="12", confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["What is 7 + 5?", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "compare"]) == 0
    output = capsys.readouterr().out
    assert "🔬 COMPARE" in output
    assert "vote skipped" in output
    assert "agree" in output
    assert backend.calls == 6


def test_chat_stats_command_prints_session_chart(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["/stats", "What is 7 + 5?", "/stats", "/help", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert "no questions yet" in output
    assert "📊 SESSION · 1 question" in output
    assert "/compare" in output


def test_chat_recovers_from_keyboard_interrupt_during_generation(monkeypatch, capsys) -> None:
    backend = InterruptingBackend("fake-model", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["What is 7 + 5?", "What is 7 + 5?", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert "Stopped" in output
    assert "🎯 Answer: 12" in output


def test_download_notice_only_when_model_is_not_cached(monkeypatch, capsys) -> None:
    backend = FakeDownloadBackend("mlx-community/fake", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert main(["ask", "What is 7 + 5?", "--model", "mlx-community/fake"]) == 0
    captured = capsys.readouterr()
    assert "Downloading" in captured.err
    assert backend.downloads == 1

    backend.cached = True
    assert main(["ask", "What is 7 + 5?", "--model", "mlx-community/fake"]) == 0
    captured = capsys.readouterr()
    assert "Downloading" not in captured.err
    assert backend.downloads == 1


def test_general_question_gets_one_chat_style_pass(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["Beijing"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["中国的首都在哪", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert backend.calls == 1
    assert "Natalia" not in backend.prompts[0]
    assert backend.prompts[0].endswith("User: 中国的首都在哪\nAssistant:")
    assert "💬 General question" in output
    assert "The answer is Beijing." in output
    assert "🎯 Answer" not in output
    assert "📊 session · 1 question · 1 vs 5 passes" in output


def test_math_prefix_forces_the_paper_route(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["/math Who are you?", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert "Natalia" in backend.prompts[0]
    assert "🎯 Answer: 12" in output


def test_talk_prefix_forces_a_chat_style_answer(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["/talk What is 7 + 5?", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "siftsc"]) == 0
    output = capsys.readouterr().out
    assert "Natalia" not in backend.prompts[0]
    assert "💬 General question" in output


def test_compare_mode_skips_voting_for_general_questions(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["Sifty"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)
    _scripted_input(monkeypatch, ["Who are you?", "/exit"])

    assert main(["chat", "--model", "fake-model", "--mode", "compare"]) == 0
    output = capsys.readouterr().out
    assert backend.calls == 1
    assert "Compare skipped" in output
    assert "🔬 COMPARE ·" not in output


def test_ask_question_type_flag_overrides_detection(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["12", "12"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert (
        main(["ask", "What is 7 + 5?", "--model", "fake-model", "--question-type", "general"]) == 0
    )
    assert "💬 General question" in capsys.readouterr().out
    assert "Natalia" not in backend.prompts[0]

    assert main(["ask", "Who are you?", "--model", "fake-model", "--question-type", "math"]) == 0
    assert "🎯 Answer: 12" in capsys.readouterr().out
    assert "Natalia" in backend.prompts[1]


def test_ask_json_reports_the_general_route(monkeypatch, capsys) -> None:
    backend = FakeMLXBackend("fake-model", answers=["Beijing"], confident_after=0)
    monkeypatch.setattr(cli, "MLXBackend", lambda *args, **kwargs: backend)

    assert main(["ask", "中国的首都在哪", "--model", "fake-model", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["question_type"] == "general"
    assert payload["passes"] == 1
    assert payload["text"] == "The answer is Beijing."


def test_keyboard_interrupt_while_loading_exits_cleanly(monkeypatch, capsys) -> None:
    class InterruptingLoader(FakeMLXBackend):
        def load(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "MLXBackend", InterruptingLoader)

    assert main(["ask", "What is 7 + 5?", "--model", "fake-model"]) == 130
    assert "Interrupted" in capsys.readouterr().err


def test_default_public_chat_uses_reproducible_threshold() -> None:
    args = cli.build_parser().parse_args(["chat", "--mode", "siftsc"])
    runtime_args = cli._chat_runtime_args(args)

    assert runtime_args.threshold == cli.DEMO_THRESHOLD


def test_demo_arguments_include_raw_prompt_for_the_chat_continuation() -> None:
    args = cli.build_parser().parse_args(["demo"])

    assert args.raw_prompt is False
