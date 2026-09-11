"""Command-line interface for local selective reasoning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from . import __version__
from .backends import Backend, MeteredBackend, MLXBackend
from .display import (
    ALWAYS_SC,
    COMPARE,
    PLAIN,
    POLICY_DESCRIPTIONS,
    POLICY_ICONS,
    SIFTSC,
    CompareRow,
    SessionStats,
    TurnCost,
    answer_lines,
    compare_table,
    compute_chart,
    session_chart,
    session_line,
    workload_chart,
)
from .gates import list_profiles, load_profile
from .prompts import fallback_chat_prompt, looks_like_math, math_prompt
from .router import SiftSC
from .terminal import (
    AnimatedStatus,
    QuietLibraryOutput,
    format_seconds,
    terminal_width,
    wrap_labeled,
)
from .types import Generation, SiftResult
from .voting import parse_answer, plurality_vote

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_MODEL_REVISION = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
DEFAULT_PROFILE = "qwen05b-q4-confidence"
DEMO_QUESTION = (
    "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. "
    "His second stop was 15 miles before the end of the trip. How many miles did he travel "
    "between his first and second stops?"
)
DEMO_ANSWER = "25"
PROTECTION_QUESTION = (
    "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, "
    "calculate Allen's age 10 years from now."
)
PROTECTION_ANSWER = "109"
DEMO_THRESHOLD = 0.84
BENCHMARK_PROMPTS = 400
BENCHMARK_ESCALATIONS = 3
BENCHMARK_ALWAYS_SC_PASSES = BENCHMARK_PROMPTS * 5
BENCHMARK_SIFTSC_PASSES = BENCHMARK_PROMPTS + BENCHMARK_ESCALATIONS * 5
BENCHMARK_COMPUTE_REDUCTION = 1.0 - BENCHMARK_SIFTSC_PASSES / BENCHMARK_ALWAYS_SC_PASSES
BENCHMARK_ACCURACY_RETENTION = 0.987
MODES = (PLAIN, SIFTSC, COMPARE)
QUESTION_TYPES = ("auto", "math", "general")
EXAMPLE_QUESTION = "If 3 notebooks cost £4 each, what is the total?"
_EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"local MLX directory or Hugging Face model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--threshold", type=float, help="override the profile threshold")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--raw-prompt", action="store_true", help="do not add the paper's math prompt"
    )
    parser.add_argument(
        "--question-type",
        choices=QUESTION_TYPES,
        default="auto",
        help="math: the paper's reasoning route; general: one chat-style pass (default: auto)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siftsc",
        description="Let a small local model vote only when one answer is not enough.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="start an interactive local reasoning chat")
    _add_runtime_arguments(chat)
    chat.set_defaults(max_tokens=128)
    chat.add_argument("--mode", choices=MODES, help="skip the startup menu")

    demo = subparsers.add_parser(
        "demo", help="reproduce two fixed cases, then keep the model loaded for your questions"
    )
    _add_runtime_arguments(demo)
    demo.set_defaults(max_tokens=128)
    demo.add_argument("--json", action="store_true", help="emit machine-readable output")
    demo.add_argument(
        "--no-chat",
        action="store_true",
        help="exit after the fixed showcase instead of opening interactive chat",
    )

    ask = subparsers.add_parser("ask", help="run one prompt through a local MLX model")
    ask.add_argument("prompt", help="question or prompt text")
    _add_runtime_arguments(ask)
    ask.add_argument("--mode", choices=MODES, default=SIFTSC)
    ask.add_argument(
        "--chat-template",
        action="store_true",
        help="apply the model tokenizer's chat template",
    )
    ask.add_argument("--json", action="store_true", help="emit machine-readable output")

    subparsers.add_parser("profiles", help="list bundled gate profiles")
    return parser


def _make_runner(args: argparse.Namespace, backend: Backend) -> SiftSC:
    gate = load_profile(args.profile)
    if args.threshold is not None:
        gate = gate.with_threshold(args.threshold)
    return SiftSC(
        backend,
        gate,
        samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


# --------------------------------------------------------------------------- loading


def _model_label(model: str) -> str:
    if Path(model).exists():
        return Path(model).name or model
    return model.rsplit("/", 1)[-1]


def _megabytes(done: int, total: int | None) -> str:
    text = f"{done / 1e6:,.0f} MB"
    return f"{text} of {total / 1e6:,.0f} MB" if total else text


def _download_if_needed(backend: MLXBackend, label: str, console: TextIO) -> None:
    """Fetch model files first so loading itself is quick and the progress is visible."""

    is_cached = getattr(backend, "is_cached", None)
    ensure_downloaded = getattr(backend, "ensure_downloaded", None)
    if is_cached is None or ensure_downloaded is None or is_cached():
        return
    print(f"📦 One-time download · {label} · later runs start from the local cache", file=console)
    with AnimatedStatus(
        "📦 Downloading the model", "✅ Model downloaded · {elapsed}", stream=console
    ) as status:

        def progress(done: int, total: int | None) -> None:
            status.update(_megabytes(done, total))

        ensure_downloaded(progress)


def _print_library_output(captured: str, *, force: bool) -> None:
    if not captured.strip() or not (force or os.environ.get("SIFTSC_VERBOSE")):
        return
    print("🔎 Library output while loading:", file=sys.stderr)
    for line in captured.strip().splitlines():
        print(f"   {line}", file=sys.stderr)


def _load_backend(model: str, *, chat_template: bool = False) -> MLXBackend:
    revision = DEFAULT_MODEL_REVISION if model == DEFAULT_MODEL else None
    backend = MLXBackend(model, use_chat_template=chat_template, revision=revision)
    label = _model_label(model)
    previous_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    quiet = QuietLibraryOutput()
    try:
        with quiet:
            _download_if_needed(backend, label, quiet.console)
            with AnimatedStatus(
                f"🚀 Waking up Sifty · {label}",
                "✅ Sifty is ready · {elapsed}",
                stream=quiet.console,
            ):
                backend.load()
    except BaseException:
        _print_library_output(quiet.captured, force=True)
        raise
    finally:
        if previous_progress is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_progress
    _print_library_output(quiet.captured, force=False)
    print(file=sys.stderr)
    return backend


# --------------------------------------------------------------------------- engine


@dataclass(slots=True)
class _Comparison:
    result: SiftResult
    rows: list[CompareRow]
    cost: TurnCost


@dataclass(slots=True)
class _Engine:
    """Run one question under a policy and measure what it cost."""

    runner: SiftSC
    metered: MeteredBackend
    samples: int
    max_tokens: int
    temperature: float
    top_p: float
    raw_prompt: bool
    question_type: str
    backend: Backend
    stats: SessionStats

    @classmethod
    def create(cls, args: argparse.Namespace, backend: Backend) -> _Engine:
        metered = MeteredBackend(backend)
        return cls(
            runner=_make_runner(args, metered),
            metered=metered,
            samples=args.samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            raw_prompt=bool(getattr(args, "raw_prompt", False)),
            question_type=str(getattr(args, "question_type", "auto")),
            backend=backend,
            stats=SessionStats(samples=args.samples),
        )

    def kind_of(self, question: str, forced: str | None = None) -> str:
        """Classify a question as ``math`` (paper route) or ``general`` (chat route)."""

        if self.raw_prompt:
            return "math"
        chosen = forced or self.question_type
        if chosen in {"math", "general"}:
            return chosen
        return "math" if looks_like_math(question) else "general"

    def prompt_for(self, question: str) -> str:
        return question if self.raw_prompt else math_prompt(question)

    def chat_prompt_for(self, question: str) -> str:
        render = getattr(self.backend, "chat_prompt", None)
        if render is None:
            return fallback_chat_prompt(question)
        return str(render(question))

    def general(self, question: str) -> tuple[Generation, TurnCost]:
        """Answer a general question once, in chat style, without voting."""

        generation = self.metered.generate(
            self.chat_prompt_for(question),
            greedy=True,
            seed=0,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return generation, self._take_cost()

    def _take_cost(self) -> TurnCost:
        costs = self.metered.take()
        return TurnCost(
            passes=len(costs),
            seconds=sum(cost.seconds for cost in costs),
            tokens=sum(cost.tokens for cost in costs),
        )

    def plain(self, question: str) -> tuple[Generation, TurnCost]:
        generation = self.metered.generate(
            self.prompt_for(question),
            greedy=True,
            seed=0,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return generation, self._take_cost()

    def siftsc(self, question: str) -> tuple[SiftResult, TurnCost]:
        result = self.runner(self.prompt_for(question), feature_text=question)
        return result, self._take_cost()

    def compare(self, question: str) -> _Comparison:
        """Answer with SiftSC, then complete the always-SC baseline without waste.

        Plain inference is exactly SiftSC's greedy draft, so it is never rerun. When
        SiftSC called a vote, always-SC is the same five voters; otherwise five fresh
        voters are drawn with the router's own seeds.
        """

        prompt = self.prompt_for(question)
        result = self.runner(prompt, feature_text=question)
        costs = self.metered.take()
        draft_cost, voter_costs = costs[0], costs[1:]
        if result.used_self_consistency:
            voters = result.generations[1:]
            always_answer = result.parsed_answer
        else:
            voters = self.runner.sample_traces(prompt)
            voter_costs = self.metered.take()
            _, always_answer, _ = plurality_vote(voters, self.runner.parser)
        voter_seconds = sum(cost.seconds for cost in voter_costs)
        votes = " · ".join(self.runner.parser(voter.text) or "∅" for voter in voters)
        siftsc_seconds = draft_cost.seconds + (
            voter_seconds if result.used_self_consistency else 0.0
        )
        siftsc_note = "🗳️ vote called" if result.used_self_consistency else "🛡️ vote skipped"
        rows = [
            CompareRow(
                PLAIN, self.runner.parser(result.generations[0].text), 1, draft_cost.seconds
            ),
            CompareRow(ALWAYS_SC, always_answer, len(voters), voter_seconds, f"votes {votes}"),
            CompareRow(
                SIFTSC, result.parsed_answer, result.generation_passes, siftsc_seconds, siftsc_note
            ),
        ]
        cost = TurnCost(
            passes=1 + len(voters),
            seconds=draft_cost.seconds + voter_seconds,
            tokens=draft_cost.tokens + sum(cost.tokens for cost in voter_costs),
        )
        return _Comparison(result=result, rows=rows, cost=cost)


# --------------------------------------------------------------------------- rendering


def _compute_comparison(passes: int, always_sc_passes: int) -> dict[str, int | float]:
    saved = always_sc_passes - passes
    reduction = saved / always_sc_passes if always_sc_passes else 0.0
    return {
        "actual_passes": passes,
        "always_sc_passes": always_sc_passes,
        "passes_saved": saved,
        "compute_reduction": reduction,
    }


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _print_plain_turn(
    generation: Generation, cost: TurnCost, engine: _Engine, *, width: int
) -> None:
    _print_lines(answer_lines(generation.text, width=width))
    answer = parse_answer(generation.text)
    print(f"\n🎯 Answer: {answer or '∅'} · ⚡ one pass · {format_seconds(cost.seconds)}")
    _print_lines(compute_chart(actual_passes=1, samples=engine.samples, mode=PLAIN))


def _print_siftsc_turn(result: SiftResult, cost: TurnCost, engine: _Engine, *, width: int) -> None:
    _print_lines(answer_lines(result.text, width=width))
    draft_answer = parse_answer(result.generations[0].text)
    if result.used_self_consistency:
        votes = " · ".join(parse_answer(item.text) or "∅" for item in result.generations[1:])
        route = f"🗳️ vote called · votes {votes}"
        draft_note = f"(the draft said {draft_answer or '∅'})"
        if draft_answer == result.parsed_answer:
            actual_note = "🤝 vote confirmed the draft"
        else:
            actual_note = "🛠️ vote changed the draft"
    else:
        route, draft_note, actual_note = "🛡️ first answer accepted", "", ""
    passes = result.generation_passes
    unit = "pass" if passes == 1 else "passes"
    print(
        f"\n🎯 Answer: {result.parsed_answer or '∅'} · {route} · {passes} {unit} · "
        f"{format_seconds(cost.seconds)}"
    )
    _print_lines(
        compute_chart(
            actual_passes=passes,
            samples=engine.samples,
            draft_note=draft_note,
            actual_note=actual_note,
        )
    )


def _print_compare_turn(comparison: _Comparison, engine: _Engine, *, width: int) -> None:
    result = comparison.result
    _print_lines(answer_lines(result.text, width=width))
    passes = result.generation_passes
    unit = "pass" if passes == 1 else "passes"
    route = "🗳️ vote called" if result.used_self_consistency else "🛡️ first answer accepted"
    print(
        f"\n🎯 Answer: {result.parsed_answer or '∅'} · {route} · {passes} {unit} · "
        f"{format_seconds(comparison.cost.seconds)} for all three policies"
    )
    _print_lines(compare_table(comparison.rows, samples=engine.samples))


def _print_general_turn(generation: Generation, cost: TurnCost, *, mode: str, width: int) -> None:
    _print_lines(answer_lines(generation.text, width=width))
    print(f"\n💬 General question · answered once in chat style · {format_seconds(cost.seconds)}")
    print("   Votes are for reasoning questions with a checkable answer · try /math <q>")
    if mode == COMPARE:
        print("   🔬 Compare skipped · a free-text answer cannot be counted as votes")


def _run_turn(
    engine: _Engine, mode: str, question: str, *, width: int, kind: str | None = None
) -> None:
    """Answer one question in ``mode``; status goes to stderr, results to stdout."""

    if engine.kind_of(question, kind) == "general":
        with AnimatedStatus("💬 Sifty is answering", "✨ Answer ready · {elapsed}"):
            generation, cost = engine.general(question)
        print()
        _print_general_turn(generation, cost, mode=mode, width=width)
        engine.stats.record(cost, voted=False)
        return
    if mode == PLAIN:
        with AnimatedStatus("💭 Sifty is thinking once", "✨ Answer ready · {elapsed}"):
            generation, cost = engine.plain(question)
        print()
        _print_plain_turn(generation, cost, engine, width=width)
        engine.stats.record(cost, voted=False)
    elif mode == COMPARE:
        with AnimatedStatus(
            "🔬 Sifty is running plain, always-SC and siftsc", "✨ Comparison ready · {elapsed}"
        ):
            comparison = engine.compare(question)
        print()
        _print_compare_turn(comparison, engine, width=width)
        engine.stats.record(comparison.cost, voted=comparison.result.used_self_consistency)
    else:
        with AnimatedStatus(
            "🧠 Sifty is deciding whether to call a vote", "✨ Answer ready · {elapsed}"
        ):
            result, cost = engine.siftsc(question)
        print()
        _print_siftsc_turn(result, cost, engine, width=width)
        engine.stats.record(cost, voted=result.used_self_consistency)


# --------------------------------------------------------------------------- ask


def _run_ask(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model, chat_template=args.chat_template)
    engine = _Engine.create(args, backend)
    width = terminal_width()
    if engine.kind_of(args.prompt) == "general":
        with AnimatedStatus("💬 Answering", "✨ Answer ready · {elapsed}"):
            generation, cost = engine.general(args.prompt)
        if args.json:
            payload_general: dict[str, object] = {
                "mode": args.mode,
                "question_type": "general",
                "text": generation.text,
                "passes": 1,
                "cost": asdict(cost),
            }
            print(json.dumps(payload_general, ensure_ascii=False, indent=2))
        else:
            print()
            _print_general_turn(generation, cost, mode=args.mode, width=width)
        return 0
    if args.mode == PLAIN:
        with AnimatedStatus("💭 Thinking once", "✨ Answer ready · {elapsed}"):
            generation, cost = engine.plain(args.prompt)
        if args.json:
            payload: dict[str, object] = {
                "mode": PLAIN,
                "text": generation.text,
                "parsed_answer": parse_answer(generation.text),
                "passes": 1,
                "cost": asdict(cost),
                "compute_comparison": _compute_comparison(1, args.samples),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print()
            _print_plain_turn(generation, cost, engine, width=width)
        return 0

    if args.mode == COMPARE:
        with AnimatedStatus(
            "🔬 Running plain, always-SC and siftsc", "✨ Comparison ready · {elapsed}"
        ):
            comparison = engine.compare(args.prompt)
        if args.json:
            payload = asdict(comparison.result)
            payload["policies"] = [asdict(row) for row in comparison.rows]
            payload["cost"] = asdict(comparison.cost)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print()
            _print_compare_turn(comparison, engine, width=width)
        return 0

    with AnimatedStatus(
        "🧠 Deciding whether this answer needs a vote", "✨ Answer ready · {elapsed}"
    ):
        result, cost = engine.siftsc(args.prompt)
    if args.json:
        payload = asdict(result)
        payload["cost"] = asdict(cost)
        payload["compute_comparison"] = _compute_comparison(result.generation_passes, args.samples)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print()
        _print_siftsc_turn(result, cost, engine, width=width)
    return 0


# --------------------------------------------------------------------------- chat


def _choose_mode() -> str:
    print("🚀 Choose a reasoning mode:")
    print(f"  1  {POLICY_ICONS[PLAIN]} plain     {POLICY_DESCRIPTIONS[PLAIN]}")
    print(f"  2  {POLICY_ICONS[SIFTSC]} siftsc    {POLICY_DESCRIPTIONS[SIFTSC]}")
    print(f"  3  {POLICY_ICONS[COMPARE]} compare   {POLICY_DESCRIPTIONS[COMPARE]}")
    choice = input("✨ mode [2] > ").strip().lower()
    if choice in {"1", PLAIN, "p"}:
        return PLAIN
    if choice in {"3", COMPARE, "c"}:
        return COMPARE
    return SIFTSC


def _chat_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.raw_prompt = bool(getattr(args, "raw_prompt", False))
    if (
        runtime_args.threshold is None
        and runtime_args.model == DEFAULT_MODEL
        and runtime_args.profile == DEFAULT_PROFILE
    ):
        runtime_args.threshold = DEMO_THRESHOLD
    return runtime_args


def _print_chat_help() -> None:
    print(f"  /plain     {POLICY_ICONS[PLAIN]} {POLICY_DESCRIPTIONS[PLAIN]}")
    print(f"  /siftsc    {POLICY_ICONS[SIFTSC]} {POLICY_DESCRIPTIONS[SIFTSC]}")
    print(f"  /compare   {POLICY_ICONS[COMPARE]} {POLICY_DESCRIPTIONS[COMPARE]}")
    print("  /math <q>  🧮 force the paper's reasoning route for one question")
    print("  /talk <q>  💬 force a one-pass chat-style answer for one question")
    print("  /stats     📊 show the session compute chart")
    print("  /clear     🧹 clear the screen")
    print("  /help      🧭 show these commands")
    print("  /exit      👋 leave SiftSC")


def _enable_line_editing() -> None:
    """Give ``input()`` arrow keys and history on real terminals."""

    if not sys.stdin.isatty():
        return
    try:
        import readline  # noqa: F401
    except ImportError:
        return


def _print_welcome(mode: str, *, launched_from_demo: bool, demo_threshold: bool) -> None:
    if launched_from_demo:
        print("\n🚀 YOUR TURN · The model stays loaded, ask your own question!")
    else:
        print(
            f"🚀 Sifty is online · mode {POLICY_ICONS[mode]} {mode} · {POLICY_DESCRIPTIONS[mode]}"
        )
    print("💡 Word problems are routed and voted · other questions get one chat-style answer")
    print("   Each turn is independent · /help lists commands")
    print(f'   Try: "{EXAMPLE_QUESTION}"')
    if demo_threshold:
        print(f"🧭 Public-model routing uses the reproducible demo threshold ({DEMO_THRESHOLD}).")
    print()


def _run_chat_session(
    args: argparse.Namespace,
    backend: Backend,
    *,
    mode: str,
    launched_from_demo: bool = False,
) -> int:
    runtime_args = _chat_runtime_args(args)
    engine = _Engine.create(runtime_args, backend)
    width = terminal_width()
    _enable_line_editing()
    _print_welcome(
        mode,
        launched_from_demo=launched_from_demo,
        demo_threshold=runtime_args.threshold == DEMO_THRESHOLD and args.threshold is None,
    )
    while True:
        try:
            question = input(f"💬 you · {mode} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for trying SiftSC!")
            return 0
        if not question:
            continue
        command = question.lower()
        if command in _EXIT_COMMANDS:
            print("👋 Thanks for trying SiftSC!")
            return 0
        if command in {f"/{name}" for name in MODES}:
            mode = command[1:]
            print(f"🔀 Mode switched · {POLICY_ICONS[mode]} {mode} · {POLICY_DESCRIPTIONS[mode]}\n")
            continue
        if command == "/stats":
            _print_lines(session_chart(engine.stats, width=width))
            print()
            continue
        if command == "/clear":
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            else:
                print("\n" * 2, end="")
            print("🚀 Sifty is still here · type /help for commands.\n")
            continue
        if command == "/help":
            _print_chat_help()
            print()
            continue
        forced_kind: str | None = None
        for prefix, kind in (("/math ", "math"), ("/talk ", "general")):
            if command.startswith(prefix):
                forced_kind, question = kind, question[len(prefix) :].strip()
                break
        if question.startswith("/"):
            print(f"🤔 Unknown command {question} · type /help to see the list\n")
            continue
        if not question:
            continue

        try:
            _run_turn(engine, mode, question, width=width, kind=forced_kind)
        except KeyboardInterrupt:
            engine.metered.take()
            print("\n⏹  Stopped · ask another question or type /exit\n")
            continue
        print(session_line(engine.stats))
        print()


def _run_chat(args: argparse.Namespace) -> int:
    mode = args.mode or _choose_mode()
    backend = _load_backend(args.model)
    return _run_chat_session(args, backend, mode=mode)


# --------------------------------------------------------------------------- demo


def _demo_row(label: str, value: str | None, mark: str, note: str) -> str:
    return f"{label:<11}{value or '∅'}  {mark}   {note}"


def _mark(value: str | None, expected: str) -> str:
    return "✓" if value == expected else "✗"


def _run_demo(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model)
    demo_args = argparse.Namespace(**vars(args))
    demo_args.threshold = args.threshold if args.threshold is not None else DEMO_THRESHOLD
    forced_args = argparse.Namespace(**vars(args))
    forced_args.threshold = 0.0
    with AnimatedStatus(
        "🧪 Case 1 · Henry's bike trip · will a vote help?", "✅ Case 1 done · {elapsed}"
    ):
        result = _make_runner(demo_args, backend)(
            math_prompt(DEMO_QUESTION), feature_text=DEMO_QUESTION
        )
    with AnimatedStatus(
        "🧪 Case 2 · Darrell and Allen · should the vote be skipped?",
        "✅ Case 2 done · {elapsed}",
    ):
        protected = _make_runner(demo_args, backend)(
            math_prompt(PROTECTION_QUESTION), feature_text=PROTECTION_QUESTION
        )
    with AnimatedStatus(
        "🧪 Case 2 · forcing a blind vote for comparison", "✅ Comparison done · {elapsed}"
    ):
        always_sc = _make_runner(forced_args, backend)(
            math_prompt(PROTECTION_QUESTION), feature_text=PROTECTION_QUESTION
        )

    plain = parse_answer(result.generations[0].text)
    votes = [parse_answer(item.text) for item in result.generations[1:]]
    repair_reproduced = plain != DEMO_ANSWER and result.parsed_answer == DEMO_ANSWER
    protected_plain = parse_answer(protected.generations[0].text)
    protection_reproduced = (
        protected_plain == PROTECTION_ANSWER
        and protected.parsed_answer == PROTECTION_ANSWER
        and not protected.used_self_consistency
        and always_sc.parsed_answer != PROTECTION_ANSWER
    )
    reproduced = repair_reproduced and protection_reproduced
    payload = {
        "repair_case": {
            "question": DEMO_QUESTION,
            "expected": DEMO_ANSWER,
            "plain": plain,
            "votes": votes,
            "siftsc": result.parsed_answer,
            "reproduced": repair_reproduced,
        },
        "protection_case": {
            "question": PROTECTION_QUESTION,
            "expected": PROTECTION_ANSWER,
            "plain": protected_plain,
            "always_sc": always_sc.parsed_answer,
            "siftsc": protected.parsed_answer,
            "siftsc_passes": protected.generation_passes,
            "reproduced": protection_reproduced,
        },
        "measured_workload": {
            "prompts": BENCHMARK_PROMPTS,
            "always_sc_passes": BENCHMARK_ALWAYS_SC_PASSES,
            "siftsc_actual_passes": BENCHMARK_SIFTSC_PASSES,
            "compute_reduction": BENCHMARK_COMPUTE_REDUCTION,
            "accuracy_retention": BENCHMARK_ACCURACY_RETENTION,
        },
        "reproduced": reproduced,
        "model": args.model,
        "seed": args.seed,
    }
    interactive = not args.json and not args.no_chat and sys.stdin.isatty()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_demo_report(
            args,
            result=result,
            plain=plain,
            votes=votes,
            protected=protected,
            protected_plain=protected_plain,
            always_sc=always_sc,
            reproduced=reproduced,
        )
    if not reproduced:
        print(
            "[siftsc] The fixed demo did not reproduce. "
            "Check the documented model revision and mlx-lm version.",
            file=sys.stderr,
        )
        if not interactive:
            return 2
    if interactive:
        return _run_chat_session(args, backend, mode=SIFTSC, launched_from_demo=True)
    return 0


def _print_demo_report(
    args: argparse.Namespace,
    *,
    result: SiftResult,
    plain: str | None,
    votes: list[str | None],
    protected: SiftResult,
    protected_plain: str | None,
    always_sc: SiftResult,
    reproduced: bool,
) -> None:
    width = terminal_width()
    rule = "─" * min(width, 72)
    vote_text = " · ".join(vote or "∅" for vote in votes)
    print(rule)
    print("🗳️  CASE 1 · VOTE WHEN IT HELPS")
    _print_lines(wrap_labeled("Question", DEMO_QUESTION, width=width, label_width=11))
    print(f"{'Expected':<11}{DEMO_ANSWER}")
    print(_demo_row("PLAIN", plain, _mark(plain, DEMO_ANSWER), "1 pass"))
    print(
        _demo_row(
            "SIFTSC",
            result.parsed_answer,
            _mark(result.parsed_answer, DEMO_ANSWER),
            f"{result.generation_passes} passes · votes {vote_text}",
        )
    )
    print(rule)
    print("🛡️  CASE 2 · SKIP WHEN VOTING HURTS")
    _print_lines(wrap_labeled("Question", PROTECTION_QUESTION, width=width, label_width=11))
    print(f"{'Expected':<11}{PROTECTION_ANSWER}")
    print(_demo_row("PLAIN", protected_plain, _mark(protected_plain, PROTECTION_ANSWER), "1 pass"))
    print(
        _demo_row(
            "ALWAYS-SC",
            always_sc.parsed_answer,
            _mark(always_sc.parsed_answer, PROTECTION_ANSWER),
            f"{args.samples} passes · blind voting changed a right answer",
        )
    )
    protected_note = "vote skipped" if not protected.used_self_consistency else "vote called"
    unit = "pass" if protected.generation_passes == 1 else "passes"
    print(
        _demo_row(
            "SIFTSC",
            protected.parsed_answer,
            _mark(protected.parsed_answer, PROTECTION_ANSWER),
            f"{protected.generation_passes} {unit} · {protected_note}",
        )
    )
    _print_lines(compute_chart(actual_passes=protected.generation_passes, samples=args.samples))
    print(rule)
    _print_lines(
        workload_chart(
            prompts=BENCHMARK_PROMPTS,
            always_sc_passes=BENCHMARK_ALWAYS_SC_PASSES,
            siftsc_passes=BENCHMARK_SIFTSC_PASSES,
            accuracy_retention=BENCHMARK_ACCURACY_RETENTION,
            width=width,
        )
    )
    print(rule)
    if reproduced:
        print("✅ Reproduced: repair when voting helps; skip when voting hurts.")
    else:
        print("⚠️  The fixed showcase did not reproduce exactly on this machine.")
        print("   Expected: plain 15 → siftsc 25, and plain 109 kept while a blind vote loses it.")
        print("   Check the pinned model revision and mlx-lm version; you can still chat below.")


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "profiles":
        print("\n".join(list_profiles()))
        return 0
    if args.command == "ask":
        return _run_ask(args)
    if args.command == "chat":
        return _run_chat(args)
    if args.command == "demo":
        return _run_demo(args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted · bye!", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
