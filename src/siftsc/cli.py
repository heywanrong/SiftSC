"""Command-line interface for local selective reasoning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from . import __version__
from .backends import Backend, MLXBackend
from .display import (
    COMPARE,
    PLAIN,
    POLICY_DESCRIPTIONS,
    POLICY_ICONS,
    SIFTSC,
    answer_lines,
    compute_chart,
    session_chart,
    session_line,
    workload_chart,
)
from .engine import (
    MODES,
    QUESTION_TYPES,
    Engine,
    TurnReport,
    make_runner,
    parse_input,
    status_message,
    success_message,
)
from .gates import list_profiles
from .prompts import math_prompt
from .terminal import (
    AnimatedStatus,
    QuietLibraryOutput,
    terminal_width,
    wrap_labeled,
)
from .types import SiftResult
from .voting import parse_answer

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
EXAMPLE_QUESTION = "If 3 notebooks cost £4 each, what is the total?"


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

    ui = subparsers.add_parser("ui", help="open the full-screen terminal interface (default)")
    _add_runtime_arguments(ui)
    ui.set_defaults(max_tokens=128)
    ui.add_argument("--mode", choices=MODES, default=SIFTSC, help="starting reasoning mode")

    chat = subparsers.add_parser("chat", help="start a line-mode local reasoning chat")
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


def _print_lines(lines: tuple[str, ...] | list[str]) -> None:
    for line in lines:
        print(line)


def _print_report(report: TurnReport, *, width: int) -> None:
    _print_lines(answer_lines(report.text, width=width))
    print(f"\n{report.headline}")
    _print_lines(report.notes)
    _print_lines(report.chart)


def _run_turn(
    engine: Engine, mode: str, question: str, *, width: int, kind: str | None = None
) -> None:
    """Answer one question in ``mode``; status goes to stderr, results to stdout."""

    resolved = engine.kind_of(question, kind)
    with AnimatedStatus(status_message(mode, resolved), success_message(mode, resolved)):
        report = engine.answer(mode, question, kind=resolved)
    print()
    _print_report(report, width=width)


# --------------------------------------------------------------------------- ask


def _ask_json(args: argparse.Namespace, engine: Engine) -> dict[str, object]:
    kind = engine.kind_of(args.prompt)
    if kind == "general":
        generation, cost = engine.general(args.prompt)
        return {
            "mode": args.mode,
            "question_type": "general",
            "text": generation.text,
            "passes": 1,
            "cost": asdict(cost),
        }
    if args.mode == PLAIN:
        generation, cost = engine.plain(args.prompt)
        return {
            "mode": PLAIN,
            "text": generation.text,
            "parsed_answer": parse_answer(generation.text),
            "passes": 1,
            "cost": asdict(cost),
            "compute_comparison": _compute_comparison(1, args.samples),
        }
    if args.mode == COMPARE:
        comparison = engine.compare(args.prompt)
        payload = asdict(comparison.result)
        payload["policies"] = [asdict(row) for row in comparison.rows]
        payload["cost"] = asdict(comparison.cost)
        return payload
    result, cost = engine.siftsc(args.prompt)
    payload = asdict(result)
    payload["cost"] = asdict(cost)
    payload["compute_comparison"] = _compute_comparison(result.generation_passes, args.samples)
    return payload


def _run_ask(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model, chat_template=args.chat_template)
    engine = Engine.create(args, backend)
    if args.json:
        kind = engine.kind_of(args.prompt)
        with AnimatedStatus(status_message(args.mode, kind), success_message(args.mode, kind)):
            payload = _ask_json(args, engine)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _run_turn(engine, args.mode, args.prompt, width=terminal_width())
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


def _uses_demo_threshold(args: argparse.Namespace, runtime_args: argparse.Namespace) -> bool:
    return bool(runtime_args.threshold == DEMO_THRESHOLD and args.threshold is None)


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
        icon, description = POLICY_ICONS[mode], POLICY_DESCRIPTIONS[mode]
        print(f"🚀 Sifty is online · mode {icon} {mode} · {description}")
    print("💡 Word problems are routed and voted · other questions get one chat-style answer")
    print("   Each turn is independent · /help lists commands · full-screen version: siftsc ui")
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
    engine = Engine.create(runtime_args, backend)
    width = terminal_width()
    _enable_line_editing()
    _print_welcome(
        mode,
        launched_from_demo=launched_from_demo,
        demo_threshold=_uses_demo_threshold(args, runtime_args),
    )
    while True:
        try:
            typed = input(f"💬 you · {mode} > ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for trying SiftSC!")
            return 0
        entry = parse_input(typed)
        if entry.command == "exit":
            print("👋 Thanks for trying SiftSC!")
            return 0
        if entry.command == "mode" and entry.argument is not None:
            mode = entry.argument
            print(f"🔀 Mode switched · {POLICY_ICONS[mode]} {mode} · {POLICY_DESCRIPTIONS[mode]}\n")
            continue
        if entry.command == "stats":
            _print_lines(session_chart(engine.stats, width=width))
            print()
            continue
        if entry.command == "clear":
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            else:
                print("\n" * 2, end="")
            print("🚀 Sifty is still here · type /help for commands.\n")
            continue
        if entry.command == "help":
            _print_chat_help()
            print()
            continue
        if entry.command == "unknown":
            print(f"🤔 Unknown command {entry.argument} · type /help to see the list\n")
            continue
        if entry.question is None:
            continue

        try:
            _run_turn(engine, mode, entry.question, width=width, kind=entry.kind)
        except KeyboardInterrupt:
            engine.discard_partial_cost()
            print("\n⏹  Stopped · ask another question or type /exit\n")
            continue
        print(session_line(engine.stats))
        print()


def _run_chat(args: argparse.Namespace) -> int:
    mode = args.mode or _choose_mode()
    backend = _load_backend(args.model)
    return _run_chat_session(args, backend, mode=mode)


# --------------------------------------------------------------------------- ui


def _run_ui(args: argparse.Namespace) -> int:
    try:
        from .tui import run_app
    except ImportError:
        print(
            "🖥️  The full-screen interface needs the 'ui' extra: python -m pip install 'siftsc[ui]'",
            file=sys.stderr,
        )
        print("   Meanwhile `siftsc chat` offers the same features in line mode.", file=sys.stderr)
        return 2
    backend = _load_backend(args.model)
    runtime_args = _chat_runtime_args(args)
    engine = Engine.create(runtime_args, backend)
    return run_app(engine, mode=args.mode, demo_threshold=_uses_demo_threshold(args, runtime_args))


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
        result = make_runner(demo_args, backend)(
            math_prompt(DEMO_QUESTION), feature_text=DEMO_QUESTION
        )
    with AnimatedStatus(
        "🧪 Case 2 · Darrell and Allen · should the vote be skipped?",
        "✅ Case 2 done · {elapsed}",
    ):
        protected = make_runner(demo_args, backend)(
            math_prompt(PROTECTION_QUESTION), feature_text=PROTECTION_QUESTION
        )
    with AnimatedStatus(
        "🧪 Case 2 · forcing a blind vote for comparison", "✅ Comparison done · {elapsed}"
    ):
        always_sc = make_runner(forced_args, backend)(
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


# --------------------------------------------------------------------------- main


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "profiles":
        print("\n".join(list_profiles()))
        return 0
    if args.command == "ask":
        return _run_ask(args)
    if args.command == "chat":
        return _run_chat(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "demo":
        return _run_demo(args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        if sys.stdin.isatty():
            arguments = ["ui"]
        else:
            build_parser().print_help()
            return 2
    args = build_parser().parse_args(arguments)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted · bye!", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
