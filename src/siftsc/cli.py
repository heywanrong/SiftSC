"""Command-line interface for local selective reasoning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .backends import MLXBackend
from .gates import list_profiles, load_profile
from .prompts import math_prompt
from .router import SiftSC
from .terminal import AnimatedStatus
from .types import Generation
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siftsc",
        description="Let a small local model vote only when one answer is not enough.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="start an interactive local reasoning chat")
    _add_runtime_arguments(chat)
    chat.set_defaults(max_tokens=128)
    chat.add_argument("--mode", choices=("plain", "siftsc"), help="skip the startup menu")
    chat.add_argument(
        "--raw-prompt", action="store_true", help="do not add the paper's math prompt"
    )

    demo = subparsers.add_parser(
        "demo", help="reproduce a fixed case where voting repairs a plain answer"
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
    ask.add_argument("--mode", choices=("plain", "siftsc"), default="siftsc")
    ask.add_argument("--raw-prompt", action="store_true", help="do not add the paper's math prompt")
    ask.add_argument(
        "--chat-template",
        action="store_true",
        help="apply the model tokenizer's chat template",
    )
    ask.add_argument("--json", action="store_true", help="emit machine-readable output")

    subparsers.add_parser("profiles", help="list bundled gate profiles")
    return parser


def _make_runner(args: argparse.Namespace, backend: MLXBackend) -> SiftSC:
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


def _load_backend(model: str, *, chat_template: bool = False) -> MLXBackend:
    revision = DEFAULT_MODEL_REVISION if model == DEFAULT_MODEL else None
    backend = MLXBackend(model, use_chat_template=chat_template, revision=revision)
    if not Path(model).exists() and "/" in model:
        print(
            "📦 First launch downloads about 290 MB from Hugging Face; later runs use cache.",
            file=sys.stderr,
        )
        label = model
    else:
        label = f"local model · {model}"

    previous_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        with AnimatedStatus(
            f"🚀 Waking up Sifty · {label}",
            "✅ Sifty is ready!",
        ):
            backend.load()
    finally:
        if previous_progress is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_progress
    print(file=sys.stderr)
    return backend


def _format_prompt(question: str, *, raw: bool) -> str:
    return question if raw else math_prompt(question)


def _plain_generation(args: argparse.Namespace, backend: MLXBackend, prompt: str) -> Generation:
    return backend.generate(
        prompt,
        greedy=True,
        seed=0,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def _compute_comparison(passes: int, always_sc_passes: int) -> dict[str, int | float]:
    saved = always_sc_passes - passes
    reduction = saved / always_sc_passes if always_sc_passes else 0.0
    return {
        "actual_passes": passes,
        "always_sc_passes": always_sc_passes,
        "passes_saved": saved,
        "compute_reduction": reduction,
    }


def _compute_line(passes: int, always_sc_passes: int, *, label: str = "compute") -> str:
    comparison = _compute_comparison(passes, always_sc_passes)
    saved = int(comparison["passes_saved"])
    percentage = abs(float(comparison["compute_reduction"])) * 100
    direction = "saved" if saved >= 0 else "extra"
    icon = "⚡" if label == "compute" else "📊"
    return (
        f"{icon} "
        f"[{label}] actual={passes} pass{'es' if passes != 1 else ''} · "
        f"Always-SC={always_sc_passes} passes · "
        f"{direction}={abs(saved)} ({percentage:.1f}%)"
    )


def _run_ask(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model, chat_template=args.chat_template)
    prompt = _format_prompt(args.prompt, raw=args.raw_prompt)
    if args.mode == "plain":
        with AnimatedStatus("💭 Thinking once", "✨ Answer ready"):
            generation = _plain_generation(args, backend, prompt)
        answer = parse_answer(generation.text)
        if args.json:
            print(
                json.dumps(
                    {
                        "mode": "plain",
                        "text": generation.text,
                        "parsed_answer": answer,
                        "passes": 1,
                        "compute_comparison": _compute_comparison(1, args.samples),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(generation.text.strip())
            print("\n[plain] passes=1")
            print(_compute_line(1, args.samples))
        return 0

    with AnimatedStatus("🧠 Deciding whether this answer needs a vote", "✨ Answer ready"):
        result = _make_runner(args, backend)(prompt, feature_text=args.prompt)
    if args.json:
        payload = asdict(result)
        payload["compute_comparison"] = _compute_comparison(result.generation_passes, args.samples)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        route = "SC" if result.used_self_consistency else "greedy"
        print(result.text.strip())
        print(f"\n[siftsc] route={route} passes={result.generation_passes}")
        print(_compute_line(result.generation_passes, args.samples))
    return 0


def _choose_mode() -> str:
    print("🚀 Choose a reasoning mode:")
    print("  1  ⚡ plain   one quick deterministic answer")
    print("  2  🗳️  siftsc  vote only when the router escalates")
    choice = input("✨ mode [2] > ").strip().lower()
    return "plain" if choice in {"1", "plain", "p"} else "siftsc"


def _chat_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    runtime_args = argparse.Namespace(**vars(args))
    if (
        runtime_args.threshold is None
        and runtime_args.model == DEFAULT_MODEL
        and runtime_args.profile == DEFAULT_PROFILE
    ):
        runtime_args.threshold = DEMO_THRESHOLD
    return runtime_args


def _print_chat_help() -> None:
    print("  /plain   ⚡ one deterministic pass")
    print("  /siftsc  🗳️  selective voting")
    print("  /clear   🧹 clear the screen")
    print("  /help    🧭 show these commands")
    print("  /exit    👋 leave SiftSC")


def _run_chat_session(
    args: argparse.Namespace,
    backend: MLXBackend,
    *,
    mode: str,
    launched_from_demo: bool = False,
) -> int:
    runtime_args = _chat_runtime_args(args)
    runner = _make_runner(runtime_args, backend)
    session_passes = 0
    session_always_sc_passes = 0
    if launched_from_demo:
        print("\n🚀 YOUR TURN · The model stays loaded—ask your own question!")
    else:
        print("\n🚀 Sifty is online · Ask your own reasoning question!")
    print("💡 Each turn is independent; type /help for commands.")
    if runtime_args.threshold == DEMO_THRESHOLD and args.threshold is None:
        print("🧭 Public-model routing is tuned to the reproducible demo threshold.\n")
    else:
        print()
    while True:
        try:
            question = input(f"💬 you · {mode} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for trying SiftSC!")
            return 0
        if not question:
            continue
        command = question.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            print("👋 Thanks for trying SiftSC!")
            return 0
        if command in {"/plain", "/siftsc"}:
            mode = command[1:]
            icon = "⚡" if mode == "plain" else "🗳️"
            print(f"🔀 Mode switched · {icon} {mode}\n")
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

        prompt = _format_prompt(question, raw=runtime_args.raw_prompt)
        if mode == "plain":
            with AnimatedStatus("💭 Sifty is thinking once", "✨ Answer ready"):
                generation = _plain_generation(runtime_args, backend, prompt)
            passes = 1
            print(f"\n🤖 Sifty > {generation.text.strip()}")
            print("⚡ [plain] one pass · passes=1")
            print(_compute_line(passes, runtime_args.samples))
            session_passes += passes
            session_always_sc_passes += runtime_args.samples
            print(_compute_line(session_passes, session_always_sc_passes, label="session"), "\n")
            continue

        with AnimatedStatus(
            "🧠 Sifty is deciding whether to call a vote",
            "✨ Answer ready",
        ):
            result = runner(prompt, feature_text=question)
        route = "🗳️ voted" if result.used_self_consistency else "🛡️ accepted the first answer"
        print(f"\n🤖 Sifty > {result.text.strip()}")
        print(f"🧭 [siftsc] {route} · passes={result.generation_passes}")
        print(_compute_line(result.generation_passes, runtime_args.samples))
        session_passes += result.generation_passes
        session_always_sc_passes += runtime_args.samples
        print(_compute_line(session_passes, session_always_sc_passes, label="session"), "\n")


def _run_chat(args: argparse.Namespace) -> int:
    mode = args.mode or _choose_mode()
    backend = _load_backend(args.model)
    return _run_chat_session(args, backend, mode=mode)


def _run_demo(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model)
    demo_args = argparse.Namespace(**vars(args))
    demo_args.threshold = args.threshold if args.threshold is not None else DEMO_THRESHOLD
    with AnimatedStatus(
        "🧪 Running two verified showcase cases",
        "✅ Showcase ready",
    ):
        result = _make_runner(demo_args, backend)(
            math_prompt(DEMO_QUESTION), feature_text=DEMO_QUESTION
        )
        protected = _make_runner(demo_args, backend)(
            math_prompt(PROTECTION_QUESTION), feature_text=PROTECTION_QUESTION
        )
        forced_args = argparse.Namespace(**vars(args))
        forced_args.threshold = 0.0
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
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        plain_mark = "✗" if plain != DEMO_ANSWER else "✓"
        sift_mark = "✓" if result.parsed_answer == DEMO_ANSWER else "✗"
        vote_text = " · ".join(v or "∅" for v in votes)
        print("─" * 68)
        print("🗳️  CASE 1 · VOTE WHEN IT HELPS")
        print(f"Question  {DEMO_QUESTION}")
        print(f"Expected  {DEMO_ANSWER}")
        print("─" * 68)
        print(f"PLAIN    {plain or '∅'}  {plain_mark}   (1 deterministic pass)")
        print(f"SIFTSC   {result.parsed_answer or '∅'}  {sift_mark}   (votes: {vote_text})")
        print("─" * 68)
        print("🛡️  CASE 2 · SKIP WHEN VOTING HURTS")
        print(f"Question  {PROTECTION_QUESTION}")
        print(f"Expected  {PROTECTION_ANSWER}")
        print("─" * 68)
        print(f"PLAIN       {protected_plain or '∅'}  ✓")
        print(
            f"ALWAYS-SC   {always_sc.parsed_answer or '∅'}  ✗   "
            "(blind voting changed a right answer)"
        )
        print(f"SIFTSC      {protected.parsed_answer or '∅'}  ✓   (vote skipped)")
        print(_compute_line(protected.generation_passes, args.samples))
        print("─" * 68)
        print(f"⚡ MEASURED WORKLOAD · {BENCHMARK_PROMPTS} PROMPTS")
        print(f"ALWAYS-SC   {BENCHMARK_ALWAYS_SC_PASSES:,} generation passes")
        print(f"SIFTSC        {BENCHMARK_SIFTSC_PASSES:,} actual generation passes")
        print(
            f"SAVED       {BENCHMARK_ALWAYS_SC_PASSES - BENCHMARK_SIFTSC_PASSES:,} passes "
            f"({BENCHMARK_COMPUTE_REDUCTION:.1%} less compute)"
        )
        print(f"QUALITY     {BENCHMARK_ACCURACY_RETENTION:.1%} of Always-SC accuracy retained")
        print("─" * 68)
        print(
            "✅ Reproduced: repair when voting helps; skip when voting hurts."
            if reproduced
            else "Result drifted; see notes below."
        )
    if not reproduced:
        print(
            "[siftsc] The fixed demo did not reproduce. "
            "Check the documented model and mlx-lm version.",
            file=sys.stderr,
        )
        return 2
    if not args.json and not args.no_chat and sys.stdin.isatty():
        return _run_chat_session(args, backend, mode="siftsc", launched_from_demo=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
