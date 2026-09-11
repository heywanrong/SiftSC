"""Command-line interface for local selective reasoning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .backends import MLXBackend
from .gates import list_profiles, load_profile
from .prompts import math_prompt
from .router import SiftSC
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
DEMO_THRESHOLD = 0.84


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
        print(f"[siftsc] Loading {model}", file=sys.stderr)
        print(
            "[siftsc] First run downloads about 290 MB from Hugging Face; later runs use cache.",
            file=sys.stderr,
        )
    else:
        print(f"[siftsc] Loading local model: {model}", file=sys.stderr)
    backend.load()
    print("[siftsc] Ready.\n", file=sys.stderr)
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


def _run_ask(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model, chat_template=args.chat_template)
    prompt = _format_prompt(args.prompt, raw=args.raw_prompt)
    if args.mode == "plain":
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
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(generation.text.strip())
            print("\n[plain] passes=1")
        return 0

    result = _make_runner(args, backend)(prompt, feature_text=args.prompt)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        route = "SC" if result.used_self_consistency else "greedy"
        print(result.text.strip())
        print(f"\n[siftsc] route={route} passes={result.generation_passes}")
    return 0


def _choose_mode() -> str:
    print("Choose how the model should answer:")
    print("  1  plain   one deterministic answer")
    print("  2  siftsc  vote only when the router escalates")
    choice = input("mode [2]> ").strip().lower()
    return "plain" if choice in {"1", "plain", "p"} else "siftsc"


def _run_chat(args: argparse.Namespace) -> int:
    mode = args.mode or _choose_mode()
    backend = _load_backend(args.model)
    runner = _make_runner(args, backend)
    print("Ask a reasoning question. Commands: /plain, /siftsc, /clear, /exit")
    print("Each turn is independent so the bundled research profile stays in-distribution.\n")
    while True:
        try:
            question = input(f"you [{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not question:
            continue
        command = question.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            print("bye")
            return 0
        if command in {"/plain", "/siftsc"}:
            mode = command[1:]
            print(f"[siftsc] mode={mode}\n")
            continue
        if command == "/clear":
            print("\n" * 2, end="")
            continue

        prompt = _format_prompt(question, raw=args.raw_prompt)
        if mode == "plain":
            generation = _plain_generation(args, backend, prompt)
            print(f"assistant> {generation.text.strip()}")
            print("[plain] passes=1\n")
            continue

        result = runner(prompt, feature_text=question)
        route = "voted" if result.used_self_consistency else "one pass"
        print(f"assistant> {result.text.strip()}")
        print(f"[siftsc] {route} · passes={result.generation_passes}\n")


def _run_demo(args: argparse.Namespace) -> int:
    backend = _load_backend(args.model)
    demo_args = argparse.Namespace(**vars(args))
    demo_args.threshold = args.threshold if args.threshold is not None else DEMO_THRESHOLD
    result = _make_runner(demo_args, backend)(
        math_prompt(DEMO_QUESTION), feature_text=DEMO_QUESTION
    )
    plain = parse_answer(result.generations[0].text)
    votes = [parse_answer(item.text) for item in result.generations[1:]]
    reproduced = plain != DEMO_ANSWER and result.parsed_answer == DEMO_ANSWER
    payload = {
        "question": DEMO_QUESTION,
        "expected": DEMO_ANSWER,
        "plain": plain,
        "votes": votes,
        "siftsc": result.parsed_answer,
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
        print(f"Question  {DEMO_QUESTION}")
        print(f"Expected  {DEMO_ANSWER}")
        print("─" * 68)
        print(f"PLAIN    {plain or '∅'}  {plain_mark}   (1 deterministic pass)")
        print(f"SIFTSC   {result.parsed_answer or '∅'}  {sift_mark}   (votes: {vote_text})")
        print("─" * 68)
        if reproduced:
            print("Reproduced: plain is wrong, SiftSC is right.")
        else:
            print("Result drifted; see notes below.")
    if not reproduced:
        print(
            "[siftsc] The fixed demo did not reproduce. "
            "Check the documented model and mlx-lm version.",
            file=sys.stderr,
        )
        return 2
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
