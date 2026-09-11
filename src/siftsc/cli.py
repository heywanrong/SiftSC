"""Command-line interface for local selective reasoning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .backends import MLXBackend
from .gates import list_profiles, load_profile
from .prompts import math_prompt
from .router import SiftSC


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siftsc",
        description="Spend self-consistency compute only when a small gate asks for it.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="run one prompt through a local MLX model")
    ask.add_argument("prompt", help="question or prompt text")
    ask.add_argument("--model", required=True, help="local MLX model directory")
    ask.add_argument("--profile", default="qwen05b-q4-confidence")
    ask.add_argument("--threshold", type=float, help="override the profile threshold")
    ask.add_argument("--samples", type=int, default=5)
    ask.add_argument("--max-tokens", type=int, default=256)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--top-p", type=float, default=0.95)
    ask.add_argument("--seed", type=int, default=1)
    ask.add_argument(
        "--raw-prompt", action="store_true", help="do not add the paper's math template"
    )
    ask.add_argument(
        "--chat-template",
        action="store_true",
        help="apply the model tokenizer's chat template",
    )
    ask.add_argument("--json", action="store_true", help="emit machine-readable output")

    subparsers.add_parser("profiles", help="list bundled gate profiles")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profiles":
        print("\n".join(list_profiles()))
        return 0

    gate = load_profile(args.profile)
    if args.threshold is not None:
        gate = gate.with_threshold(args.threshold)
    prompt = args.prompt if args.raw_prompt else math_prompt(args.prompt)
    backend = MLXBackend(args.model, use_chat_template=args.chat_template)
    runner = SiftSC(
        backend,
        gate,
        samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    result = runner(prompt, feature_text=args.prompt)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        route = "SC" if result.used_self_consistency else "greedy"
        print(result.text.strip())
        print(
            f"\n[siftsc] route={route} score={result.decision.score:.3f} "
            f"threshold={result.decision.threshold:.3f} "
            f"passes={result.generation_passes}/{args.samples}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
