"""Run both SiftSC routes on a local MLX model and write a scrubbed report."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from siftsc import MLXBackend, SiftResult, SiftSC, load_profile, math_prompt


def summarize(result: SiftResult) -> dict[str, object]:
    return {
        "parsed_answer": result.parsed_answer,
        "used_self_consistency": result.used_self_consistency,
        "generation_passes": result.generation_passes,
        "gate_score": result.decision.score,
        "threshold": result.decision.threshold,
        "greedy_tokens": result.decision.features["greedy_n_tokens"],
        "vote_counts": result.vote_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", default="qwen05b-q4")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/local_mlx_smoke.json"))
    args = parser.parse_args()
    question = "What is 7 + 5?"
    prompt = math_prompt(question)
    base_gate = load_profile("qwen05b-q4-confidence")

    backend = MLXBackend(args.model)
    default = SiftSC(backend, base_gate, max_tokens=64)(prompt, feature_text=question)
    forced = SiftSC(backend, base_gate.with_threshold(0.0), max_tokens=64)(
        prompt, feature_text=question
    )
    report = {
        "schema_version": 1,
        "ran_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "model": args.model_label,
        "prompt": question,
        "default_profile_route": summarize(default),
        "forced_sc_route": summarize(forced),
        "path_scrubbed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
