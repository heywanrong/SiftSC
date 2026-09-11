"""Run SiftSC with a local MLX model.

Usage:
    python examples/basic_mlx.py /path/to/qwen05b-q4
"""

from __future__ import annotations

import argparse

from siftsc import MLXBackend, SiftSC, load_profile, math_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    question = "A box has 8 rows of 7 oranges. How many oranges are there?"
    router = SiftSC(
        MLXBackend(args.model),
        load_profile("qwen05b-q4-confidence"),
        samples=5,
    )
    result = router(math_prompt(question), feature_text=question)
    print(result.text)
    print(
        {
            "route": "SC" if result.used_self_consistency else "greedy",
            "passes": result.generation_passes,
            "score": round(result.decision.score, 3),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
