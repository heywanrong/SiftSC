from __future__ import annotations

import os

import pytest

from siftsc import MLXBackend, SiftSC, load_profile, math_prompt


@pytest.mark.local_model
def test_local_mlx_model_end_to_end() -> None:
    model_path = os.getenv("SIFTSC_MLX_MODEL")
    if not model_path:
        pytest.skip("set SIFTSC_MLX_MODEL to run the local-model integration test")
    runner = SiftSC(
        MLXBackend(model_path),
        load_profile("qwen05b-q4-confidence").with_threshold(1.1),
        samples=5,
        max_tokens=48,
    )
    result = runner(math_prompt("What is 7 + 5?"), feature_text="What is 7 + 5?")
    assert result.generation_passes == 1
    assert result.text.strip()
    assert result.decision.features["greedy_n_tokens"] > 0
