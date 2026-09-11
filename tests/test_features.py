from __future__ import annotations

import numpy as np
import pytest

from siftsc.features import (
    FEATURE_NAMES,
    as_vector,
    extract_features,
    margin_confidence,
    prompt_features,
)
from siftsc.types import Generation


def test_margin_confidence_uses_top_two_probabilities() -> None:
    logits = ((2.0, 1.0, 0.0), (1.0, 1.0, 0.0))
    value = margin_confidence(logits)
    expected_first = (np.exp(2) - np.exp(1)) / (np.exp(2) + np.exp(1) + 1)
    assert value == pytest.approx(expected_first / 2)


def test_prompt_and_generation_features_match_schema() -> None:
    prompt = "If Ana has 12 apples, then buys 3, how many?"
    generation = Generation(
        text="The answer is 15.",
        token_ids=(1, 2),
        top_logits=((2.0, 1.0), (3.0, 1.0)),
        entropies=(0.2, 0.4),
    )
    features = extract_features(prompt, generation)
    assert as_vector(features).shape == (len(FEATURE_NAMES),)
    assert features["greedy_n_tokens"] == 2
    assert prompt_features(prompt)["text_n_digits"] == 3
    assert prompt_features(prompt)["text_n_steps"] == 1
