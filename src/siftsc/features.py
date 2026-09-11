"""Feature extraction matching the gate described in the ICONIP paper."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from .types import Generation

FEATURE_NAMES: tuple[str, ...] = (
    "greedy_conf",
    "greedy_top5_ent",
    "greedy_mean_ent",
    "greedy_n_tokens",
    "text_n_chars",
    "text_n_words",
    "text_n_digits",
    "text_n_ops",
    "text_n_qs",
    "text_n_commas",
    "text_n_lines",
    "text_n_steps",
)

_WORDS = re.compile(r"\S+")
_STEP_MARKERS = (" then ", " after ", " so ", " if ", " given ")


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    shifted = array - float(np.max(array))
    exp = np.exp(shifted)
    return exp / float(exp.sum())


def margin_confidence(top_logits: Sequence[Sequence[float]]) -> float:
    """Mean top-1 minus top-2 probability margin across generated tokens."""

    margins: list[float] = []
    for row in top_logits:
        if len(row) < 2:
            continue
        probs = _softmax(row)
        margins.append(float(probs[0] - probs[1]))
    return float(np.mean(margins)) if margins else 0.0


def mean_topk_entropy(top_logits: Sequence[Sequence[float]]) -> float:
    """Mean entropy of the normalized top-k support."""

    values: list[float] = []
    for row in top_logits:
        probs = _softmax(row)
        if probs.size:
            values.append(float(-np.sum(probs * np.log(probs + 1e-12))))
    return float(np.mean(values)) if values else 0.0


def prompt_features(prompt: str) -> dict[str, float]:
    """Eight cheap, model-independent prompt features from the paper."""

    lowered = prompt.lower()
    return {
        "text_n_chars": float(len(prompt)),
        "text_n_words": float(len(_WORDS.findall(prompt))),
        "text_n_digits": float(sum(char.isdigit() for char in prompt)),
        "text_n_ops": float(sum(char in "+-*/\u00d7\u00f7=" for char in prompt)),
        "text_n_qs": float(prompt.count("?")),
        "text_n_commas": float(prompt.count(",")),
        "text_n_lines": float(prompt.count("\n") + 1),
        "text_n_steps": float(sum(lowered.count(marker) for marker in _STEP_MARKERS)),
    }


def extract_features(prompt: str, greedy: Generation, *, sc_samples: int = 5) -> dict[str, float]:
    """Extract the complete 12-feature vector from one greedy pass."""

    full_entropy = float(np.mean(greedy.entropies)) if greedy.entropies else 0.0
    if not math.isfinite(full_entropy):
        full_entropy = 0.0
    features = {
        "greedy_conf": margin_confidence(greedy.top_logits),
        "greedy_top5_ent": mean_topk_entropy(greedy.top_logits),
        "greedy_mean_ent": full_entropy,
        "greedy_n_tokens": float(len(greedy.token_ids)),
        **prompt_features(prompt),
    }
    # Metadata used for reporting, deliberately excluded from model features.
    features["sc_samples"] = float(sc_samples)
    return features


def as_vector(features: dict[str, float]) -> np.ndarray:
    """Return the paper-defined feature ordering."""

    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
