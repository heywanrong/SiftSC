"""Small, auditable gates for selective self-consistency."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Protocol, Self

import numpy as np

from .features import FEATURE_NAMES, as_vector


class Gate(Protocol):
    """A score above the threshold means: spend compute on SC."""

    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    def score(self, features: dict[str, float]) -> float: ...

    def with_threshold(self, threshold: float) -> Self: ...


@dataclass(frozen=True, slots=True)
class ConfidenceGate:
    """Invoke SC when the greedy token-margin confidence is low."""

    threshold: float
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    name: str = "confidence"

    def score(self, features: dict[str, float]) -> float:
        confidence = features["greedy_conf"]
        span = max(self.confidence_max - self.confidence_min, 1e-12)
        normalized = (confidence - self.confidence_min) / span
        return float(np.clip(1.0 - normalized, 0.0, 1.0))

    def with_threshold(self, threshold: float) -> Self:
        return replace(self, threshold=float(threshold))


@dataclass(frozen=True, slots=True)
class LogisticGate:
    """The paper's standardized 12-feature logistic gate."""

    weights: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    threshold: float
    name: str = "logistic"

    def score(self, features: dict[str, float]) -> float:
        vector = as_vector(features)
        if len(vector) != len(self.weights):
            raise ValueError("profile feature count does not match SiftSC's feature schema")
        scale = np.asarray(self.scale, dtype=np.float64)
        standardized = (vector - np.asarray(self.mean, dtype=np.float64)) / np.where(
            scale == 0.0, 1.0, scale
        )
        logit = float(np.dot(np.asarray(self.weights), standardized) + self.intercept)
        if logit >= 0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exp_logit = math.exp(logit)
        return float(exp_logit / (1.0 + exp_logit))

    def with_threshold(self, threshold: float) -> Self:
        return replace(self, threshold=float(threshold))


def load_profile(name_or_path: str) -> Gate:
    """Load a bundled profile name or a compatible JSON file."""

    candidate = Path(name_or_path)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        filename = name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        profile = resources.files("siftsc.profiles").joinpath(filename)
        try:
            payload = json.loads(profile.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            available = ", ".join(list_profiles())
            raise ValueError(f"unknown profile {name_or_path!r}; available: {available}") from exc

    gate_type = payload["gate_type"]
    if gate_type == "confidence":
        return ConfidenceGate(
            threshold=float(payload["threshold"]),
            confidence_min=float(payload["confidence_min"]),
            confidence_max=float(payload["confidence_max"]),
            name=str(payload.get("name", name_or_path)),
        )
    if gate_type == "logistic":
        profile_features = payload.get("feature_names")
        if profile_features is not None and tuple(profile_features) != FEATURE_NAMES:
            raise ValueError("profile feature_names do not match this SiftSC version")
        sizes = {len(payload[key]) for key in ("weights", "mean", "scale")}
        if sizes != {len(FEATURE_NAMES)}:
            raise ValueError("profile weights, mean, and scale must match the feature schema")
        return LogisticGate(
            weights=tuple(float(value) for value in payload["weights"]),
            intercept=float(payload["intercept"]),
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            threshold=float(payload["threshold"]),
            name=str(payload.get("name", name_or_path)),
        )
    raise ValueError(f"unsupported gate_type: {gate_type!r}")


def list_profiles() -> list[str]:
    """Return bundled profile names."""

    root = resources.files("siftsc.profiles")
    return sorted(
        path.name.removesuffix(".json") for path in root.iterdir() if path.name.endswith(".json")
    )
