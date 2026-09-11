"""Public data structures used by SiftSC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Generation:
    """One model completion and the statistics needed by a gate."""

    text: str
    token_ids: tuple[int, ...] = ()
    top_logits: tuple[tuple[float, ...], ...] = ()
    entropies: tuple[float, ...] = ()
    sum_logprob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The gate's decision after the first, deterministic pass."""

    invoke_self_consistency: bool
    score: float
    threshold: float
    profile: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class SiftResult:
    """Final answer plus an auditable account of the compute spent."""

    text: str
    parsed_answer: str | None
    used_self_consistency: bool
    decision: GateDecision
    generations: tuple[Generation, ...]
    vote_counts: dict[str, int]
    generation_passes: int

    @property
    def saved_passes(self) -> int:
        """Generation passes saved when SC is skipped."""

        return max(0, int(self.decision.features.get("sc_samples", 1)) - self.generation_passes)
