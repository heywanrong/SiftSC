from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from siftsc import ConfidenceGate, Generation, SiftSC


@dataclass
class FakeBackend:
    answers: list[str]
    confidence_logits: tuple[tuple[float, ...], ...]
    calls: list[bool] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        *,
        greedy: bool,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generation:
        index = len(self.calls)
        self.calls.append(greedy)
        return Generation(
            text=f"The answer is {self.answers[index]}.",
            token_ids=(1, 2),
            top_logits=self.confidence_logits,
            entropies=(0.2, 0.3),
            sum_logprob=-float(index),
        )


def test_high_confidence_skips_extra_calls() -> None:
    backend = FakeBackend(["4"], ((10.0, 0.0),))
    gate = ConfidenceGate(threshold=0.5)
    result = SiftSC(backend, gate, samples=5)("2 + 2?")
    assert result.parsed_answer == "4"
    assert not result.used_self_consistency
    assert result.generation_passes == 1
    assert backend.calls == [True]


def test_low_confidence_invokes_and_votes() -> None:
    backend = FakeBackend(["4", "5", "5", "5", "4", "5"], ((0.0, 0.0),))
    gate = ConfidenceGate(threshold=0.5)
    result = SiftSC(backend, gate, samples=5)("2 + 2?")
    assert result.parsed_answer == "5"
    assert result.used_self_consistency
    assert result.generation_passes == 6
    assert backend.calls == [True, False, False, False, False, False]
    assert result.vote_counts == {"4": 1, "5": 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"samples": 1}, "at least 2"), ({"max_tokens": 0}, "positive")],
)
def test_router_validates_configuration(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SiftSC(FakeBackend(["4"], ((1.0, 0.0),)), ConfidenceGate(0.5), **kwargs)


def test_router_rejects_empty_prompt() -> None:
    runner = SiftSC(FakeBackend(["4"], ((1.0, 0.0),)), ConfidenceGate(0.5))
    with pytest.raises(ValueError, match="must not be empty"):
        runner("  ")


def test_sample_traces_uses_the_same_voter_seeds_as_escalation() -> None:
    seeds: list[int] = []

    class SeedRecorder(FakeBackend):
        def generate(self, prompt: str, **kwargs: object) -> Generation:  # type: ignore[override]
            seeds.append(int(kwargs["seed"]))  # type: ignore[call-overload]
            return super().generate(prompt, **kwargs)  # type: ignore[arg-type]

    backend = SeedRecorder(["4", "5", "5", "5", "4", "5"] + ["5"] * 5, ((0.0, 0.0),))
    runner = SiftSC(backend, ConfidenceGate(threshold=0.5), samples=5, seed=1)
    runner("2 + 2?")
    escalation_seeds = seeds[1:]
    seeds.clear()

    traces = runner.sample_traces("2 + 2?")

    assert len(traces) == 5
    assert seeds == escalation_seeds == [10001, 10002, 10003, 10004, 10005]
    assert backend.calls[-5:] == [False] * 5
