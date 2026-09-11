"""Selective self-consistency orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import Backend
from .features import extract_features
from .gates import Gate
from .types import GateDecision, Generation, SiftResult
from .voting import AnswerParser, parse_answer, plurality_vote


@dataclass(slots=True)
class SiftSC:
    """Pay for extra reasoning traces only when the gate requests them."""

    backend: Backend
    gate: Gate
    samples: int = 5
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 1
    parser: AnswerParser = parse_answer

    def __post_init__(self) -> None:
        if self.samples < 2:
            raise ValueError("samples must be at least 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def __call__(self, prompt: str, *, feature_text: str | None = None) -> SiftResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        greedy = self.backend.generate(
            prompt,
            greedy=True,
            seed=0,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        feature_source = feature_text if feature_text is not None else prompt
        features = extract_features(feature_source, greedy, sc_samples=self.samples)
        score = self.gate.score(features)
        invoke = score >= self.gate.threshold
        decision = GateDecision(
            invoke_self_consistency=invoke,
            score=score,
            threshold=self.gate.threshold,
            profile=self.gate.name,
            features=features,
        )
        if not invoke:
            parsed = self.parser(greedy.text)
            counts = {parsed: 1} if parsed is not None else {}
            return SiftResult(
                text=greedy.text,
                parsed_answer=parsed,
                used_self_consistency=False,
                decision=decision,
                generations=(greedy,),
                vote_counts=counts,
                generation_passes=1,
            )

        samples = self.sample_traces(prompt)
        winner, parsed, counts = plurality_vote(samples, self.parser)
        return SiftResult(
            text=winner.text,
            parsed_answer=parsed,
            used_self_consistency=True,
            decision=decision,
            generations=(greedy, *samples),
            vote_counts=counts,
            generation_passes=1 + len(samples),
        )

    def sample_traces(self, prompt: str) -> tuple[Generation, ...]:
        """Draw the fresh stochastic voter traces used when SC is invoked.

        This matches the paper's SC@N protocol exactly: the gate observes one
        greedy draft, then N independent stochastic traces vote. Keeping the draft
        out of the vote also makes offline paper examples reproducible in the
        packaged tool. Compare mode calls this directly so an always-SC baseline
        uses the very same seeds.
        """

        return tuple(
            self.backend.generate(
                prompt,
                greedy=False,
                seed=self.seed * 10_000 + index + 1,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            for index in range(self.samples)
        )
