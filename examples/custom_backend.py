"""Minimal backend adapter example with deterministic fake outputs."""

from __future__ import annotations

from siftsc import ConfidenceGate, Generation, SiftSC


class ExampleBackend:
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
        del prompt, seed, max_tokens, temperature, top_p
        logits = ((8.0, 0.0),) if greedy else ((2.0, 1.0),)
        return Generation("The answer is 42.", token_ids=(42,), top_logits=logits)


result = SiftSC(ExampleBackend(), ConfidenceGate(threshold=0.5))("What is 6 \u00d7 7?")
print(result.parsed_answer, result.generation_passes)
