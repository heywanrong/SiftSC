"""Inference backends. MLX is imported lazily so the core package stays portable."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from .types import Generation


class Backend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        greedy: bool,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generation: ...


@dataclass(slots=True)
class MLXBackend:
    """Local Apple-Silicon inference with top-logit capture for gating."""

    model_path: str
    top_k: int = 5
    use_chat_template: bool = False
    revision: str | None = None
    stop_sequences: tuple[str, ...] = (
        "\n\nAnswer the following",
        "\n\nQuestion:",
        "\n\nProblem:",
    )
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Load a local model or download a Hugging Face model into its cache."""

        self._load()

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError(
                    "MLX support is optional. On macOS, run: pip install 'siftsc[mlx]'"
                ) from exc
            if self.revision is None:
                loaded = load(self.model_path)
            else:
                loaded = load(self.model_path, revision=self.revision)
            self._model, self._tokenizer = cast(tuple[Any, Any], loaded)
        return self._model, self._tokenizer

    def _format_prompt(self, prompt: str, tokenizer: Any) -> str:
        if not self.use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        return cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

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
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._load()
        formatted = self._format_prompt(prompt, tokenizer)
        mx.random.seed(seed)
        prompt_ids = tokenizer.encode(formatted)
        cache = make_prompt_cache(model)
        logits = model(mx.array(prompt_ids)[None], cache=cache)[:, -1, :]
        sampler = make_sampler(temp=0.0 if greedy else temperature, top_p=top_p)

        output_ids: list[int] = []
        top_logits: list[tuple[float, ...]] = []
        entropies: list[float] = []
        chosen_logprobs: list[float] = []
        stop_at: int | None = None
        for _ in range(max_tokens):
            top_ids = mx.argpartition(-logits[0], self.top_k)[: self.top_k]
            top_ids = top_ids[mx.argsort(-logits[0][top_ids])]
            top_values = logits[0][top_ids]
            top_logits.append(tuple(float(value) for value in top_values.tolist()))

            probabilities = mx.softmax(logits[0].astype(mx.float32))
            entropies.append(float(-(probabilities * mx.log(probabilities + 1e-12)).sum()))
            token_array = sampler(logits)
            token_id = int(cast(Any, token_array.item()))
            chosen_logprobs.append(math.log(float(probabilities[token_id]) + 1e-12))
            output_ids.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            partial_text = cast(str, tokenizer.decode(output_ids))
            positions = [partial_text.find(marker) for marker in self.stop_sequences]
            valid_positions = [position for position in positions if position >= 0]
            if valid_positions:
                stop_at = min(valid_positions)
                break
            logits = model(token_array[None], cache=cache)[:, -1, :]

        text = cast(str, tokenizer.decode(output_ids))
        if stop_at is not None:
            text = text[:stop_at]
        return Generation(
            text=text,
            token_ids=tuple(output_ids),
            top_logits=tuple(top_logits),
            entropies=tuple(entropies),
            sum_logprob=float(sum(chosen_logprobs)),
            metadata={
                "backend": "mlx",
                "model": self.model_path,
                "revision": self.revision,
                "seed": seed,
                "greedy": greedy,
            },
        )
