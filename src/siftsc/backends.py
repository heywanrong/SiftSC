"""Inference backends. MLX is imported lazily so the core package stays portable."""

from __future__ import annotations

import fnmatch
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from .types import Generation

# The file patterns mlx-lm fetches for a Hub repository; used for the cache check.
HF_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)

ProgressCallback = Callable[[int, int | None], None]

_FINAL_ANSWER_LINE = re.compile(
    r"(?i)(?:final[ \t]+)?(?:the[ \t]+)?answer(?:[ \t]+is|:)[ \t]*[^\n]+"
)


def _completion_stop_position(text: str, stop_sequences: tuple[str, ...]) -> int | None:
    """Find the earliest model turn boundary."""

    positions = [text.find(marker) for marker in stop_sequences]
    valid_positions = [position for position in positions if position >= 0]
    return min(valid_positions) if valid_positions else None


def _clean_completion_text(text: str) -> str:
    """Hide repetition after a completed answer without changing token statistics."""

    answer_match = _FINAL_ANSWER_LINE.search(text)
    if answer_match is not None and text[answer_match.end() :].strip():
        return text[: answer_match.end()]
    return text


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


@dataclass(frozen=True, slots=True)
class GenerationCost:
    """Wall-clock seconds and generated tokens for one generation pass."""

    seconds: float
    tokens: int


@dataclass(slots=True)
class MeteredBackend:
    """Record the measured cost of every pass without changing any output."""

    backend: Backend
    _costs: list[GenerationCost] = field(default_factory=list, init=False, repr=False)

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
        started = time.perf_counter()
        generation = self.backend.generate(
            prompt,
            greedy=greedy,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.perf_counter() - started
        self._costs.append(GenerationCost(seconds=elapsed, tokens=len(generation.token_ids)))
        return generation

    def take(self) -> list[GenerationCost]:
        """Return the costs recorded since the previous call and reset the record."""

        costs, self._costs = self._costs, []
        return costs


def _cache_folder_bytes(folder: Path) -> int:
    total = 0
    if not folder.exists():
        return 0
    for path in folder.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _remote_model_bytes(repo: str, revision: str | None) -> int | None:
    """Best-effort total size of the files mlx-lm will download."""

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception:  # size is optional; the download proceeds without it
        return None
    total = 0
    for sibling in info.siblings or []:
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in HF_ALLOW_PATTERNS):
            total += sibling.size or 0
    return total or None


@dataclass(slots=True)
class MLXBackend:
    """Local Apple-Silicon inference with top-logit capture for gating."""

    model_path: str
    top_k: int = 5
    use_chat_template: bool = False
    revision: str | None = None
    stop_sequences: tuple[str, ...] = (
        "<|endoftext|>",
        "<|im_start|>user",
        "\nHuman:",
        "\nUser:",
        "\n\nAnswer the following",
        "\n\nQuestion:",
        "\n\nProblem:",
    )
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Load a local model or download a Hugging Face model into its cache."""

        self._load()

    def _hub_repo(self) -> str | None:
        """Return the Hub repository id, or ``None`` for a local model directory."""

        if Path(self.model_path).exists() or "/" not in self.model_path:
            return None
        return self.model_path

    def is_cached(self) -> bool:
        """Return whether every file mlx-lm needs is already in the local Hub cache."""

        repo = self._hub_repo()
        if repo is None:
            return True
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return True
        try:
            snapshot_download(
                repo,
                revision=self.revision,
                allow_patterns=list(HF_ALLOW_PATTERNS),
                local_files_only=True,
            )
        except Exception:  # any cache miss means a download is needed
            return False
        return True

    def ensure_downloaded(self, progress: ProgressCallback | None = None) -> None:
        """Download the model files into the Hub cache, reporting bytes as they land."""

        repo = self._hub_repo()
        if repo is None:
            return
        from huggingface_hub import constants, snapshot_download

        total = _remote_model_bytes(repo, self.revision)
        folder = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        stop = threading.Event()

        def poll() -> None:
            assert progress is not None
            while not stop.wait(0.5):
                progress(_cache_folder_bytes(folder), total)

        thread: threading.Thread | None = None
        if progress is not None:
            thread = threading.Thread(target=poll, daemon=True)
            thread.start()
        try:
            snapshot_download(repo, revision=self.revision, allow_patterns=list(HF_ALLOW_PATTERNS))
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=1.0)
        if progress is not None:
            progress(_cache_folder_bytes(folder), total)

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
            stop_at = _completion_stop_position(partial_text, self.stop_sequences)
            if stop_at is not None:
                break
            logits = model(token_array[None], cache=cache)[:, -1, :]

        text = cast(str, tokenizer.decode(output_ids))
        if stop_at is not None:
            text = text[:stop_at]
        text = _clean_completion_text(text)
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
