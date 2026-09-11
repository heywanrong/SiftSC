"""Tiny dependency-free terminal presentation helpers."""

from __future__ import annotations

import os
import sys
import threading
from types import TracebackType
from typing import Self, TextIO

_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def animations_enabled(stream: TextIO) -> bool:
    """Return whether an animated status line is safe for this terminal."""

    return (
        stream.isatty()
        and os.environ.get("TERM") != "dumb"
        and "CI" not in os.environ
        and "SIFTSC_NO_ANIMATION" not in os.environ
    )


class AnimatedStatus:
    """Render a compact spinner on TTYs and clean status lines elsewhere."""

    def __init__(
        self,
        message: str,
        success: str,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.message = message
        self.success = success
        self.stream = stream
        self.enabled = enabled
        self._output: TextIO | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self._output = self.stream if self.stream is not None else sys.stderr
        use_animation = animations_enabled(self._output) if self.enabled is None else self.enabled
        self.enabled = use_animation
        if use_animation:
            print(f"\r{_FRAMES[0]}  {self.message}", end="", file=self._output, flush=True)
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            print(f"{self.message}…", file=self._output)
        return self

    def _animate(self) -> None:
        assert self._output is not None
        index = 1
        while not self._stop.wait(0.09):
            frame = _FRAMES[index % len(_FRAMES)]
            print(f"\r{frame}  {self.message}", end="", file=self._output, flush=True)
            index += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        assert self._output is not None
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            print("\r\033[2K", end="", file=self._output)
        if exc_type is None:
            print(self.success, file=self._output)
        else:
            print(f"❌ {self.message} failed", file=self._output)
