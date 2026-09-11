"""Tiny dependency-free terminal presentation helpers."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from types import TracebackType
from typing import IO, Self, TextIO

_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
FULL_CELL = "█"
EMPTY_CELL = "░"


def animations_enabled(stream: TextIO) -> bool:
    """Return whether an animated status line is safe for this terminal."""

    try:
        is_tty = stream.isatty()
    except (AttributeError, ValueError):
        return False
    return (
        is_tty
        and os.environ.get("TERM") != "dumb"
        and "CI" not in os.environ
        and "SIFTSC_NO_ANIMATION" not in os.environ
    )


def terminal_width(default: int = 80, *, minimum: int = 60, maximum: int = 100) -> int:
    """Return a usable line width for wrapped text and charts."""

    columns = shutil.get_terminal_size((default, 24)).columns
    return max(minimum, min(maximum, columns))


def format_seconds(seconds: float) -> str:
    """Render a duration compactly: ``0.4s`` or ``1m 05s``."""

    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:02.0f}s"


def bar(value: float, maximum: float, width: int) -> str:
    """Draw a horizontal bar; ``value`` fills ``width`` cells when it equals ``maximum``."""

    if width <= 0:
        return ""
    if maximum <= 0 or value <= 0:
        filled = 0
    else:
        filled = round(width * min(value, maximum) / maximum)
        filled = max(1, filled)
    return FULL_CELL * filled + EMPTY_CELL * (width - filled)


def wrap_labeled(label: str, text: str, *, width: int, label_width: int) -> list[str]:
    """Wrap ``text`` after a fixed-width label with a hanging indent."""

    prefix = f"{label:<{label_width}}"
    lines = textwrap.wrap(
        " ".join(text.split()),
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * label_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return lines or [prefix.rstrip()]


class AnimatedStatus:
    """Render a compact spinner on TTYs and clean status lines elsewhere.

    ``success`` may contain ``{elapsed}``; it is replaced by the elapsed time. Call
    :meth:`update` to append live detail (for example download progress) to the
    animated line. Detail is only shown while animating so logs stay clean.
    """

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
        self.detail = ""
        self.elapsed = 0.0
        self._started = 0.0
        self._output: TextIO | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, detail: str) -> None:
        """Replace the live detail shown after the message."""

        self.detail = detail

    def _line(self, frame: str) -> str:
        text = f"{frame}  {self.message}"
        return f"{text} · {self.detail}" if self.detail else text

    def _render(self, frame: str) -> None:
        assert self._output is not None
        print(f"\r\033[2K{self._line(frame)}", end="", file=self._output, flush=True)

    def __enter__(self) -> Self:
        self._output = self.stream if self.stream is not None else sys.stderr
        self._started = time.perf_counter()
        use_animation = animations_enabled(self._output) if self.enabled is None else self.enabled
        self.enabled = use_animation
        if use_animation:
            self._render(_FRAMES[0])
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            print(f"{self.message}…", file=self._output)
        return self

    def _animate(self) -> None:
        index = 1
        while not self._stop.wait(0.09):
            self._render(_FRAMES[index % len(_FRAMES)])
            index += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        assert self._output is not None
        self.elapsed = time.perf_counter() - self._started
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            print("\r\033[2K", end="", file=self._output)
        if exc_type is None:
            success = self.success.replace("{elapsed}", format_seconds(self.elapsed))
            print(success, file=self._output, flush=True)
        elif issubclass(exc_type, KeyboardInterrupt):
            print(f"⏹  {self.message} · stopped", file=self._output, flush=True)
        else:
            print(f"❌ {self.message} failed", file=self._output, flush=True)


class QuietLibraryOutput:
    """Capture everything libraries print while ``console`` still reaches the user.

    Model libraries write progress bars, warnings, and even native-code messages to
    the process's standard streams, which garbles an animated status line. Inside
    this context both file descriptors are redirected to a temporary file, and
    ``console`` is a private copy of the original standard error for status
    messages. Streams without a real file descriptor (tests, ``StringIO``) fall
    back to Python-level redirection. The captured text is available afterwards
    as ``captured``.
    """

    def __init__(self) -> None:
        self.console: TextIO = sys.stderr
        self.captured = ""
        self._fd_mode = False
        self._targets: tuple[int, int] | None = None
        self._saved: tuple[int, int] | None = None
        self._sink: IO[bytes] | None = None
        self._console_file: TextIO | None = None
        self._buffer: io.StringIO | None = None
        self._redirects: list[contextlib.AbstractContextManager[io.StringIO]] = []

    def __enter__(self) -> Self:
        try:
            out_fd = sys.stdout.fileno()
            err_fd = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            self._enter_python_mode()
            return self
        sys.stdout.flush()
        sys.stderr.flush()
        encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        self._console_file = os.fdopen(os.dup(err_fd), "w", encoding=encoding, errors="replace")
        self._sink = tempfile.TemporaryFile()
        self._saved = (os.dup(out_fd), os.dup(err_fd))
        self._targets = (out_fd, err_fd)
        os.dup2(self._sink.fileno(), out_fd)
        os.dup2(self._sink.fileno(), err_fd)
        self._fd_mode = True
        self.console = self._console_file
        return self

    def _enter_python_mode(self) -> None:
        self._buffer = io.StringIO()
        self.console = sys.stderr
        self._redirects = [
            contextlib.redirect_stdout(self._buffer),
            contextlib.redirect_stderr(self._buffer),
        ]
        for redirect in self._redirects:
            redirect.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._fd_mode:
            for redirect in reversed(self._redirects):
                redirect.__exit__(exc_type, exc_value, traceback)
            assert self._buffer is not None
            self.captured = self._buffer.getvalue()
            return
        assert self._saved is not None
        assert self._targets is not None
        assert self._sink is not None
        assert self._console_file is not None
        with contextlib.suppress(OSError, ValueError):
            sys.stdout.flush()
        with contextlib.suppress(OSError, ValueError):
            sys.stderr.flush()
        saved_out, saved_err = self._saved
        out_fd, err_fd = self._targets
        os.dup2(saved_out, out_fd)
        os.dup2(saved_err, err_fd)
        os.close(saved_out)
        os.close(saved_err)
        self._console_file.flush()
        self._console_file.close()
        self._sink.seek(0)
        self.captured = self._sink.read().decode("utf-8", "replace")
        self._sink.close()
