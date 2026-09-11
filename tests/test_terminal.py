from __future__ import annotations

import subprocess
import sys
import textwrap
from io import StringIO

from siftsc.terminal import (
    AnimatedStatus,
    QuietLibraryOutput,
    animations_enabled,
    bar,
    wrap_labeled,
)


def test_status_has_clean_fallback_for_non_tty_output() -> None:
    output = StringIO()

    with AnimatedStatus("Thinking", "Ready", stream=output):
        pass

    assert output.getvalue() == "Thinking…\nReady\n"


def test_status_formats_elapsed_time_and_ignores_detail_when_not_animated() -> None:
    output = StringIO()

    with AnimatedStatus("Loading", "Ready · {elapsed}", stream=output) as status:
        status.update("12 MB")

    text = output.getvalue()
    assert text.startswith("Loading…\nReady · ")
    assert text.rstrip().endswith("s")
    assert "12 MB" not in text


def test_animations_are_disabled_for_non_tty_streams() -> None:
    assert not animations_enabled(StringIO())


def test_bar_fills_one_cell_per_unit() -> None:
    assert bar(1, 6, 6) == "█░░░░░"
    assert bar(6, 6, 6) == "██████"
    assert bar(0, 6, 6) == "░░░░░░"
    assert bar(9, 6, 6) == "██████"
    assert bar(415, 2000, 20) == "████░░░░░░░░░░░░░░░░"


def test_wrap_labeled_uses_hanging_indent() -> None:
    lines = wrap_labeled("Question", "word " * 30, width=40, label_width=10)

    assert lines[0].startswith("Question  word")
    assert all(line.startswith(" " * 10) for line in lines[1:])
    assert all(len(line) <= 40 for line in lines)


def test_quiet_output_captures_python_level_noise_in_fallback_mode(capsys) -> None:
    with QuietLibraryOutput() as quiet:
        print("stdout noise")
        print("stderr noise", file=sys.stderr)
        print("visible", file=quiet.console)

    captured = capsys.readouterr()
    assert "stdout noise" in quiet.captured
    assert "stderr noise" in quiet.captured
    assert "noise" not in captured.out
    assert "noise" not in captured.err
    assert "visible" in captured.err


def test_quiet_output_captures_native_writes_in_fd_mode() -> None:
    script = textwrap.dedent(
        """
        import os, sys
        from siftsc.terminal import QuietLibraryOutput
        with QuietLibraryOutput() as quiet:
            os.write(2, b"native noise\\n")
            os.write(1, b"native stdout noise\\n")
            print("python noise", file=sys.stderr)
            print("visible", file=quiet.console, flush=True)
        print("after", file=sys.stderr)
        print("CAPTURED:" + repr(quiet.captured))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert completed.stderr == "visible\nafter\n"
    assert completed.stdout.startswith("CAPTURED:")
    assert "native noise" in completed.stdout
    assert "native stdout noise" in completed.stdout
    assert "python noise" in completed.stdout
