from io import StringIO

from siftsc.terminal import AnimatedStatus, animations_enabled


def test_status_has_clean_fallback_for_non_tty_output() -> None:
    output = StringIO()

    with AnimatedStatus("Thinking", "Ready", stream=output):
        pass

    assert output.getvalue() == "Thinking…\nReady\n"


def test_animations_are_disabled_for_non_tty_streams() -> None:
    assert not animations_enabled(StringIO())
