from __future__ import annotations

from siftsc.cli import main


def test_profiles_command_lists_bundled_profiles(capsys) -> None:
    assert main(["profiles"]) == 0
    output = capsys.readouterr().out
    assert "qwen05b-q4-confidence" in output
