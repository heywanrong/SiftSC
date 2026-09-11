from __future__ import annotations

import asyncio

from test_cli import FakeMLXBackend

import siftsc.cli as cli
from siftsc.display import (
    CompareRow,
    SessionStats,
    TurnCost,
    compact_compare_rows,
    compact_cost_rows,
    compact_session_rows,
)
from siftsc.engine import Engine
from siftsc.tui import SiftApp


def _engine(backend: FakeMLXBackend) -> Engine:
    args = cli.build_parser().parse_args(["ui", "--model", "fake-model"])
    return Engine.create(cli._chat_runtime_args(args), backend)


def _log_text(app: SiftApp) -> str:
    return "\n".join(strip.text for strip in app._conversation().lines)


def _static_text(app: SiftApp, selector: str) -> str:
    widget = app.query_one(selector)
    for attribute in ("content", "renderable", "visual"):
        value = getattr(widget, attribute, None)
        if value is not None:
            return str(value)
    raise AssertionError(f"cannot read the text of {selector}")


def test_compact_rows_fit_a_narrow_panel() -> None:
    rows = compact_cost_rows(6, 5)
    assert rows[0].startswith("⚡ plain") and rows[0].endswith("1")
    assert rows[2].endswith("6") and "██████" in rows[2]
    assert "not run" in compact_cost_rows(1, 5, mode="plain")[2]
    compare = compact_compare_rows(
        [CompareRow("plain", "15", 1, 0.1), CompareRow("siftsc", "25", 6, 0.9)], samples=5
    )
    assert compare[0].endswith("1 → 15") and compare[1].endswith("6 → 25")
    stats = SessionStats(samples=5)
    assert compact_session_rows(stats) == ["no questions yet"]
    stats.record(TurnCost(passes=6, seconds=1.5, tokens=120), voted=True)
    session = compact_session_rows(stats)
    assert session[0] == "1 question · 1 vote"
    assert "spent 6 · always-SC 5" in session[1]
    assert all(len(row) <= 32 for row in rows + compare + session)


def test_app_answers_a_question_and_updates_the_side_panel() -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    app = SiftApp(_engine(backend), mode="siftsc")

    async def scenario() -> None:
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.click("#question")
            await pilot.press(*"What is 7 + 5?", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = _log_text(app)
            assert "💬 you · siftsc" in text
            assert "🎯 Answer: 12" in text
            assert "🛡️ vote skipped · ✅ saved 4 (80.0%)" in text
            assert "📊 session · 1 question" in text
            assert "🧭 siftsc" in _static_text(app, "#chart")
            assert "1 question" in _static_text(app, "#session")
            assert app.busy is False

    asyncio.run(scenario())
    assert backend.calls == 1


def test_app_commands_switch_mode_show_help_and_exit() -> None:
    backend = FakeMLXBackend("fake-model", answers=["12"], confident_after=0)
    app = SiftApp(_engine(backend), mode="siftsc")

    async def scenario() -> None:
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.click("#question")
            await pilot.press(*"/plain", "enter")
            assert app.policy == "plain"
            assert "plain" in app.sub_title
            await pilot.press("ctrl+t")
            assert app.policy == "siftsc"
            await pilot.press("ctrl+t")
            assert app.policy == "compare"
            await pilot.press(*"/help", "enter")
            await pilot.press(*"/bogus", "enter")
            await pilot.pause()
            text = _log_text(app)
            assert "/compare" in text
            assert "Unknown command /bogus" in text
            await pilot.press(*"/exit", "enter")
            await pilot.pause()

    asyncio.run(scenario())
    assert backend.calls == 0


def test_app_general_question_shows_chat_style_note() -> None:
    backend = FakeMLXBackend("fake-model", answers=["Beijing"], confident_after=0)
    app = SiftApp(_engine(backend), mode="compare")

    async def scenario() -> None:
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.click("#question")
            await pilot.press(*"Where is Beijing?", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = _log_text(app)
            assert "💬 General question" in text
            assert "Compare skipped" in text
            assert "general question" in _static_text(app, "#chart")

    asyncio.run(scenario())
    assert backend.calls == 1
