"""Full-screen terminal interface built on Textual (the optional ``ui`` extra).

The model is loaded before the app starts so loading messages stay in the normal
terminal. Inside the app, generation runs in a worker thread while the screen
keeps a conversation log on the left and live cost panels on the right.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from .display import (
    POLICY_DESCRIPTIONS,
    POLICY_ICONS,
    SIFTSC,
    compact_session_rows,
    session_chart,
    session_line,
)
from .engine import MODES, Engine, TurnReport, parse_input, status_message

_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
EXAMPLE_QUESTION = "If 3 notebooks cost £4 each, what is the total?"


class SiftApp(App[None]):
    """Sifty's full-screen chat."""

    TITLE = "SiftSC"
    CSS = """
    #body { height: 1fr; }
    #conversation { width: 1fr; border: round $primary; padding: 0 1; }
    #side { width: 34; border: round $secondary; padding: 0 1; }
    #side Static { margin-bottom: 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+t", "cycle_mode", "Mode"),
        Binding("ctrl+g", "stats", "Session"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, engine: Engine, *, mode: str = SIFTSC, demo_threshold: bool = False) -> None:
        super().__init__()
        self.engine = engine
        self.policy = mode
        self.demo_threshold = demo_threshold
        self.busy = False
        self._frame = 0
        self._status_text = ""

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield RichLog(id="conversation", wrap=True, markup=False, highlight=False, min_width=20)
            with Vertical(id="side"):
                yield Static(id="chart")
                yield Static(id="session")
        yield Static("", id="status")
        yield Input(
            placeholder="Ask a question · Enter to send · /help for commands", id="question"
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_titles()
        self._write_welcome()
        self._refresh_side(None)
        self.set_interval(0.1, self._tick)
        self.query_one("#question", Input).focus()

    # ----------------------------------------------------------------- helpers

    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)

    def _write(self, line: str | Text) -> None:
        self._conversation().write(line)

    def _refresh_titles(self) -> None:
        icon, description = POLICY_ICONS[self.policy], POLICY_DESCRIPTIONS[self.policy]
        self.sub_title = f"{icon} {self.policy} · {description}"

    def _write_welcome(self) -> None:
        self._write(Text("🚀 Sifty is online", style="bold"))
        self._write("💡 Word problems are routed and voted · other questions get one chat answer")
        self._write("   Commands: /plain /siftsc /compare /stats /clear /help /exit")
        self._write("   Prefix one question with /math or /talk to force its route")
        self._write(f'   Try: "{EXAMPLE_QUESTION}"')
        if self.demo_threshold:
            self._write("🧭 Public-model routing uses the reproducible demo threshold (0.84).")
        self._write("")

    def _write_help(self) -> None:
        for mode in MODES:
            self._write(f"  /{mode:<8} {POLICY_ICONS[mode]} {POLICY_DESCRIPTIONS[mode]}")
        self._write("  /math <q> 🧮 force the paper's reasoning route for one question")
        self._write("  /talk <q> 💬 force a one-pass chat-style answer for one question")
        self._write("  /stats    📊 show the session compute chart")
        self._write("  /clear    🧹 clear the conversation")
        self._write("  /exit     👋 leave SiftSC")
        self._write("")

    def _refresh_side(self, report: TurnReport | None) -> None:
        chart = Text("📈 This question\n", style="bold")
        if report is None:
            chart.append("ask something to see its cost", style="dim")
        else:
            chart.append("\n".join(report.side))
        self.query_one("#chart", Static).update(chart)
        session = Text("📊 Session\n", style="bold")
        session.append("\n".join(compact_session_rows(self.engine.stats)))
        self.query_one("#session", Static).update(session)

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", Static).update(text)

    def _tick(self) -> None:
        if not self.busy:
            return
        self._frame = (self._frame + 1) % len(_FRAMES)
        self.query_one("#status", Static).update(f"{_FRAMES[self._frame]}  {self._status_text}")

    def _set_mode(self, mode: str) -> None:
        self.policy = mode
        self._refresh_titles()
        self._write(f"🔀 Mode switched · {POLICY_ICONS[mode]} {mode} · {POLICY_DESCRIPTIONS[mode]}")
        self._write("")

    # ---------------------------------------------------------------- actions

    def action_cycle_mode(self) -> None:
        self._set_mode(MODES[(MODES.index(self.policy) + 1) % len(MODES)])

    def action_stats(self) -> None:
        for line in session_chart(self.engine.stats, width=80):
            self._write(line)
        self._write("")

    def action_clear_log(self) -> None:
        self._conversation().clear()
        self._write_welcome()

    # ------------------------------------------------------------------ input

    @on(Input.Submitted, "#question")
    def _submitted(self, event: Input.Submitted) -> None:
        typed = event.value
        event.input.value = ""
        if self.busy:
            return
        entry = parse_input(typed)
        if entry.command == "exit":
            self.exit()
        elif entry.command == "mode" and entry.argument is not None:
            self._set_mode(entry.argument)
        elif entry.command == "stats":
            self.action_stats()
        elif entry.command == "clear":
            self.action_clear_log()
        elif entry.command == "help":
            self._write_help()
        elif entry.command == "unknown":
            self._write(f"🤔 Unknown command {entry.argument} · type /help to see the list")
            self._write("")
        elif entry.question is not None:
            self._ask(entry.question, entry.kind)

    def _ask(self, question: str, kind: str | None) -> None:
        resolved = self.engine.kind_of(question, kind)
        self.busy = True
        self._set_status(status_message(self.policy, resolved))
        self._write(Text(f"💬 you · {self.policy}", style="bold cyan"))
        self._write(f"   {question}")
        self._write("")
        self._run_answer(question, resolved)

    @work(thread=True, exclusive=True)
    def _run_answer(self, question: str, kind: str) -> None:
        try:
            report = self.engine.answer(self.policy, question, kind=kind)
        except Exception as exc:  # keep the app alive and show what happened
            self.call_from_thread(self._show_failure, exc)
            return
        self.call_from_thread(self._show_report, report)

    def _show_report(self, report: TurnReport) -> None:
        self._write(Text("🤖 Sifty", style="bold green"))
        for line in report.text.strip().splitlines():
            if line.strip():
                self._write(f"   {line.strip()}")
        self._write("")
        self._write(Text(report.headline, style="bold yellow"))
        for note in report.notes:
            self._write(Text(note, style="dim"))
        if report.summary:
            self._write(f"   {report.summary}")
        for line in report.side:
            self._write(f"   {line}")
        self._write(Text(session_line(self.engine.stats), style="dim"))
        self._write("")
        self._refresh_side(report)
        self._idle()

    def _show_failure(self, exc: BaseException) -> None:
        self.engine.discard_partial_cost()
        self._write(Text(f"❌ Sifty hit an error: {exc}", style="bold red"))
        self._write("")
        self._idle()

    def _idle(self) -> None:
        self.busy = False
        self._set_status("")
        self.query_one("#question", Input).focus()


def run_app(engine: Engine, *, mode: str = SIFTSC, demo_threshold: bool = False) -> int:
    """Run the full-screen interface until the user quits."""

    SiftApp(engine, mode=mode, demo_threshold=demo_threshold).run()
    return 0
