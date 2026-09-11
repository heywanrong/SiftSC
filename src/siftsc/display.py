"""Pure text formatting for the CLI: answer blocks, cost charts, and tables.

Every function returns a list of lines so the CLI decides where to print them and
tests can check the exact wording. One chart cell is one generation pass, so bar
length is the compute cost.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from .terminal import bar, format_seconds

PLAIN = "plain"
ALWAYS_SC = "always-SC"
SIFTSC = "siftsc"
COMPARE = "compare"

# Only East-Asian-Width "wide" emoji are used in aligned columns so rows line up.
POLICY_ICONS = {PLAIN: "⚡", ALWAYS_SC: "👥", SIFTSC: "🧭", COMPARE: "🔬"}
POLICY_DESCRIPTIONS = {
    PLAIN: "one deterministic pass, never votes",
    SIFTSC: "votes only when the router escalates",
    COMPARE: "runs plain, always-SC and siftsc on every question",
}
_LABEL_WIDTH = 10
_INDENT = "   "


@dataclass(frozen=True, slots=True)
class TurnCost:
    """Measured cost of one answered question."""

    passes: int
    seconds: float
    tokens: int


@dataclass(slots=True)
class SessionStats:
    """Running compute totals for an interactive session."""

    samples: int = 5
    questions: int = 0
    passes: int = 0
    seconds: float = 0.0
    tokens: int = 0
    votes: int = 0

    def record(self, cost: TurnCost, *, voted: bool) -> None:
        self.questions += 1
        self.passes += cost.passes
        self.seconds += cost.seconds
        self.tokens += cost.tokens
        if voted:
            self.votes += 1

    @property
    def plain_passes(self) -> int:
        return self.questions

    @property
    def always_sc_passes(self) -> int:
        return self.questions * self.samples

    @property
    def saved(self) -> int:
        return self.always_sc_passes - self.passes


@dataclass(frozen=True, slots=True)
class CompareRow:
    """One policy's outcome for the same question."""

    policy: str
    answer: str | None
    passes: int
    seconds: float
    note: str = ""


def _plural(count: int, singular: str = "pass", plural: str = "passes") -> str:
    return singular if count == 1 else plural


def _questions(count: int) -> str:
    return f"{count} {'question' if count == 1 else 'questions'}"


def saving_words(actual: int, baseline: int) -> str:
    """Describe ``actual`` passes against ``baseline`` passes."""

    saved = baseline - actual
    if baseline <= 0:
        return "no baseline"
    if saved > 0:
        return f"saved {saved} ({saved / baseline:.1%})"
    if saved < 0:
        return f"extra {-saved} (+{-saved / baseline:.1%})"
    return "same cost"


def _saving_icon(actual: int, baseline: int) -> str:
    if baseline - actual > 0:
        return "✅"
    if baseline - actual < 0:
        return "⚠️"
    return "🔸"


def _chart_row(policy: str, passes: int, width: int, note: str = "") -> str:
    label = f"{POLICY_ICONS[policy]} {policy:<{_LABEL_WIDTH}}"
    line = f"{_INDENT}{label} {bar(passes, width, width)}  {passes:>2} {_plural(passes):<6}"
    return f"{line} {note}".rstrip()


def compute_chart(
    *,
    actual_passes: int,
    samples: int,
    seconds: float | None = None,
    draft_note: str = "",
    actual_note: str = "",
    mode: str = SIFTSC,
) -> list[str]:
    """Chart one question's cost under plain, always-SC, and the policy that ran."""

    width = samples + 1
    timing = f" · {format_seconds(seconds)}" if seconds is not None else ""
    if mode == PLAIN:
        note = f"{_saving_icon(1, samples)} this run · {saving_words(1, samples)}{timing}"
        return [
            _chart_row(PLAIN, 1, width, note),
            _chart_row(ALWAYS_SC, samples, width),
            f"{_INDENT}{POLICY_ICONS[SIFTSC]} {SIFTSC:<{_LABEL_WIDTH}} "
            "not run · /siftsc lets the router decide",
        ]
    note = f"{_saving_icon(actual_passes, samples)} {saving_words(actual_passes, samples)}{timing}"
    if actual_note:
        note = f"{actual_note} · {note}"
    return [
        _chart_row(PLAIN, 1, width, draft_note),
        _chart_row(ALWAYS_SC, samples, width),
        _chart_row(SIFTSC, actual_passes, width, note),
    ]


def session_line(stats: SessionStats) -> str:
    """One-line running total shown after every answer."""

    return (
        f"📊 session · {_questions(stats.questions)} · "
        f"{stats.passes} vs {stats.always_sc_passes} passes · "
        f"{saving_words(stats.passes, stats.always_sc_passes)} vs always-SC"
    )


def _session_row(icon: str, label: str, value: int, maximum: int, width: int, note: str) -> str:
    cells = bar(value, maximum, width)
    line = f"{_INDENT}{icon} {label:<{_LABEL_WIDTH}} {cells}  {value:>4} {_plural(value):<6}"
    return f"{line} {note}".rstrip()


def session_chart(stats: SessionStats, *, width: int) -> list[str]:
    """Full session chart for the ``/stats`` command."""

    if stats.questions == 0:
        return ["📊 SESSION · no questions yet · ask something and Sifty will keep score"]
    votes = f"{stats.votes} {'vote' if stats.votes == 1 else 'votes'} called"
    bar_width = max(10, min(24, width - 56))
    maximum = max(stats.always_sc_passes, stats.passes)
    spent_note = (
        f"{_saving_icon(stats.passes, stats.always_sc_passes)} "
        f"{saving_words(stats.passes, stats.always_sc_passes)}"
    )
    return [
        f"📊 SESSION · {_questions(stats.questions)} · {votes}",
        _session_row("⚡", PLAIN, stats.plain_passes, maximum, bar_width, "if all used one pass"),
        _session_row("👥", ALWAYS_SC, stats.always_sc_passes, maximum, bar_width, "if all voted"),
        _session_row("🧭", "spent", stats.passes, maximum, bar_width, spent_note),
        f"{_INDENT}⏱️  {format_seconds(stats.seconds)} of generation · {stats.tokens:,} tokens",
    ]


def workload_chart(
    *,
    prompts: int,
    always_sc_passes: int,
    siftsc_passes: int,
    accuracy_retention: float,
    width: int,
) -> list[str]:
    """Chart the measured benchmark workload."""

    bar_width = max(10, min(32, width - 44))
    saved = always_sc_passes - siftsc_passes
    reduction = saved / always_sc_passes if always_sc_passes else 0.0
    always_cells = bar(always_sc_passes, always_sc_passes, bar_width)
    siftsc_cells = bar(siftsc_passes, always_sc_passes, bar_width)
    return [
        f"📉 MEASURED WORKLOAD · {prompts} PROMPTS",
        f"{_INDENT}👥 {ALWAYS_SC:<{_LABEL_WIDTH}} {always_cells}  {always_sc_passes:>5,} passes",
        f"{_INDENT}🧭 {SIFTSC:<{_LABEL_WIDTH}} {siftsc_cells}  {siftsc_passes:>5,} passes",
        f"{_INDENT}✅ saved {saved:,} passes · {reduction:.1%} less compute",
        f"{_INDENT}🎯 {accuracy_retention:.1%} of always-SC accuracy retained",
    ]


def compare_table(rows: list[CompareRow], *, samples: int) -> list[str]:
    """Table for compare mode: same question, three policies."""

    width = samples + 1
    lines = ["🔬 COMPARE · one question, three policies"]
    for row in rows:
        icon = POLICY_ICONS.get(row.policy, "•")
        line = (
            f"{_INDENT}{icon} {row.policy:<{_LABEL_WIDTH}} {bar(row.passes, width, width)}  "
            f"{row.passes} {_plural(row.passes):<6} {format_seconds(row.seconds):>6}  "
            f"→ {row.answer or '∅'}"
        )
        if row.note:
            line = f"{line}   {row.note}"
        lines.append(line)
    answers = {row.answer for row in rows}
    if len(answers) == 1:
        lines.append(f"{_INDENT}🤝 all three policies agree")
    else:
        lines.append(f"{_INDENT}⚠️  the policies disagree · voting is where they split")
    lines.append(
        f"{_INDENT}💡 plain is siftsc's own draft · always-SC reuses siftsc's voters when it voted"
    )
    return lines


def answer_lines(text: str, *, width: int) -> list[str]:
    """Indent and wrap a model answer under a ``🤖 Sifty`` header."""

    lines = ["🤖 Sifty"]
    for paragraph in text.strip().splitlines():
        stripped = paragraph.strip()
        if not stripped:
            continue
        lines.extend(
            textwrap.wrap(
                stripped,
                width=width,
                initial_indent=_INDENT,
                subsequent_indent=_INDENT,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines
