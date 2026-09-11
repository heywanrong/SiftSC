"""Answer questions under a policy and measure what they cost.

The engine is shared by the line-mode chat, the full-screen UI, and ``siftsc
ask``. It never prints: every turn returns a :class:`TurnReport` whose lines the
front ends render, so the two interfaces cannot drift apart.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .backends import Backend, MeteredBackend
from .display import (
    ALWAYS_SC,
    COMPARE,
    PLAIN,
    SIFTSC,
    CompareRow,
    SessionStats,
    TurnCost,
    compact_compare_rows,
    compact_cost_rows,
    compare_table,
    compute_chart,
    saving_icon,
    saving_words,
)
from .gates import load_profile
from .prompts import fallback_chat_prompt, looks_like_math, math_prompt
from .router import SiftSC
from .terminal import format_seconds
from .types import Generation, SiftResult
from .voting import parse_answer, plurality_vote

MODES = (PLAIN, SIFTSC, COMPARE)
QUESTION_TYPES = ("auto", "math", "general")
GENERAL_NOTE = "   Votes are for reasoning questions with a checkable answer · try /math <q>"
COMPARE_SKIPPED_NOTE = "   🔬 Compare skipped · a free-text answer cannot be counted as votes"


def make_runner(args: argparse.Namespace, backend: Backend) -> SiftSC:
    gate = load_profile(args.profile)
    if args.threshold is not None:
        gate = gate.with_threshold(args.threshold)
    return SiftSC(
        backend,
        gate,
        samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


@dataclass(frozen=True, slots=True)
class UserInput:
    """What the user typed: a command, a question, or nothing."""

    command: str | None = None
    argument: str | None = None
    question: str | None = None
    kind: str | None = None


def parse_input(text: str) -> UserInput:
    """Interpret one line of chat input; shared by every front end."""

    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return UserInput()
    if lowered in {"/exit", "/quit", "exit", "quit"}:
        return UserInput(command="exit")
    if lowered in {f"/{mode}" for mode in MODES}:
        return UserInput(command="mode", argument=lowered[1:])
    if lowered in {"/stats", "/clear", "/help"}:
        return UserInput(command=lowered[1:])
    for prefix, kind in (("/math ", "math"), ("/talk ", "general")):
        if lowered.startswith(prefix):
            question = stripped[len(prefix) :].strip()
            return UserInput(question=question, kind=kind) if question else UserInput()
    if stripped.startswith("/"):
        return UserInput(command="unknown", argument=stripped)
    return UserInput(question=stripped)


def status_message(mode: str, kind: str) -> str:
    """Spinner text while a question is being answered."""

    if kind == "general":
        return "💬 Sifty is answering"
    if mode == PLAIN:
        return "💭 Sifty is thinking once"
    if mode == COMPARE:
        return "🔬 Sifty is running plain, always-SC and siftsc"
    return "🧠 Sifty is deciding whether to call a vote"


def success_message(mode: str, kind: str) -> str:
    if kind != "general" and mode == COMPARE:
        return "✨ Comparison ready · {elapsed}"
    return "✨ Answer ready · {elapsed}"


@dataclass(frozen=True, slots=True)
class TurnReport:
    """Everything a front end needs to show one answered question."""

    mode: str
    kind: str
    question: str
    text: str
    headline: str
    notes: tuple[str, ...]
    chart: tuple[str, ...]
    side: tuple[str, ...]
    summary: str
    cost: TurnCost
    voted: bool


@dataclass(slots=True)
class Comparison:
    result: SiftResult
    rows: list[CompareRow]
    cost: TurnCost


@dataclass(slots=True)
class Engine:
    """Run one question under a policy and measure what it cost."""

    runner: SiftSC
    metered: MeteredBackend
    samples: int
    max_tokens: int
    temperature: float
    top_p: float
    raw_prompt: bool
    question_type: str
    backend: Backend
    stats: SessionStats

    @classmethod
    def create(cls, args: argparse.Namespace, backend: Backend) -> Engine:
        metered = MeteredBackend(backend)
        return cls(
            runner=make_runner(args, metered),
            metered=metered,
            samples=args.samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            raw_prompt=bool(getattr(args, "raw_prompt", False)),
            question_type=str(getattr(args, "question_type", "auto")),
            backend=backend,
            stats=SessionStats(samples=args.samples),
        )

    # ----------------------------------------------------------------- routing

    def kind_of(self, question: str, forced: str | None = None) -> str:
        """Classify a question as ``math`` (paper route) or ``general`` (chat route)."""

        if self.raw_prompt:
            return "math"
        chosen = forced or self.question_type
        if chosen in {"math", "general"}:
            return chosen
        return "math" if looks_like_math(question) else "general"

    def prompt_for(self, question: str) -> str:
        return question if self.raw_prompt else math_prompt(question)

    def chat_prompt_for(self, question: str) -> str:
        render = getattr(self.backend, "chat_prompt", None)
        if render is None:
            return fallback_chat_prompt(question)
        return str(render(question))

    # ------------------------------------------------------------- generation

    def _take_cost(self) -> TurnCost:
        costs = self.metered.take()
        return TurnCost(
            passes=len(costs),
            seconds=sum(cost.seconds for cost in costs),
            tokens=sum(cost.tokens for cost in costs),
        )

    def discard_partial_cost(self) -> None:
        """Forget passes recorded by an interrupted turn."""

        self.metered.take()

    def _greedy(self, prompt: str) -> Generation:
        return self.metered.generate(
            prompt,
            greedy=True,
            seed=0,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def plain(self, question: str) -> tuple[Generation, TurnCost]:
        generation = self._greedy(self.prompt_for(question))
        return generation, self._take_cost()

    def general(self, question: str) -> tuple[Generation, TurnCost]:
        """Answer a general question once, in chat style, without voting."""

        generation = self._greedy(self.chat_prompt_for(question))
        return generation, self._take_cost()

    def siftsc(self, question: str) -> tuple[SiftResult, TurnCost]:
        result = self.runner(self.prompt_for(question), feature_text=question)
        return result, self._take_cost()

    def compare(self, question: str) -> Comparison:
        """Answer with SiftSC, then complete the always-SC baseline without waste.

        Plain inference is exactly SiftSC's greedy draft, so it is never rerun. When
        SiftSC called a vote, always-SC is the same five voters; otherwise five fresh
        voters are drawn with the router's own seeds.
        """

        prompt = self.prompt_for(question)
        result = self.runner(prompt, feature_text=question)
        costs = self.metered.take()
        draft_cost, voter_costs = costs[0], costs[1:]
        if result.used_self_consistency:
            voters = result.generations[1:]
            always_answer = result.parsed_answer
        else:
            voters = self.runner.sample_traces(prompt)
            voter_costs = self.metered.take()
            _, always_answer, _ = plurality_vote(voters, self.runner.parser)
        voter_seconds = sum(cost.seconds for cost in voter_costs)
        votes = " · ".join(self.runner.parser(voter.text) or "∅" for voter in voters)
        siftsc_seconds = draft_cost.seconds + (
            voter_seconds if result.used_self_consistency else 0.0
        )
        siftsc_note = "🗳️ vote called" if result.used_self_consistency else "🛡️ vote skipped"
        rows = [
            CompareRow(
                PLAIN, self.runner.parser(result.generations[0].text), 1, draft_cost.seconds
            ),
            CompareRow(ALWAYS_SC, always_answer, len(voters), voter_seconds, f"votes {votes}"),
            CompareRow(
                SIFTSC, result.parsed_answer, result.generation_passes, siftsc_seconds, siftsc_note
            ),
        ]
        cost = TurnCost(
            passes=1 + len(voters),
            seconds=draft_cost.seconds + voter_seconds,
            tokens=draft_cost.tokens + sum(cost.tokens for cost in voter_costs),
        )
        return Comparison(result=result, rows=rows, cost=cost)

    # ----------------------------------------------------------------- reports

    def answer(self, mode: str, question: str, kind: str | None = None) -> TurnReport:
        """Answer one question, record its cost, and return what to display."""

        resolved = self.kind_of(question, kind)
        if resolved == "general":
            generation, cost = self.general(question)
            report = self._general_report(mode, question, generation, cost)
        elif mode == PLAIN:
            generation, cost = self.plain(question)
            report = self._plain_report(question, generation, cost)
        elif mode == COMPARE:
            comparison = self.compare(question)
            report = self._compare_report(question, comparison)
        else:
            result, cost = self.siftsc(question)
            report = self._siftsc_report(question, result, cost)
        self.stats.record(report.cost, voted=report.voted)
        return report

    def _general_report(
        self, mode: str, question: str, generation: Generation, cost: TurnCost
    ) -> TurnReport:
        notes = [GENERAL_NOTE]
        if mode == COMPARE:
            notes.append(COMPARE_SKIPPED_NOTE)
        return TurnReport(
            mode=mode,
            kind="general",
            question=question,
            text=generation.text,
            headline=(
                "💬 General question · answered once in chat style · "
                f"{format_seconds(cost.seconds)}"
            ),
            notes=tuple(notes),
            chart=(),
            side=("💬 general question", "one pass · no vote"),
            summary="",
            cost=cost,
            voted=False,
        )

    def _plain_report(self, question: str, generation: Generation, cost: TurnCost) -> TurnReport:
        answer = parse_answer(generation.text)
        return TurnReport(
            mode=PLAIN,
            kind="math",
            question=question,
            text=generation.text,
            headline=f"🎯 Answer: {answer or '∅'} · ⚡ one pass · {format_seconds(cost.seconds)}",
            notes=(),
            chart=tuple(compute_chart(actual_passes=1, samples=self.samples, mode=PLAIN)),
            side=tuple(compact_cost_rows(1, self.samples, mode=PLAIN)),
            summary=(
                f"⚡ one pass · {saving_words(1, self.samples)} vs always-SC · "
                "siftsc not run (/siftsc lets the router decide)"
            ),
            cost=cost,
            voted=False,
        )

    def _siftsc_report(self, question: str, result: SiftResult, cost: TurnCost) -> TurnReport:
        draft_answer = parse_answer(result.generations[0].text)
        if result.used_self_consistency:
            votes = " · ".join(parse_answer(item.text) or "∅" for item in result.generations[1:])
            route = f"🗳️ vote called · votes {votes}"
            draft_note = f"(the draft said {draft_answer or '∅'})"
            if draft_answer == result.parsed_answer:
                actual_note = "🤝 vote confirmed the draft"
            else:
                actual_note = "🛠️ vote changed the draft"
        else:
            route, draft_note, actual_note = "🛡️ first answer accepted", "", ""
        passes = result.generation_passes
        unit = "pass" if passes == 1 else "passes"
        saving = f"{saving_icon(passes, self.samples)} {saving_words(passes, self.samples)}"
        if result.used_self_consistency and draft_answer != result.parsed_answer:
            summary = (
                f"🛠️ vote changed the draft ({draft_answer or '∅'} → "
                f"{result.parsed_answer or '∅'}) · {saving}"
            )
        elif result.used_self_consistency:
            summary = f"🤝 vote confirmed the draft · {saving}"
        else:
            summary = f"🛡️ vote skipped · {saving}"
        return TurnReport(
            mode=SIFTSC,
            kind="math",
            question=question,
            text=result.text,
            headline=(
                f"🎯 Answer: {result.parsed_answer or '∅'} · {route} · {passes} {unit} · "
                f"{format_seconds(cost.seconds)}"
            ),
            notes=(),
            chart=tuple(
                compute_chart(
                    actual_passes=passes,
                    samples=self.samples,
                    draft_note=draft_note,
                    actual_note=actual_note,
                )
            ),
            side=tuple(compact_cost_rows(passes, self.samples)),
            summary=summary,
            cost=cost,
            voted=result.used_self_consistency,
        )

    def _compare_report(self, question: str, comparison: Comparison) -> TurnReport:
        result = comparison.result
        passes = result.generation_passes
        unit = "pass" if passes == 1 else "passes"
        route = "🗳️ vote called" if result.used_self_consistency else "🛡️ first answer accepted"
        answers = {row.answer for row in comparison.rows}
        summary = (
            "🤝 all three policies agree"
            if len(answers) == 1
            else "⚠️ the policies disagree · voting is where they split"
        )
        return TurnReport(
            mode=COMPARE,
            kind="math",
            question=question,
            text=result.text,
            headline=(
                f"🎯 Answer: {result.parsed_answer or '∅'} · {route} · {passes} {unit} · "
                f"{format_seconds(comparison.cost.seconds)} for all three policies"
            ),
            notes=(),
            chart=tuple(compare_table(comparison.rows, samples=self.samples)),
            side=tuple(compact_compare_rows(comparison.rows, samples=self.samples)),
            summary=summary,
            cost=comparison.cost,
            voted=result.used_self_consistency,
        )
