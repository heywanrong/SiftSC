from __future__ import annotations

from siftsc.display import (
    CompareRow,
    SessionStats,
    TurnCost,
    answer_lines,
    compare_table,
    compute_chart,
    session_chart,
    session_line,
    workload_chart,
)


def test_compute_chart_uses_one_cell_per_pass() -> None:
    lines = compute_chart(actual_passes=1, samples=5, seconds=0.4)

    assert len(lines) == 3
    assert "⚡ plain" in lines[0] and "█░░░░░" in lines[0] and lines[0].endswith("1 pass")
    assert "👥 always-SC" in lines[1] and "█████░" in lines[1] and "5 passes" in lines[1]
    assert "🧭 siftsc" in lines[2] and "█░░░░░" in lines[2] and "saved 4 (80.0%)" in lines[2]
    assert "0.4s" in lines[2]


def test_compute_chart_reports_extra_cost_when_voting() -> None:
    lines = compute_chart(actual_passes=6, samples=5, draft_note="the draft said 15")

    assert "██████" in lines[2]
    assert "extra 1 (+20.0%)" in lines[2]
    assert "the draft said 15" in lines[0]


def test_session_stats_track_passes_and_savings() -> None:
    stats = SessionStats(samples=5)
    stats.record(TurnCost(passes=1, seconds=0.5, tokens=20), voted=False)
    stats.record(TurnCost(passes=6, seconds=3.0, tokens=200), voted=True)

    assert stats.questions == 2
    assert stats.passes == 7
    assert stats.always_sc_passes == 10
    assert stats.saved == 3
    assert stats.votes == 1
    assert "2 questions" in session_line(stats)
    assert "7 vs 10 passes" in session_line(stats)
    assert "saved 3 (30.0%)" in session_line(stats)


def test_session_chart_scales_bars_to_always_sc() -> None:
    stats = SessionStats(samples=5)
    stats.record(TurnCost(passes=1, seconds=0.5, tokens=20), voted=False)
    stats.record(TurnCost(passes=6, seconds=3.0, tokens=200), voted=True)
    lines = session_chart(stats, width=80)

    assert lines[0].startswith("📊 SESSION")
    assert "1 vote called" in lines[0]
    always = next(line for line in lines if "always-SC" in line)
    spent = next(line for line in lines if "spent" in line)
    assert always.count("█") > spent.count("█")
    assert "10 passes" in always
    assert "7 passes" in spent
    assert any("3.5s" in line and "220 tokens" in line for line in lines)
    assert all(len(line) <= 80 for line in lines)


def test_session_chart_handles_empty_session() -> None:
    lines = session_chart(SessionStats(samples=5), width=80)

    assert any("no questions yet" in line for line in lines)


def test_compare_table_lists_three_policies() -> None:
    rows = [
        CompareRow("plain", "15", 1, 0.5, ""),
        CompareRow("always-SC", "25", 5, 2.4, "votes 20 · 15 · 25 · 25 · 30"),
        CompareRow("siftsc", "25", 6, 2.9, "🗳️ vote called"),
    ]
    lines = compare_table(rows, samples=5)

    assert lines[0].startswith("🔬 COMPARE")
    assert "⚡ plain" in lines[1] and "→ 15" in lines[1] and "1 pass" in lines[1]
    assert "👥 always-SC" in lines[2] and "5 passes" in lines[2] and "2.4s" in lines[2]
    assert "🧭 siftsc" in lines[3] and "6 passes" in lines[3] and "vote called" in lines[3]
    assert any("disagree" in line for line in lines)


def test_compare_table_notes_agreement() -> None:
    rows = [
        CompareRow("plain", "12", 1, 0.5, ""),
        CompareRow("always-SC", "12", 5, 2.4, ""),
        CompareRow("siftsc", "12", 1, 0.5, "🛡️ vote skipped"),
    ]
    lines = compare_table(rows, samples=5)

    assert any("agree" in line and "disagree" not in line for line in lines)


def test_workload_chart_shows_measured_savings() -> None:
    lines = workload_chart(
        prompts=400,
        always_sc_passes=2000,
        siftsc_passes=415,
        accuracy_retention=0.987,
        width=80,
    )

    assert lines[0].startswith("📉 MEASURED WORKLOAD · 400 PROMPTS")
    assert "2,000 passes" in lines[1]
    assert "415 passes" in lines[2]
    assert lines[1].count("█") > lines[2].count("█")
    assert "1,585 passes" in lines[3]
    assert "79.2% less compute" in lines[3]
    assert "98.7% of always-SC accuracy retained" in lines[4]
    assert all(len(line) <= 80 for line in lines)


def test_answer_lines_indent_and_wrap() -> None:
    lines = answer_lines("First line.\n\nThe answer is 12.", width=40)

    assert lines[0] == "🤖 Sifty"
    assert lines[1] == "   First line."
    assert lines[-1] == "   The answer is 12."
    assert all(len(line) <= 40 for line in lines)
