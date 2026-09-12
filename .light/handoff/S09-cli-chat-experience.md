---
session_no: S09
suggested_title: "[SiftSC] S10 private preview iteration"
parent_session: S08
project: siftsc
date: 2026-09-11
---

## Current stage

The terminal experience was reviewed in a real pseudo-terminal at 80 columns, including a forced fresh Hugging Face download. The first typed question after `siftsc demo` used to crash; it now answers. Compute cost is charted in the terminal for plain, always-SC, and SiftSC, per question and per session, and a compare mode runs all three policies on one question. Library noise is captured while loading. All local gates passed and the work is pushed to `main`.

## Repository and release state

- Project root: `~/Documents/ChatGPT/硅基线程/SiftSC`
- Remote: `https://github.com/heywanrong/SiftSC` (**PUBLIC** since 2026-09-11 at the maintainer's request; every push is visible)
- Default branch: `main`
- Package version is `0.3.0`. License is MIT since 2026-09-12 (was Apache-2.0). pip does **not** reinstall a Git direct-URL requirement whose version is unchanged, so every user-visible release needs a version bump; `siftsc --version` shows the installed one.
- Design note for this session: `.light/design/2026-09-11-cli-chat-experience.md`

## What changed in S09

- `src/siftsc/cli.py` — rewritten around a small `_Engine`: `plain`, `siftsc`, and `compare` turns; `🎯 Answer` line; compute chart after every answer; `/compare`, `/stats`, unknown-command hint; `readline` on TTYs; Ctrl-C during generation returns to the prompt, Ctrl-C while loading exits 130; `demo` gained `--raw-prompt` and continues into chat with a warning if the showcase drifts (non-interactive runs still exit 2).
- `src/siftsc/display.py` (new) — pure formatting: `compute_chart`, `session_line`, `session_chart`, `compare_table`, `workload_chart`, `answer_lines`, `SessionStats`, `TurnCost`, `CompareRow`. One chart cell = one generation pass. Only wide emoji are used in aligned columns.
- `src/siftsc/terminal.py` — spinner with live `update()` detail and `{elapsed}`; `QuietLibraryOutput` (file-descriptor capture with Python-level fallback); `bar`, `wrap_labeled`, `terminal_width`, `format_seconds`.
- `src/siftsc/backends.py` — `MeteredBackend` (seconds and tokens per pass), `MLXBackend.is_cached()` and `ensure_downloaded(progress)` using mlx-lm's own file patterns.
- `src/siftsc/router.py` — `SiftSC.sample_traces()`; escalation behaviour unchanged (same seeds).
- `src/siftsc/engine.py` — the turn engine (routing, generation, `TurnReport` with headline/notes/chart/side rows, `parse_input`, `status_message`) shared by `cli.py` and the new `tui.py`.
- `src/siftsc/tui.py` — Textual full-screen app (`siftsc ui`, also bare `siftsc`): RichLog conversation, side panel with compact cost rows, worker-thread generation, Ctrl+T/G/L/Q. Headless tests in `tests/test_tui.py` use `App.run_test`.
- `install.sh` + README: uv-based one-line install for the public release; `uv tool install --python 3.12` fetches Python itself. `.github/workflows/release.yml` publishes tags to PyPI once trusted publishing is configured.
- README: condensed to roughly a third; opens with the two hard limits (Apple Silicon only, checkable answers only), keeps the test-locked strings, and ends with a Scope section. Transcripts match the 0.2.1 output.
- `src/siftsc/prompts.py` — `looks_like_math()` routes questions: digits, number words, quantity phrases, math signs, or Chinese quantity words take the paper prompt; everything else takes the model's chat template with a short Sifty system prompt (`MLXBackend.chat_prompt`, `render_chat_prompt`). One-off `/math` and `/talk` prefixes and `--question-type` override it. General questions are answered in one pass and never voted, in every mode.

## Verified behaviour (real model, Qwen2.5-0.5B MLX 4-bit, mlx-lm 0.31.3)

- Fresh cache: `📦 One-time download … ✅ Model downloaded · 11.3s`; the native "unauthenticated requests" warning no longer garbles the spinner.
- Demo: plain 15 → SiftSC 25 with votes 20 · 15 · 25 · 25 · 30; plain 109 kept while the forced vote returns 99 (stable across four runs).
- Chat: Henry question voted, `🎯 Answer: 25`, six passes; compare mode reused the voters when a vote was called and drew five fresh voters otherwise; `/stats` fits in 80 columns.
- General questions: `中国的首都在哪` → `中国的首都是北京。`, `Who are you?` → `I am a small local assistant.`, `什么是机器学习？` answered in Chinese; `小明有三个苹果，吃了一个，还剩几个？` still takes the math route.
- Gates: ruff check, ruff format, strict mypy, 62 tests (1 opt-in local-model test skipped), `python -m build`, wheel install into a fresh venv, `siftsc demo --no-chat` from the wheel.

## Commands to reproduce

```bash
cd ~/Documents/ChatGPT/硅基线程/SiftSC
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
PYTHONPATH=src .venv/bin/mypy src/siftsc
PYTHONPATH=src .venv/bin/python -m siftsc.cli demo            # interactive
PYTHONPATH=src .venv/bin/python -m siftsc.cli chat --mode compare
SIFTSC_VERBOSE=1 PYTHONPATH=src .venv/bin/python -m siftsc.cli ask "What is 7 + 5?" --mode compare
```

## Design invariants (unchanged from S08, still true)

1. Generation is never stopped at the first `Answer:`; only displayed text is cleaned.
2. The demo threshold `0.84` applies only to the bundled model and profile in chat and demo; `ask` uses the profile threshold.
3. An escalated request costs six actual passes and is reported as such; compare mode spends `1 + samples` passes and records that honestly in the session total.
4. Apple Silicon only for the built-in MLX runner.
5. No paper files, prompts, weights, or secrets in the repository.

## Next steps (maximum three)

1. Try the animated flow in a real macOS Terminal window (not only the pseudo-terminal) and check emoji column alignment in the user's font; adjust `POLICY_ICONS` if a glyph renders narrow.
2. Watch for questions the router misclassifies (`looks_like_math`); extend the phrase lists rather than loosening the digit rule, and keep the math prompt byte-for-byte.
3. Create the PyPI project and trusted publisher (environment `pypi`), then tag `v0.3.0`; afterwards the install spec can become `siftsc[mlx]`. The `curl | sh` line was verified against the public repository from an isolated uv sandbox on 2026-09-11. An npm wrapper was considered and rejected: it could only shell out to this Python stack, and a JavaScript port would break the mlx-lm reproducibility the demo depends on.

## Do not

- Do not treat this card as current fact; refresh `git status`, `git log -3`, and the latest CI run first.
- Do not test with the global `siftsc` from miniconda; use `PYTHONPATH=src .venv/bin/python`.
- CI runs mypy on Linux with only the `dev` extra, so any lazy import of an optional library (`huggingface_hub`, `mlx`, `mlx_lm`) needs an entry in `[[tool.mypy.overrides]]`; reproduce with a venv that has only `.[dev]` installed before pushing.
- Do not delete the Hugging Face cache; the demo reuses the cached 290 MB checkpoint.
- Do not change benchmark claims or mutate the ICONIP workspace. The repository is public: scan for secrets and personal paths before every push, and keep `.light/` free of anything private.
- At the end of the next session, create an `S10` handoff card and print the next copyable startup prompt.
