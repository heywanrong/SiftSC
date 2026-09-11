# CLI chat experience — design note (S09)

Date: 2026-09-11 · Scope: `siftsc demo`, `siftsc chat`, `siftsc ask` presentation and robustness.

## Problems found in the real terminal

1. `siftsc demo` crashed on the first typed question: the `demo` parser has no
   `--raw-prompt`, but the shared chat loop read `args.raw_prompt`.
2. A native-code Hugging Face warning printed in the middle of the loading
   spinner and garbled the line; the first-launch download showed no progress.
3. The "290 MB download" notice printed even when the model was already cached.
4. Answers ran together with the prompt; the final answer was not highlighted.
5. Compute cost was text only. The maintainer wants the cost of the different
   policies visualised in the terminal.
6. No line editing (arrow keys printed escape codes); Ctrl-C during generation
   crashed with a traceback.
7. `pip install` printed ~124 lines; the README command did not use `-q`.

## Design

### Modules

- `terminal.py` — presentation primitives only: spinner with live detail and
  elapsed time, `QuietLibraryOutput` (file-descriptor level capture with a
  Python-level fallback so tests and pipes still work), `bar`, `wrap_labeled`,
  `terminal_width`.
- `display.py` — pure formatting functions that return lines: answer block,
  per-turn compute chart, session summary, session chart, compare table,
  measured-workload chart, and `SessionStats`.
- `backends.py` — `MLXBackend.is_cached()` / `ensure_downloaded(progress)` using
  the same file patterns as `mlx_lm`, plus `MeteredBackend` that records seconds
  and generated tokens per pass without touching gate statistics.
- `router.py` — `SiftSC.sample_traces(prompt)` exposes the exact voter sampling
  (same seeds) so compare mode can reuse or draw voters honestly.
- `cli.py` — wiring only.

### Chat turn output (siftsc mode)

```
🎯 Answer: 25 · 🗳️ vote called · votes 20 · 15 · 25 · 25 · 30 · 6 passes · 3.1s
   ⚡ plain       █░░░░░   1 pass    (the draft said 15)
   👥 always-SC   █████░   5 passes
   🧭 siftsc      ██████   6 passes  🛠️ vote fixed the draft · extra 1 (+20%)
📊 session · 2 questions · 7 vs 10 passes · saved 3 (30.0%) vs always-SC
```

One bar cell is one generation pass, so the bar length is the cost. Emoji used
in aligned columns are all East-Asian-Width "wide" (two cells) so columns line up.

### Modes

- `/plain` one greedy pass; `/siftsc` selective voting (default);
  `/compare` runs plain, always-SC, and SiftSC on the same question. Plain and
  SiftSC share the same greedy draft; always-SC reuses SiftSC's five voters
  when a vote was called, otherwise five fresh voters with the router's seeds.
- `/stats` prints the session chart; `/help`, `/clear`, `/exit` unchanged.

### Loading

- Cache check first; the download spinner appears only when files are missing
  and shows downloaded MB, total MB, and elapsed seconds.
- Everything the model libraries print is captured at the file-descriptor level
  while loading and shown only on failure or with `SIFTSC_VERBOSE=1`.

### Invariants preserved

Generation is never stopped at the first `Answer:`; the demo threshold applies
only to the bundled model/profile; escalation is reported as six actual passes;
Apple-Silicon-only wording stays; no paper files are vendored.
