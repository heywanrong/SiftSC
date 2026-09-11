# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

### Added

- One-command Hugging Face model download and a verified `siftsc demo` comparison.
- Interactive `siftsc chat` with live switching between plain and selective inference.
- Per-request and cumulative generation-pass savings relative to Always-SC@5.
- A second verified demo showing SiftSC preserving an answer that blind voting changes.
- A product-first README focused on one compute result and one quality result.
- A compact, friendly cartoon mascot for the project identity.
- Platform and onboarding badges that expose the Apple-Silicon requirement before installation.
- Private-preview connectivity, authentication, and local-clone installation guidance.
- A continuous `siftsc demo` experience that flows from verified cases into the user's own interactive questions without reloading the model.
- Lightweight terminal spinners, emoji route feedback, help and clear commands, and automatic non-TTY fallback.
- Terminal compute charts after every answer: one cell per generation pass for plain, always-SC, and SiftSC, plus a session chart via `/stats`.
- Compare mode (`/compare`, `--mode compare`) that answers with SiftSC and completes the always-SC baseline from the same draft and voters, reporting each policy's answer, passes, and measured seconds.
- A highlighted `🎯 Answer` line, wrapped answer blocks, and per-turn wall-clock and token accounting via `MeteredBackend`.
- First-launch download progress (downloaded MB, total MB, elapsed time) and a cache check so the download notice appears only when files are missing.
- `SiftSC.sample_traces()` for drawing the paper's fresh SC@N voters directly.

### Changed

- Escalated requests now use five fresh sampled voters, exactly matching the paper's SC@5 protocol.
- MLX CLI commands default to the public 290 MB Qwen2.5-0.5B 4-bit checkpoint.
- README navigation and section hierarchy now use restrained emoji cues for faster scanning.
- Replaced the funnel logo with the friendly Sifty mascot and moved it beside the wordmark.
- Reworked the opening, quick start, and method explanation into compact visual cards.
- The primary Git install command now upgrades an existing preview installation in place.
- Public-model chat now shares the reproducible demo routing threshold and uses shorter, cleaned completions for a focused terminal experience.
- Everything the model libraries print while loading is captured at the file-descriptor level and shown only on failure or with `SIFTSC_VERBOSE=1`, so status lines no longer garble.
- The README install command uses `pip -q`; demo questions wrap to the terminal width; the measured workload is drawn as a chart.
- A demo that does not reproduce now warns and still opens chat on a terminal; non-interactive runs keep exit code 2.

### Fixed

- `siftsc demo` crashed with `AttributeError: raw_prompt` on the first question typed after the showcase.
- Arrow keys printed escape codes at the chat prompt; line editing and history are now enabled on terminals.
- Ctrl-C during generation stops the current answer and returns to the prompt; Ctrl-C during loading exits cleanly with status 130.

## [0.1.0] - 2026-09-11

### Added

- Selective self-consistency API with auditable routing results.
- Local MLX backend with greedy-statistic capture and answer-aware stopping.
- Confidence and 12-feature logistic gates from the ICONIP study.
- Four bundled profiles derived from 1,600 real prompt-level records.
- CLI, custom-backend protocol, tests, CI, benchmark tables, and release plots.
