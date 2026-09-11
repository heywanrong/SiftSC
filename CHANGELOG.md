# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

### Added

- One-command Hugging Face model download and a verified `siftsc demo` comparison.
- Interactive `siftsc chat` with live switching between plain and selective inference.
- Per-request and cumulative generation-pass savings relative to Always-SC@5.
- A second verified demo showing SiftSC preserving an answer that blind voting changes.
- A product-first README focused on one compute result and one quality result.

### Changed

- Escalated requests now use five fresh sampled voters, exactly matching the paper's SC@5 protocol.
- MLX CLI commands default to the public 290 MB Qwen2.5-0.5B 4-bit checkpoint.

## [0.1.0] - 2026-09-11

### Added

- Selective self-consistency API with auditable routing results.
- Local MLX backend with greedy-statistic capture and answer-aware stopping.
- Confidence and 12-feature logistic gates from the ICONIP study.
- Four bundled profiles derived from 1,600 real prompt-level records.
- CLI, custom-backend protocol, tests, CI, benchmark tables, and release plots.
