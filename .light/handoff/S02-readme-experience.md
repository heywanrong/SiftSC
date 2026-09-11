---
session_no: S02
suggested_title: "[SiftSC] S03 redesigned preview review"
parent_session: S01
project: siftsc
date: 2026-09-11
---

## Current stage

The private preview now has a product-first README and a tested command-line experience.

## Completed

- `siftsc demo` automatically downloads a public 290 MB Qwen MLX model and reproduces a fixed plain-wrong/SiftSC-right comparison.
- `siftsc chat` provides an interactive reasoning loop with `/plain` and `/siftsc` live mode switching.
- The router now matches the paper protocol: one routing draft, then five fresh sampled voters on escalation.
- The README has one hero visual, one command, one reproducible example, and only two headline measurements.
- Unit tests, strict types, lint, package builds, wheel install, local-model inference, public-model inference, and interactive terminal behavior passed.

## Evidence

- Public model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, Hugging Face revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Runtime: Python 3.13.12, MLX 0.32.2, mlx-lm 0.31.3.
- Fixed demo: plain `15`; voter answers `20, 15, 25, 25, 30`; SiftSC `25`.

## Review notes

- The demo threshold is intentionally more permissive than the published pooled profile and is disclosed as such.
- The public community conversion enables immediate use but is not claimed to be byte-identical to the paper checkpoint.
- Before public release, confirm co-author credit, institutional IP expectations, and final proceedings metadata.
