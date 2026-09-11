---
session_no: S08
suggested_title: "[SiftSC] S09 private preview iteration"
parent_session: S07
project: siftsc
date: 2026-09-11
---

## Current stage

SiftSC is an independent, installable private-preview repository extracted from the ICONIP paper workspace. The first branded CLI release is implemented, tested, and pushed. The next maintainer should review the real Apple-Silicon terminal experience and continue product/README refinements without changing the paper source project.

## Repository and release state

- Project root: `~/Documents/ChatGPT/硅基线程/SiftSC`
- Remote: `https://github.com/heywanrong/SiftSC`
- Visibility: **PRIVATE**; do not make it public without explicit maintainer approval.
- Default branch: `main`
- Latest feature baseline: `82515d3873c086d2273671d43f3b6af1e2324177` (`feat: make demo an interactive terminal experience`)
- CI: GitHub Actions run `34607179786` passed on Python 3.11, 3.12, and 3.13.
- Paper workspace: `~/Desktop/Projects/ICONIP`; it is source context only and must not be mutated or vendored into SiftSC.

## Completed artifacts and verified behavior

- `README.md` — product-first presentation, mascot wordmark, Apple-Silicon warning, one-command private installation, two headline results, verified examples, interactive transcript, and troubleshooting.
- `src/siftsc/cli.py` — `siftsc demo` now runs the verified showcase and, on an interactive terminal, keeps the same model loaded and opens `💬 you · siftsc >` for user questions. `--no-chat`, `--json`, pipes, and CI still exit cleanly.
- `src/siftsc/terminal.py` — dependency-free Braille spinner, emoji status messages, and automatic non-TTY/CI fallback. `SIFTSC_NO_ANIMATION=1` disables animation explicitly.
- `src/siftsc/backends.py` — strips repeated answers and model-turn markers from displayed completions while preserving the generated token statistics used by the router.
- `tests/test_cli.py`, `tests/test_backends.py`, `tests/test_terminal.py` — cover demo-to-chat continuation, default public-model threshold, completion cleanup, and non-TTY presentation.
- Local Qwen2.5-0.5B MLX 4-bit validation reproduced both showcase cases. A user-entered Henry question triggered voting, returned `25`, reported `passes=6`, and did not emit repeated `<|endoftext|>`/`Human:` text.
- Local quality gates passed: Ruff lint, Ruff format, strict mypy, 34 tests (1 opt-in local-model test skipped), package build, wheel installation, and CLI smoke test.

## Commands to reproduce

User upgrade and launch:

```bash
python -m pip install --upgrade --force-reinstall \
  "siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git"
siftsc demo
```

Expected terminal transition after the showcase:

```text
🚀 YOUR TURN · The model stays loaded—ask your own question!
💬 you · siftsc >
```

Source validation (use the project environment and source path so an older globally installed package is not imported):

```bash
cd ~/Documents/ChatGPT/硅基线程/SiftSC
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
PYTHONPATH=src .venv/bin/mypy src/siftsc
PYTHONPATH=src .venv/bin/python -m build
PYTHONPATH=src .venv/bin/python -m siftsc.cli demo --no-chat
```

## Design invariants

1. Do not stop generation as soon as the first `Answer:` appears. That changes token-level gate features and previously caused the Henry case to stop escalating. Generate normally, preserve the statistics, then clean only the displayed completion.
2. For the bundled public model/profile, interactive chat intentionally uses the documented demo threshold `0.84` so the public conversion reproduces the correction. Custom models/profiles must not silently inherit it.
3. An escalated request costs six actual passes: one routing draft plus five fresh SC voters. The CLI must continue reporting this honestly against Always-SC@5, even when that individual request says `extra=1`; the workload-level result remains 79.2% fewer passes with 98.7% of Always-SC accuracy retained.
4. The built-in MLX demo/chat currently supports Apple Silicon only. The NumPy router can be used cross-platform with a custom backend, but do not imply that a built-in Linux/Windows runner exists.
5. Keep the tool repository independent from the ICONIP paper files, keep benchmark claims traceable, and do not expose raw paper prompts, private artifacts, secrets, or model weights.

## Next steps (maximum three)

1. Install the current GitHub version in a fresh Apple terminal and review the full animated `siftsc demo` → user-question flow at the actual terminal width; collect concrete UX changes before editing.
2. Implement the maintainer's requested CLI/README refinements while preserving the invariants above and keeping the repository private.
3. Re-run unit, lint, type, package, wheel-install, and real local-model checks; push to `main` only after the Henry answer remains `25`, then wait for the full GitHub CI matrix.

## Risks and known limitations

- Private Git installation requires GitHub connectivity and collaborator authentication; a port-443 failure occurs before pip can build SiftSC.
- Routing thresholds are distribution-specific. Recalibrate when changing the model, quantization, task, prompt template, or decoding settings.
- SiftSC is not a correctness verifier and must not be presented as one, especially for high-stakes questions.
- The fixed demo and public-model chat use a slightly more permissive threshold to demonstrate the mechanism; they are not substitutes for the aggregate benchmark.
- The Python package version remains `0.1.0`; Git direct-URL upgrades identify the latest private-preview code by commit.

## Required reading order

1. This card.
2. `.light/passport.yaml`.
3. `README.md`.
4. `src/siftsc/cli.py` and `src/siftsc/terminal.py`.
5. `src/siftsc/backends.py`, `tests/test_cli.py`, `tests/test_backends.py`, and `tests/test_terminal.py`.
6. `docs/benchmarks/RESULTS.md` and `docs/ETHICS_REVIEW.md` before changing any performance claim.

## Do not

- Do not redo completed packaging or branding work without evidence that an artifact is stale.
- Do not treat this handoff as current fact; first refresh `git status`, `git log -3`, remote visibility, and the latest CI run.
- Do not test with an unqualified system `python -m pytest` if an older SiftSC is globally installed; use `PYTHONPATH=src` and `.venv/bin/python`.
- Do not delete the user's Hugging Face model cache; upgrades reuse the cached ~290 MB checkpoint.
- Do not publish the repository, change benchmark claims, or mutate the ICONIP workspace without explicit authorization.
- At the end of the next maintenance session, create an `S09` handoff card and print the next copyable startup prompt so the chain continues.
