# S07 — Interactive terminal experience

## Outcome

- `siftsc demo` now keeps the loaded model alive and opens an interactive question prompt on a terminal.
- Added compact loading and reasoning spinners, emoji route feedback, `/help`, `/clear`, live mode switching, and a friendly exit message.
- Preserved clean non-interactive output for pipes and CI; `--no-chat` explicitly exits after the fixed showcase.
- Removed repeated answers and model-turn markers from displayed completions without changing the token statistics used by the router.
- Default public-model chat now uses the same documented threshold as the reproducible demo.

## Validation

- All unit tests passed, including TTY continuation and non-TTY animation behavior.
- The cached Qwen2.5-0.5B MLX 4-bit model reproduced both showcase cases.
- A piped interactive Henry question triggered voting, returned `25`, reported six actual passes, and exited only after `/exit`.
