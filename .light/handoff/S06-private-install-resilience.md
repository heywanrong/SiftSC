# S06 — Private install resilience

## Diagnosis

The reported failure occurred in `git clone` before Python packaging or SiftSC ran. The terminal could not connect to `github.com:443`; the package was not the failure source.

## Reproduction

- GitHub connectivity later returned HTTP 200.
- `git ls-remote` resolved the private repository at commit `4fcdb66`.
- The exact README dependency specification installed successfully into a clean Python 3.13 virtual environment.
- The freshly installed CLI completed `siftsc demo` and reproduced both verified cases and workload metrics.

## UX hardening

- Added `--upgrade` to the primary install command.
- Documented the private-preview authentication requirement.
- Added checks for GitHub connectivity, GitHub CLI authentication, and Git repository access.
- Added a local-clone installation path that bypasses GitHub when the source is already present.
