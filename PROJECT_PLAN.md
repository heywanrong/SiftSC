# SiftSC release plan

## v0.1.0 private preview

- [x] Extract the paper method into a standalone package.
- [x] Add a stable Python API, CLI, MLX backend, backend protocol, gates, and voting.
- [x] Export four provenance-bearing profiles from the paper records.
- [x] Generate real benchmark CSVs and figures.
- [x] Add unit and local-model integration tests.
- [ ] Complete release audit, Git commit, private GitHub push, and remote CI check.

## Later, after author review

- [ ] Add a documented calibration CLI with a held-out threshold split.
- [ ] Validate more generation seeds, tasks, model families, and sample budgets.
- [ ] Add Hugging Face and OpenAI-compatible backend examples where logit access permits.
- [ ] Publish a tagged release and package only after the repository becomes public.
