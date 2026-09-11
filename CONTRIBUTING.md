# Contributing

Thanks for helping make selective inference more reliable.

## Development setup

```bash
git clone https://github.com/heywanrong/SiftSC.git
cd SiftSC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src/siftsc
pytest --cov=siftsc --cov-report=term-missing
python -m build
```

Changes to a gate, feature definition, threshold, benchmark number, or profile must include provenance and a regression test. Do not commit model weights, raw benchmark examples, API keys, personal data, or generated completions without an explicit redistribution review.

Use focused commits and explain user-visible behavior in `CHANGELOG.md`.
