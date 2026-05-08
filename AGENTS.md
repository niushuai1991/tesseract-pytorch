# AGENTS.md

## Commands

```bash
# Run all tests (uv sync is currently broken; use venv directly)
.venv/bin/python -m pytest tests/ -v

# Run tests excluding the real-eng suite (no /tmp/eng.traineddata needed)
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_real_eng.py

# Run a single test file
.venv/bin/python -m pytest tests/test_layers.py -v

# CLI entry point
tesseract-pytorch train --traineddata eng.traineddata --train-list train.txt
```

## Known issues

- **`uv sync` fails** with the current `pyproject.toml`: the `[tool.uv.sources]` torch entries lack required platform markers. The existing `.venv` works; do not attempt `uv sync`.

## Test prerequisites

- `tests/test_real_eng.py` requires `/tmp/eng.traineddata` to exist (11 tests, all skip/fail without it). A copy lives at the repo root as `eng.traineddata`.

## Architecture

- Python package is `tesseract_cuda` (under `src/`), not `tesseract_pytorch`.
- No linter, formatter, or typechecker is configured. Only `pytest` is in dev dependencies.
- Network spec language (e.g. `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`) is parsed by `src/tesseract_cuda/network/spec_parser.py`.
- Weight conversion between Tesseract binary and PyTorch formats lives in `src/tesseract_cuda/network/weight_mapper.py`.
