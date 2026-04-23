## Install dev venv
`uv sync --extra dev`

## Running Tests

This repository uses `unittest` for tests and `tox` for venv-sandboxed testing.

### Tox
- Run default unittests via tox in isolated Python venvs:
  - `uv run tox`
- Run all, including those redownloading weights even if already present:
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run tox`
- Append `-r` to recreate env from scratch, and/or `-vv` for verbose output.

### Unit tests
- Run all default tests:
  - `uv run python -m unittest discover -s test`
- Run all tests, including those taking time to re-download weights:
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run python -m unittest discover -s test`


### Other unit tests
- Run a specific test module:
  - `uv run python -m unittest test.test_weight_cache`
- Run multiple modules:
  - `uv run python -m unittest test.test_weight_cache test.test_registration`
- Run only the weights download tests:
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run python -m unittest test.test_weight_cache_download_integration`

## Syntax check:
`uv run python -m compileall fundus_image_toolbox`