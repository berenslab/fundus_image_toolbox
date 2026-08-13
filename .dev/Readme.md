## Auto-testing
As of August 2026, Dependabot opens monthly PRs against `dev`, considering dependencies of uv and github-actions. GitHub Actions tox tests run on PRs into `dev` and `main`.

## Before PRs into main
- On `dev` branch, install venv: `uv sync --extra dev --extra notebook`
- Run `uv lock --upgrade` to use latest compatible package versions.
- Bump FIT version: Edit the `fundus_image_toolbox/_version.py`
- Run tests on multiple OS: This repository uses `unittest` for tests and `tox` for venv-sandboxed testing. See below how to run.
- Update compatibility table in `README.md`.
- Merge into `main` with a merge commit (not squash or rebase), so `dev` and `main` stay related.

## Testing
### Tox
- Run default unittests via tox in isolated Python venvs:
  - `uv run tox`
- Run all, including those redownloading weights even if already present NOTE: is ignored since using tox-uv; but without it I couldnt use py311 or above.
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run tox`
- Append `-r` to recreate env from scratch, and/or `-vv` for verbose output.

### Unit tests
- Run all default tests:
  - `uv run python -m unittest discover -s test`
- Run all tests, including those taking time to re-download weights:
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run python -m unittest discover -s test`


### Other unit tests and tox calls
- Run a specific test module:
  - `uv run python -m unittest test.test_weight_cache`
- Run multiple modules:
  - `uv run python -m unittest test.test_weight_cache test.test_registration`
- Run only the weights download tests:
  - `FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 uv run python -m unittest test.test_weight_cache_download_integration`
- Run with tox a specific Python env only:
  - `tox r -e py312`
- Run tox with their default venv (not uv):
  - `tox --runner virtualenv r` (`-e py312`) # will use virtualenv+pip

## Syntax check:
`uv run python -m compileall fundus_image_toolbox`

## New optional dependencies?
If we add deps to extra groups, `.github/dependabot.yml` should be updated accordingly to handle them as what they are: optional extras.