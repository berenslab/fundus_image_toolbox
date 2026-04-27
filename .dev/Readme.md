## Install dev venv
`uv sync --extra dev`

## Bumping version
Edit the `fundus_image_toolbox/_version.py``

## Running Tests

This repository uses `unittest` for tests and `tox` for venv-sandboxed testing.

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

## What did we do?
- Added configurable cache-based weight handling across fovea-od-localization, quality, and registration inference
  - Added centralized download and extraction utility with retries, safer extraction, and rollback cleanup for partial extraction/download artifacts
  - Improved error handling/messages for download failures and manual workaround guidance.
  - This also fixes https://github.com/berenslab/fundus_image_toolbox/issues/29
  - From now on, pass `cache_dir` argument or set FIT_CACHE_DIR to use a non-default directory to store pretrained weights at. Backward-compatibility was ensured.
- quality_prediction
  - make the image size, that the images are resized to internally, configurable. Kept backward-compatible behavior by setting default to 512 (as in `<= v0.1.1`).
- utils
  - ImageTorchUtils:
    - Added robust greyscale image support (in `to_tensor`, `to_batch`, `squeeze`)
    - Added handling for more dtype and shape combinations
    - Fixed numpy 2 deprecated `np.array()` calls
  - basics: 
    - added greyscale support in `show()`, thanks to @Page0526
- registration
  - Fixed original code to handle modern torch versions: `align_corners` argument was wrong if torch version was >= 2.
- vessel segmentation:
  - Fixed `dtype` argument deprecation in newer numpy versions.
- circle_crop
  - Failure handling is now informative and non-fatal, e.g. when applying inside dataset loops: failures now emit UserWarning, return shape-preserving zero outputs, and mark failure with radius=-1.
  - Fixed the `process_img` uninitialized-variable bug by ensuring deterministic fallback failure outputs after exceptions. This fixes https://github.com/berenslab/fundus_image_toolbox/issues/28
  - Changed crop sizing behavior to square-only (size int), with backward-compatible tuple/list support using the first value.
- Bump to version 0.1.2
- dev: added tox-uv for unittests

