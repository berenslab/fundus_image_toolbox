import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from fundus_image_toolbox.registration.inference import get_config
from fundus_image_toolbox.utils.model_cache import (
    N_RETRY_DL,
    cleanup_extraction_artifacts,
    download,
    extract_tar_safely,
    extract_tar_safely_with_manifest,
    resolve_fit_cache_dir,
)


class _FakeResponse:
    def __init__(self, status_code=200, ok=True, body_chunks=None, headers=None):
        self.status_code = status_code
        self.ok = ok
        self._body_chunks = body_chunks or [b""]
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size=8192):
        del chunk_size
        for chunk in self._body_chunks:
            yield chunk


class _FailAfterFirstChunkResponse(_FakeResponse):
    def iter_content(self, chunk_size=8192):
        del chunk_size
        yield b"partial"
        raise requests.ConnectionError("stream interrupted")


class TestWeightCacheUtilities(unittest.TestCase):
    def test_resolve_fit_cache_dir_argument_overrides_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "explicit"
            env_dir = Path(tmp) / "env"
            with patch.dict(os.environ, {"FIT_CACHE_DIR": str(env_dir)}):
                out = resolve_fit_cache_dir(cache_dir=explicit)
            self.assertEqual(out, explicit.resolve())

    def test_resolve_fit_cache_dir_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_dir = Path(tmp) / "env"
            with patch.dict(os.environ, {"FIT_CACHE_DIR": str(env_dir)}):
                out = resolve_fit_cache_dir()
            self.assertEqual(out, env_dir.resolve())

    def test_extract_tar_safely_sanitizes_names_and_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "weights.tar.gz"
            destination = Path(tmp) / "out"

            with tarfile.open(archive, "w:gz") as tar:
                content = b"ok"
                safe_info = tarfile.TarInfo(name="folder:bad/file.txt")
                safe_info.size = len(content)
                tar.addfile(safe_info, fileobj=io.BytesIO(content))

                evil_info = tarfile.TarInfo(name="../evil.txt")
                evil_info.size = len(content)
                tar.addfile(evil_info, fileobj=io.BytesIO(content))

            extract_tar_safely(archive, destination, replace_colon_with="-")

            self.assertTrue((destination / "folder-bad" / "file.txt").exists())
            self.assertFalse((Path(tmp) / "evil.txt").exists())

    def test_download_403_retries_before_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.tar.gz"
            with patch("fundus_image_toolbox.utils.model_cache.requests.get") as mocked_get:
                mocked_get.return_value = _FakeResponse(status_code=403, ok=False)
                with self.assertRaises(RuntimeError) as ctx:
                    download(
                        url="https://zenodo.org/some/file",
                        target_path=target,
                        component_name="quality_prediction",
                        manual_file_name="weights.tar.gz",
                        manual_target_dir=Path(tmp),
                    )
            msg = str(ctx.exception)
            self.assertIn("HTTP 403", msg)
            self.assertIn("Manual workaround", msg)
            self.assertIn("after", msg)
            self.assertIn("attempts", msg)
            self.assertEqual(mocked_get.call_count, 1 + N_RETRY_DL)

    def test_download_non_ok_retries_before_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.tar.gz"
            with patch("fundus_image_toolbox.utils.model_cache.requests.get") as mocked_get:
                mocked_get.return_value = _FakeResponse(status_code=500, ok=False)
                with self.assertRaises(RuntimeError) as ctx:
                    download(
                        url="https://zenodo.org/some/file",
                        target_path=target,
                        component_name="quality_prediction",
                        manual_file_name="weights.tar.gz",
                        manual_target_dir=Path(tmp),
                    )
            msg = str(ctx.exception)
            self.assertIn("after", msg)
            self.assertIn("attempts", msg)
            self.assertIn("please open an issue", msg.lower())
            self.assertIn("Manual workaround", msg)
            self.assertEqual(mocked_get.call_count, 1 + N_RETRY_DL)

    def test_download_partial_file_is_removed_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.tar.gz"
            side_effect = [_FailAfterFirstChunkResponse()] * (1 + N_RETRY_DL)
            with patch(
                "fundus_image_toolbox.utils.model_cache.requests.get",
                side_effect=side_effect,
            ):
                with self.assertRaises(RuntimeError):
                    download(
                        url="https://zenodo.org/some/file",
                        target_path=target,
                        component_name="quality_prediction",
                        manual_file_name="weights.tar.gz",
                        manual_target_dir=Path(tmp),
                    )
            self.assertFalse(target.exists())

    def test_cleanup_extraction_artifacts_does_not_fail_non_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            root.mkdir(parents=True, exist_ok=True)
            keep_dir = root / "keep"
            keep_dir.mkdir(parents=True, exist_ok=True)
            keep_file = keep_dir / "preexisting.txt"
            keep_file.write_text("keep")

            manifest = {
                "files": [root / "partial.txt"],
                "dirs": [keep_dir, root],
            }
            manifest["files"][0].write_text("partial")
            cleanup_extraction_artifacts(manifest)

            self.assertFalse((root / "partial.txt").exists())
            self.assertTrue(keep_file.exists())

    def test_extract_tar_with_manifest_tracks_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "weights.tar.gz"
            destination = Path(tmp) / "out"
            with tarfile.open(archive, "w:gz") as tar:
                content = b"ok"
                info = tarfile.TarInfo(name="folder/file.txt")
                info.size = len(content)
                tar.addfile(info, fileobj=io.BytesIO(content))

            manifest = extract_tar_safely_with_manifest(archive, destination)
            self.assertTrue((destination / "folder" / "file.txt").exists())
            self.assertTrue(any(p.name == "file.txt" for p in manifest["files"]))


class TestRegistrationConfigWeightPathResolution(unittest.TestCase):
    def test_registration_placeholder_prefers_explicit_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_cache = Path(tmp) / "env_cache"
            explicit_cache = Path(tmp) / "explicit_cache"
            with patch.dict(os.environ, {"FIT_CACHE_DIR": str(env_cache)}):
                cfg = get_config(
                    {"device": "cpu", "model_save_path": "default"},
                    cache_dir=explicit_cache,
                )
            expected = (
                explicit_cache.resolve() / "registration" / "SuperRetina.pth"
            )
            self.assertEqual(Path(cfg["model_save_path"]), expected)

    def test_registration_placeholder_uses_legacy_when_cache_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_cache = Path(tmp) / "env_cache"
            legacy_file = Path(tmp) / "legacy" / "SuperRetina.pth"
            legacy_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_file.write_bytes(b"stub")

            with patch.dict(os.environ, {"FIT_CACHE_DIR": str(env_cache)}):
                with patch(
                    "fundus_image_toolbox.registration.inference.WEIGHT_PATH",
                    str(legacy_file),
                ):
                    cfg = get_config({"device": "cpu", "model_save_path": None})

            self.assertEqual(Path(cfg["model_save_path"]), legacy_file.resolve())
