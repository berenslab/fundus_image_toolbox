import os
import tempfile
import unittest
from pathlib import Path

from fundus_image_toolbox.quality_prediction.scripts.model import download_weights
from fundus_image_toolbox.registration.inference import (
    DEFAULT_MODEL_SAVE_PATH,
    download_weights as download_registration_weights,
    get_config,
)


RUN_INTEGRATION = os.environ.get("FIT_RUN_WEIGHT_DOWNLOAD_TESTS", "0") == "1"


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 to run integration-like download tests.",
)
class TestWeightDownloadIntegration(unittest.TestCase):
    def test_registration_download_to_cache_and_config_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            target = download_registration_weights(cache_dir=cache_dir)
            self.assertTrue(target.exists())

            cfg = get_config(
                {"device": "cpu", "model_save_path": DEFAULT_MODEL_SAVE_PATH},
                cache_dir=cache_dir,
            )
            self.assertEqual(Path(cfg["model_save_path"]), target)

    def test_quality_broken_archive_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp) / "quality_prediction"
            component_dir.mkdir(parents=True, exist_ok=True)
            broken_archive = component_dir / "weights.tar.gz"
            broken_archive.write_bytes(b"not a valid tar archive")

            with self.assertRaises(RuntimeError) as ctx:
                download_weights(cache_dir=tmp)
            msg = str(ctx.exception).lower()
            self.assertIn("failed to extract", msg)
