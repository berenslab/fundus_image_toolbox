import os
import shutil
import unittest
from pathlib import Path

from fundus_image_toolbox.utils.model_cache import component_cache_dir, has_all_paths
from fundus_image_toolbox.vessel_segmentation.default import (
    ENSEMBLE_WEIGHT_FILES,
    VESSEL_SEGMENTATION_COMPONENT_NAME,
)
from fundus_image_toolbox.vessel_segmentation.weights import download_weights


RUN_INTEGRATION = os.environ.get("FIT_RUN_WEIGHT_DOWNLOAD_TESTS", "0") == "1"


def _has_ensemble_weights(models_dir: Path) -> bool:
    return has_all_paths(models_dir, ENSEMBLE_WEIGHT_FILES)


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 to run integration-like download tests.",
)
class TestVesselSegmentationWeightDownload(unittest.TestCase):
    def test_git_download_recreates_cache_and_restores_prior_state(self):
        cache_dir = component_cache_dir(VESSEL_SEGMENTATION_COMPONENT_NAME)
        backup_dir = cache_dir.with_name(f"{cache_dir.name}.unittest-backup")

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        try:
            if cache_dir.exists():
                cache_dir.rename(backup_dir)

            self.assertFalse(_has_ensemble_weights(cache_dir))

            models_dir = download_weights()
            self.assertEqual(models_dir.resolve(), cache_dir.resolve())
            self.assertTrue(_has_ensemble_weights(models_dir))
            for name in ENSEMBLE_WEIGHT_FILES:
                self.assertTrue((models_dir / name).is_file())
        finally:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            if backup_dir.exists():
                backup_dir.rename(cache_dir)


if __name__ == "__main__":
    unittest.main()
