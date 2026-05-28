import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fundus_image_toolbox.utils.model_cache import component_cache_dir, has_all_paths
from fundus_image_toolbox.vessel_segmentation.default import (
    ENSEMBLE_WEIGHT_FILES,
    VESSEL_SEGMENTATION_COMPONENT_NAME,
)
from fundus_image_toolbox.vessel_segmentation.weights import (
    download_weights,
    ensure_models_dir,
)


RUN_INTEGRATION = os.environ.get("FIT_RUN_WEIGHT_DOWNLOAD_TESTS", "0") == "1"


def _has_ensemble_weights(models_dir: Path) -> bool:
    return has_all_paths(models_dir, ENSEMBLE_WEIGHT_FILES)


class TestVesselSegmentationWeightWrapper(unittest.TestCase):
    def test_ensure_models_dir_delegates_to_fit_cache(self):
        fit_cache = component_cache_dir(VESSEL_SEGMENTATION_COMPONENT_NAME)
        expected = fit_cache.resolve()

        with patch(
            "fundus_image_toolbox.vessel_segmentation.weights.segqc_ensure_models_dir",
            return_value=expected,
        ) as segqc_ensure:
            resolved = ensure_models_dir()

        segqc_ensure.assert_called_once_with(models_dir=expected)

    def test_download_weights_delegates_to_fit_cache(self):
        fit_cache = component_cache_dir(VESSEL_SEGMENTATION_COMPONENT_NAME)
        expected = fit_cache.resolve()

        with patch(
            "fundus_image_toolbox.vessel_segmentation.weights.segqc_download_weights",
            return_value=expected,
        ) as segqc_download:
            resolved = download_weights()

        segqc_download.assert_called_once_with(models_dir=expected)


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set FIT_RUN_WEIGHT_DOWNLOAD_TESTS=1 to run integration-like download tests.",
)
class TestVesselSegmentationWeightDownload(unittest.TestCase):
    def test_http_download_recreates_cache_and_restores_prior_state(self):
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
