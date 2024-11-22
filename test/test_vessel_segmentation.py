import os
import unittest
import numpy as np
import torch
from PIL import Image
from fundus_image_toolbox.vessel_segmentation import (
    load_segmentation_ensemble,
    ensemble_predict_segmentation,
    plot_masks,
    save_masks,
    load_masks_from_filenames,
    FR_UNet,
)

DIR = os.path.join(os.path.dirname(__file__))

fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")


class TestVesselSegmentation(unittest.TestCase):
    def setUp(self):
        # Initialize test variables here
        self.image = Image.open(fundus1_path)
        self.device = "cpu"
        self.model = load_segmentation_ensemble(device=self.device)

    def test_load_segmentation_ensemble(self):
        # Test the load_segmentation_ensemble function
        for model in self.model:
            self.assertIsInstance(model, FR_UNet)

    def test_ensemble_predict_segmentation(self):
        # Test the ensemble_predict_segmentation function
        segmentation = ensemble_predict_segmentation(
            self.model, self.image, device=self.device
        )
        self.assertIsInstance(segmentation, np.ndarray)
        self.assertEqual(segmentation.shape, (self.image.height, self.image.width))

    def test_plot_masks(self):
        # Test the plot_masks function
        segmentation = ensemble_predict_segmentation(
            self.model, self.image, device=self.device
        )
        plot_masks(self.image, segmentation)

    def test_save_and_load_masks(self):
        # Test the save_masks and load_masks_from_filenames functions
        target_dir = os.path.join(DIR, "masks")
        mask = ensemble_predict_segmentation(self.model, self.image, device=self.device)
        save_masks(fundus1_path, mask, target_dir)
        loaded_mask = load_masks_from_filenames(fundus1_path, target_dir)
        self.assertIsInstance(loaded_mask, np.ndarray)
        self.assertEqual(loaded_mask.shape, mask.shape)

        # Clean up
        for file in os.listdir(os.path.join(target_dir)):
            os.remove(os.path.join(target_dir, file))


if __name__ == "__main__":
    unittest.main()
