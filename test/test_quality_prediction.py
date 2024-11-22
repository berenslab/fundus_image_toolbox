import unittest
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from fundus_image_toolbox.quality_prediction import (
    load_quality_ensemble,
    ensemble_predict_quality,
    plot_quality,
    FundusQualityModel,
)

DIR = os.path.join(os.path.dirname(__file__))

fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")


class TestQualityPrediction(unittest.TestCase):
    def setUp(self):
        # Initialize test variables here
        self.image_path = fundus1_path
        self.transforms = Compose([ToTensor(), Resize((350, 350))])
        self.image = self.transforms(Image.open(self.image_path)).unsqueeze(0)
        self.model = load_quality_ensemble(device="cpu")

    def test_load_quality_ensemble(self):
        # Test the load_quality_ensemble function
        for model in self.model:
            self.assertIsInstance(model, FundusQualityModel)

    def test_ensemble_predict_quality(self):
        # Test the ensemble_predict_quality function
        conf, label = ensemble_predict_quality(self.model, self.image)
        self.assertIsInstance(float(conf), float)
        self.assertIsInstance(float(label), float)

    def test_plot_quality(self):
        # Test the plot_quality function
        plot_quality(self.image.squeeze(), 0.5, 1)


if __name__ == "__main__":
    unittest.main()
