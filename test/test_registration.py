import os
import unittest
import numpy as np
from PIL import Image
from fundus_image_toolbox.registration import (
    register,
    load_registration_model,
    DEFAULT_CONFIG,
    SuperRetina,
)
from cv2 import BFMatcher

DIR = os.path.join(os.path.dirname(__file__))

fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")
fundus2_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus2.jpg")


class TestRegistration(unittest.TestCase):
    def setUp(self):
        # Initialize test variables here
        with Image.open(fundus1_path) as image1:
            self.image1 = image1.copy()
        with Image.open(fundus2_path) as image2:
            self.image2 = image2.copy()
        self.config = DEFAULT_CONFIG
        self.config["device"] = "cpu"
        self.model, self.matcher = load_registration_model(self.config)

    def tearDown(self):
        if hasattr(self, "image1") and self.image1 is not None:
            self.image1.close()
        if hasattr(self, "image2") and self.image2 is not None:
            self.image2.close()

    def test_load_registration_model(self):
        # Test the load_registration_model function
        self.assertIsInstance(self.model, SuperRetina)
        self.assertIsInstance(self.matcher, BFMatcher)

    def test_register(self):
        # Test the register function
        registered_image = register(
            self.image1,
            self.image2,
            show=False,
            show_mapping=False,
            config=self.config,
            model=self.model,
            matcher=self.matcher,
        )
        self.assertIsInstance(registered_image, np.ndarray)


if __name__ == "__main__":
    unittest.main()
