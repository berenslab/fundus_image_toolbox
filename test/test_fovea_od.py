import unittest
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from fundus_image_toolbox.fovea_od_localization import ODFoveaModel, load_fovea_od_model

DIR = os.path.join(os.path.dirname(__file__))
fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")
DEVICE = "cpu"


class TestFoveaODModel(unittest.TestCase):
    def setUp(self):
        self.image = Image.open(fundus1_path)
        transform = Compose([Resize(350, antialias=True), CenterCrop(350), ToTensor()])
        self.image_batch = torch.stack([transform(self.image) for _ in range(2)])
        self.device = DEVICE

    def test_load_fovea_od_model(self):
        model, checkpoint_path = load_fovea_od_model(
            "default", device=self.device, return_test_dataloader=False
        )
        self.assertIsInstance(model, ODFoveaModel)
        self.assertEqual(self.device, str(model.device))

        fovea_x, fovea_y, od_x, od_y = model.predict(self.image)
        self.assertIsInstance(float(fovea_x), float)
        self.assertIsInstance(float(fovea_y), float)
        self.assertIsInstance(float(od_x), float)
        self.assertIsInstance(float(od_y), float)

        labels = model.predict(self.image_batch)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), 2)
        self.assertIsInstance(labels[0], np.ndarray)
        self.assertEqual(len(labels[0]), 4)
        self.assertIsInstance(float(labels[0][0]), float)


if __name__ == "__main__":
    unittest.main()
