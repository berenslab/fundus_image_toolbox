import unittest
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from fundus_circle_crop import circle_crop

DIR = os.path.join(os.path.dirname(__file__))

class TestCircleCrop(unittest.TestCase):
    def setUp(self):
        self.image_path = os.path.join(DIR, '..', 'fundus1.jpg')
        self.image = plt.imread(self.image_path)
        self.size = (100, 100)
        self.shape = self.size + (3,)

    def test_circle_crop_with_size(self):
        cropped_image = circle_crop(self.image, size=self.size)
        self.assertEqual(cropped_image.shape, self.shape)

    def test_circle_crop_with_tensor(self):
        tensor_image = torch.from_numpy(self.image.copy())
        cropped_image = circle_crop(tensor_image, to_numpy=False)
        self.assertIsInstance(cropped_image, torch.Tensor)
    
    def test_circle_crop_returnall(self):
        # Test the circle_crop function
        img_aligned, mask, center, radius = circle_crop(self.image, size=self.size, return_all=True, to_numpy=True)

        # Assert the output types
        self.assertIsInstance(img_aligned, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(center, np.ndarray)
        self.assertIsInstance(float(center[0]), float)
        self.assertIsInstance(float(center[1]), float)
        self.assertIsInstance(float(radius), float)

        # Assert the output shapes and values
        self.assertEqual(img_aligned.shape, self.shape)
        self.assertEqual(mask.shape, self.size)
        self.assertEqual(len(center), 2)
        self.assertGreaterEqual(radius, 0)

if __name__ == '__main__':
    unittest.main()