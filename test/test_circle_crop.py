import unittest
import os
import torch
from PIL import Image
from fundus_circle_crop import circle_crop

DIR = os.path.join(os.path.dirname(__file__))

class TestCircleCrop(unittest.TestCase):
    def setUp(self):
        self.image_path = os.path.join(DIR, '..', 'fundus1.jpg')
        self.image = Image.open(self.image_path)

    def test_circle_crop_with_size(self):
        size = (100, 100)
        shape = (100, 100, 3)
        cropped_image = circle_crop(self.image, size=size)
        self.assertEqual(cropped_image.shape, shape)

    def test_circle_crop_with_tensor(self):
        tensor_image = torch.from_numpy(self.image)
        cropped_image = circle_crop(tensor_image, to_numpy=False)
        self.assertIsInstance(cropped_image, torch.Tensor)

if __name__ == '__main__':
    unittest.main()