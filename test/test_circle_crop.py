import unittest
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from unittest.mock import patch
from fundus_image_toolbox.circle_crop import crop
from fundus_image_toolbox.circle_crop import eyeq_preprocess

DIR = os.path.join(os.path.dirname(__file__))
fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")


class TestCircleCrop(unittest.TestCase):
    def setUp(self):
        self.image_path = fundus1_path
        self.image = plt.imread(self.image_path)
        self.size = 100
        self.shape = (self.size, self.size, 3)

    def test_circle_crop_with_size(self):
        cropped_image = crop(self.image, size=self.size)
        self.assertEqual(cropped_image.shape, self.shape)

    def test_circle_crop_with_tensor(self):
        tensor_image = torch.from_numpy(self.image.copy())
        cropped_image = crop(tensor_image, to_numpy=False)
        self.assertIsInstance(cropped_image, torch.Tensor)

    def test_circle_crop_returnall(self):
        # Test the circle_crop function
        img_aligned, mask, center, radius = crop(
            self.image, size=self.size, return_all=True, to_numpy=True
        )

        # Assert the output types
        self.assertIsInstance(img_aligned, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(center, np.ndarray)
        self.assertIsInstance(float(center[0]), float)
        self.assertIsInstance(float(center[1]), float)
        self.assertIsInstance(float(radius), float)

        # Assert the output shapes and values
        self.assertEqual(img_aligned.shape, self.shape)
        self.assertEqual(mask.shape, (self.size, self.size))
        self.assertEqual(len(center), 2)
        self.assertGreaterEqual(radius, 0)
        self.assertEqual(center.dtype, np.float32)
        self.assertEqual(np.asarray(radius).dtype, np.float32)

    @patch(
        "fundus_image_toolbox.circle_crop.eyeq_preprocess.process_without_gb",
        side_effect=RuntimeError("synthetic failure"),
    )
    def test_circle_crop_failure_returns_zero_numpy(self, _):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            img_aligned, mask, center, radius = crop(
                self.image, size=self.size, return_all=True, to_numpy=True
            )

        self.assertTrue(any(w.category is UserWarning for w in caught_warnings))
        self.assertEqual(img_aligned.shape, self.shape)
        self.assertEqual(mask.shape, (self.size, self.size))
        self.assertTrue((img_aligned == 0).all())
        self.assertTrue((mask == 0).all())
        self.assertTrue(np.allclose(center, np.array([0.0, 0.0], dtype=np.float32)))
        self.assertEqual(float(radius), -1.0)
        self.assertEqual(center.dtype, np.float32)
        self.assertEqual(np.asarray(radius).dtype, np.float32)

    def test_circle_crop_uses_first_size_value_for_legacy_tuple(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            img_aligned, mask, center, radius = crop(
                self.image, size=(64, 80), return_all=True, to_numpy=True
            )

        self.assertTrue(any(w.category is UserWarning for w in caught_warnings))
        self.assertEqual(img_aligned.shape, (64, 64, 3))
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(center.dtype, np.float32)
        self.assertEqual(np.asarray(radius).dtype, np.float32)

    def test_circle_crop_mixed_batch_failure_torch(self):
        original_process_without_gb = eyeq_preprocess.process_without_gb
        call_count = {"n": 0}

        def side_effect(img):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("synthetic second-item failure")
            return original_process_without_gb(img)

        with patch(
            "fundus_image_toolbox.circle_crop.eyeq_preprocess.process_without_gb",
            side_effect=side_effect,
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                imgs, masks, centers, radii = crop(
                    [self.image, self.image],
                    size=self.size,
                    return_all=True,
                    to_numpy=False,
                )

        self.assertTrue(any(w.category is UserWarning for w in caught_warnings))
        self.assertIsInstance(imgs, torch.Tensor)
        self.assertIsInstance(masks, torch.Tensor)
        self.assertIsInstance(centers, torch.Tensor)
        self.assertIsInstance(radii, torch.Tensor)
        self.assertEqual(imgs.shape[0], 2)
        self.assertEqual(masks.shape[0], 2)
        self.assertTrue(torch.all(imgs[1] == 0))
        self.assertTrue(torch.all(masks[1] == 0))
        self.assertTrue(torch.all(centers[1] == 0))
        self.assertEqual(float(radii[1]), -1.0)
        self.assertEqual(centers.dtype, torch.float32)
        self.assertEqual(radii.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
