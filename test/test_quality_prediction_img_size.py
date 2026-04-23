import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fundus_image_toolbox.quality_prediction.scripts.ensemble_inference import (
    ensemble_predict,
)
from fundus_image_toolbox.quality_prediction.scripts.model import FundusQualityModel


class _DummyModel:
    def eval(self):
        return None

    def __call__(self, image_batch):
        return torch.zeros((image_batch.shape[0], 1), dtype=torch.float32)


class TestQualityPredictionImgSize(unittest.TestCase):
    def test_predict_from_image_passes_custom_img_size(self):
        model = FundusQualityModel.__new__(FundusQualityModel)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        with patch(
            "fundus_image_toolbox.quality_prediction.scripts.model.get_transforms"
        ) as mock_get_transforms:
            mock_get_transforms.return_value = lambda pil_image: torch.zeros((3, 16, 16))
            with patch.object(
                FundusQualityModel, "predict_from_batch", return_value=np.array([0.9])
            ) as mock_predict_from_batch:
                model.predict_from_image(
                    image, threshold=None, load_best=False, img_size=256
                )

        mock_get_transforms.assert_called_once_with(split="test", img_size=256)
        self.assertEqual(mock_predict_from_batch.call_args.kwargs["img_size"], 256)
        self.assertFalse(mock_predict_from_batch.call_args.kwargs["transform"])

    def test_predict_from_batch_uses_default_img_size(self):
        model = FundusQualityModel.__new__(FundusQualityModel)
        model.config = SimpleNamespace(device="cpu")
        model.model = _DummyModel()
        model.load_checkpoint = MagicMock()

        image_batch = np.zeros((1, 3, 24, 24), dtype=np.float32)

        with patch(
            "fundus_image_toolbox.quality_prediction.scripts.model.get_transforms"
        ) as mock_get_transforms:
            mock_get_transforms.return_value = lambda pil_image: torch.zeros((3, 24, 24))
            preds = model.predict_from_batch(
                image_batch, threshold=None, load_best=False, numpy_cpu=True
            )

        mock_get_transforms.assert_called_once_with(split="test", img_size=512)
        self.assertEqual(preds.shape[0], 1)

    def test_ensemble_predict_passes_img_size_to_model_predictions(self):
        batch_model = MagicMock()
        batch_model.predict_from_batch.return_value = np.array([[0.7]])
        batch_image = [torch.zeros((3, 20, 20), dtype=torch.float32)]
        ensemble_predict([batch_model], batch_image, threshold=0.5, img_size=224)
        self.assertEqual(batch_model.predict_from_batch.call_args.kwargs["img_size"], 224)

        single_model = MagicMock()
        single_model.predict_from_image.return_value = np.array([0.7])
        single_image = torch.zeros((3, 20, 20), dtype=torch.float32)
        ensemble_predict([single_model], single_image, threshold=0.5)
        self.assertEqual(single_model.predict_from_image.call_args.kwargs["img_size"], 512)


if __name__ == "__main__":
    unittest.main()
