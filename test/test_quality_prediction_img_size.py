import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from fundus_image_toolbox.quality_prediction.scripts.ensemble_inference import (
    any_to_tensor,
    ensemble_predict,
)
from fundus_image_toolbox.quality_prediction.scripts.model import (
    FundusQualityModel,
    _MIXED_SIZE_BATCH_WARNING,
)


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

    def test_predict_from_batch_uses_align_short_edge_crop_on_mismatch(self):
        model = FundusQualityModel.__new__(FundusQualityModel)
        model.config = SimpleNamespace(device="cpu")
        model.model = _DummyModel()
        model.load_checkpoint = MagicMock()

        image_batch = [torch.zeros(3, 12, 16), torch.zeros(3, 8, 10)]

        with patch(
            "fundus_image_toolbox.quality_prediction.scripts.model.Img"
        ) as mock_img_cls:
            mock_img = MagicMock()
            mock_img.to_batch.return_value = mock_img
            mock_img.img = torch.zeros(2, 3, 8, 8)
            mock_img_cls.return_value = mock_img
            with patch(
                "fundus_image_toolbox.quality_prediction.scripts.model.get_transforms"
            ) as mock_get_transforms:
                mock_get_transforms.return_value = lambda pil_image: torch.zeros(
                    (3, 8, 8)
                )
                model.predict_from_batch(
                    image_batch, threshold=None, load_best=False, numpy_cpu=True
                )

        mock_img.to_batch.assert_called_once_with(
            on_mismatch="align_short_edge_crop",
            mismatch_warning=_MIXED_SIZE_BATCH_WARNING,
        )

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

    def test_any_to_tensor_converts_pil_image(self):
        pil_image = Image.new("RGB", (10, 12))
        tensor = any_to_tensor(pil_image)
        self.assertEqual(tensor.shape, (3, 12, 10))

    def test_ensemble_predict_routes_pil_list_to_batch_path(self):
        batch_model = MagicMock()
        batch_model.predict_from_batch.return_value = np.array([0.6, 0.8])
        pil_images = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
        ensemble_predict([batch_model], pil_images, threshold=0.5)
        batch_model.predict_from_batch.assert_called_once()
        batch_model.predict_from_image.assert_not_called()
        passed_batch = batch_model.predict_from_batch.call_args.args[0]
        self.assertEqual(len(passed_batch), 2)
        self.assertTrue(all(isinstance(img, torch.Tensor) for img in passed_batch))

    def test_ensemble_predict_routes_single_pil_to_single_path(self):
        single_model = MagicMock()
        single_model.predict_from_image.return_value = np.array([0.7])
        pil_image = Image.new("RGB", (10, 10))
        ensemble_predict([single_model], pil_image, threshold=0.5)
        single_model.predict_from_image.assert_called_once_with(
            pil_image, threshold=None, load_best=False, img_size=512
        )
        single_model.predict_from_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
