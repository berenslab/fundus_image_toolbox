import unittest
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch
from unittest.mock import patch
from torchvision.transforms.functional import to_tensor
import fundus_image_toolbox.utils as fit_utils
from fundus_image_toolbox.utils import ImageTorchUtils as Img

DIR = os.path.join(os.path.dirname(__file__))

fundus1_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus1.jpg")
fundus2_path = os.path.join(DIR, "..", "0_example_usage", "imgs", "fundus2.jpg")

fundus1_plt = plt.imread(fundus1_path)
fundus1_img = Image.open(fundus1_path)
fundus1_cv2 = cv2.imread(fundus1_path)


class TestImageTorchUtils(unittest.TestCase):
    def test_to_tensor(self):
        # Test the to_tensor function
        imgs_rgb = [fundus1_path, fundus1_plt, fundus1_img]

        # Convert to rgb tensor from rgb image
        outs = [Img(img).to_tensor().img for img in imgs_rgb]

        # Convert to rgb tensor from bgr image
        bgr_out = Img(fundus1_cv2).to_tensor(from_cspace="bgr").img
        # Alternatively, use a sequence of calls:
        bgr_out = (
            Img(fundus1_cv2)
            .to_tensor()
            .to_cspace(from_cspace="bgr", to_cspace="rgb")
            .img
        )
        outs.append(bgr_out)

        assert all([torch.equal(outs[0], out) for out in outs[1:]])

    def test_to_cspace(self):
        # Test the to_cspace function
        imgs_rgb = [fundus1_path, fundus1_plt, fundus1_img]
        img_bgr = fundus1_cv2

        # Convert from rgb to bgr
        outs = [
            Img(img).to_tensor().to_cspace(from_cspace="rgb", to_cspace="bgr").img
            for img in imgs_rgb
        ]

        assert all([torch.equal(outs[0], out) for out in outs[1:]])
        assert torch.equal(Img(img_bgr).to_tensor().img, outs[0])

        # Convert to grey scale
        outs = [
            Img(img).to_tensor().to_cspace(from_cspace="rgb", to_cspace="grey").img
            for img in imgs_rgb
        ]

        assert all([torch.equal(outs[0], out) for out in outs[1:]])
        assert torch.equal(
            Img(img_bgr).to_tensor().to_cspace(from_cspace="bgr", to_cspace="grey").img,
            outs[0],
        )

    def test_set_channel_dim(self):
        # Test set_channel_dim function
        imgs = [fundus1_path, fundus1_plt, fundus1_img, fundus1_cv2]

        # Set channel dimension to the last dimension
        outs = [Img(img).to_tensor().set_channel_dim(-1).img for img in imgs]

        # Check if the channel dimension is the last dimension
        assert all([out.shape[-1] == 3 for out in outs])

    def test_to_pil(self):
        # Test to_pil function
        imgs = [fundus1_path, fundus1_plt, fundus1_img, fundus1_cv2]

        # Convert to pil image
        outs = [Img(img).to_tensor().to_pil().img for img in imgs]

        # Check if the output is a pil image
        assert all([isinstance(out, Image.Image) for out in outs])

    def test_to_numpy(self):
        # Test to_numpy functions
        imgs = [fundus1_path, fundus1_plt, fundus1_img, fundus1_cv2]

        # Convert to numpy array
        outs = [Img(img).to_tensor().to_numpy().img for img in imgs]

        # Check if the output is a numpy array
        assert all([isinstance(out, np.ndarray) for out in outs])
        assert all([out.dtype == np.uint8 for out in outs])

        # Alternatively, chain the to_numpy and to_unit8 functions
        outs = [Img(img).to_tensor().to_numpy().to_uint8().img for img in imgs]

        # Check if the output is a numpy array
        assert all([isinstance(out, np.ndarray) for out in outs])
        assert all([out.dtype == np.uint8 for out in outs])

        # Convert to float32 numpy array
        outs = [Img(img).to_tensor().to_numpy("float32").img for img in imgs]

        # Check if the output is a numpy array
        assert all([isinstance(out, np.ndarray) for out in outs])
        assert all([out.dtype == np.float32 for out in outs])

        # Alternatively, chain the to_numpy and to_float32 functions
        outs = [Img(img).to_tensor().to_numpy().to_float32().img for img in imgs]

        # Check if the output is a numpy array
        assert all([isinstance(out, np.ndarray) for out in outs])
        assert all([out.dtype == np.float32 for out in outs])

    def test_to_batch(self):
        # Test the to_batch function

        # Test objects
        list_fundus_plt = [fundus1_plt, fundus1_plt]
        list_fundus_img = [fundus1_img, fundus1_img]
        list_fundus_cv2 = [fundus1_cv2, fundus1_cv2]

        batch_fundus = torch.stack([to_tensor(fundus1_img), to_tensor(fundus1_img)])

        ndarray_fundus_plt = np.array(list_fundus_plt)
        ndarray_fundus_img = np.array(list_fundus_img)
        ndarray_fundus_cv2 = np.array(list_fundus_cv2)

        # Apply the to_batch function
        outs = [
            Img(fundus).to_batch().img
            for fundus in [list_fundus_plt, list_fundus_img, list_fundus_cv2]
        ]
        outs += [
            Img(fundus).to_batch().img
            for fundus in [ndarray_fundus_plt, ndarray_fundus_img, ndarray_fundus_cv2]
        ]

        # Check if the output is a tensor
        for out in outs:
            assert isinstance(out, torch.Tensor)
            assert (
                out.shape[0] == len(batch_fundus)
                and out.shape[1] == 3
                and isinstance(out, torch.Tensor)
            )

        # Single image to batch
        out = Img(fundus1_img).to_batch().img
        assert len(out.shape) == 4 and isinstance(out, torch.Tensor)

        # Single path to batch
        out = Img(fundus1_path).to_batch().img
        assert len(out.shape) == 4 and isinstance(out, torch.Tensor)

    def test_to_batch_with_grayscale_mask_ndims_hint(self):
        masks = np.stack([fundus1_plt[:, :, 0], fundus1_plt[:, :, 0]], axis=0)
        out = Img(masks).to_batch(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert len(out.shape) == 4
        assert out.shape[0] == 2
        assert out.shape[1] == 1

    def test_to_tensor_additional_dummy_cases(self):
        rng = np.random.default_rng(0)
        h, w = 6, 8

        img_2d_u8 = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        img_2d_f32 = rng.random((h, w), dtype=np.float32)
        img_hwc_u8 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        img_hwc_f32 = rng.random((h, w, 3), dtype=np.float32)
        img_1hw_u8 = rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)
        img_chw_u8 = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
        torch_hw = torch.rand(h, w)

        out = Img(img_2d_u8).to_tensor(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, h, w)
        assert out.dtype == torch.float32

        out = Img(img_2d_f32).to_tensor(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, h, w)
        assert out.dtype == torch.float32

        out = Img(img_hwc_u8).to_tensor(img_ndims=3).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, h, w)
        assert out.dtype == torch.float32

        out = Img(img_hwc_f32).to_tensor(img_ndims=3).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, h, w)
        assert out.dtype == torch.float32

        out = Img(img_1hw_u8).to_tensor(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, h, w)
        assert out.dtype == torch.float32

        out = Img(img_chw_u8).to_tensor(img_ndims=3).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, h, w)
        assert out.dtype == torch.float32

        out = Img(torch_hw).to_tensor(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, h, w)
        assert out.dtype == torch.float32

    def test_to_batch_additional_dummy_cases(self):
        rng = np.random.default_rng(1)
        h, w = 6, 8

        img_2d_u8 = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        img_hwc_u8 = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        img_1hw_u8 = rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)

        batch_list_2d = [img_2d_u8, img_2d_u8.copy()]
        batch_array_bhw = np.stack(batch_list_2d, axis=0)
        batch_list_hwc = [img_hwc_u8, img_hwc_u8.copy()]
        batch_array_bhwc = np.stack(batch_list_hwc, axis=0)
        batch_list_1hw = [img_1hw_u8, img_1hw_u8.copy()]

        out = Img(batch_list_2d).to_batch(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1, h, w)

        out = Img(batch_array_bhw).to_batch(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1, h, w)

        out = Img(batch_list_hwc).to_batch(img_ndims=3).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3, h, w)

        out = Img(batch_array_bhwc).to_batch(img_ndims=3).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3, h, w)

        out = Img(batch_list_1hw).to_batch(img_ndims=2).img
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1, h, w)

        # Intentional mismatch from notebook: BHW treated as color batch-like input.
        with self.assertRaises(TypeError):
            Img(batch_array_bhw).to_batch(img_ndims=3).img

    def test_show_empty_ndarray_raises(self):
        empty_img = np.array([])
        with self.assertRaises(ValueError) as exc_info:
            fit_utils.show(empty_img)
        assert "empty" in str(exc_info.exception).lower()

    def test_show_list_filters_empty_images(self):
        empty_img = np.array([])
        with patch("matplotlib.pyplot.imshow") as mock_imshow, patch(
            "matplotlib.pyplot.show"
        ) as mock_show:
            fit_utils.show([fundus1_plt, empty_img])

        assert mock_imshow.call_count == 1
        mock_show.assert_called_once()

    def test_show_single_grayscale_image_uses_gray_cmap(self):
        fundus1_gs = fundus1_plt[:, :, 0]
        with patch("matplotlib.pyplot.imshow") as mock_imshow, patch(
            "matplotlib.pyplot.show"
        ) as mock_show:
            fit_utils.show(fundus1_gs)

        mock_show.assert_called_once()
        shown_img = mock_imshow.call_args.args[0]
        shown_kwargs = mock_imshow.call_args.kwargs
        assert shown_img.ndim == 2
        assert np.array_equal(shown_img, fundus1_gs)
        assert shown_kwargs.get("cmap") == "gray"

    def test_show_list_mixed_grayscale_and_rgb(self):
        fundus1_gs = fundus1_plt[:, :, 0]
        with patch("matplotlib.pyplot.imshow") as mock_imshow, patch(
            "matplotlib.pyplot.show"
        ) as mock_show:
            fit_utils.show([fundus1_gs, fundus1_plt])

        mock_show.assert_called_once()
        assert mock_imshow.call_count == 2

        gs_kwargs = mock_imshow.call_args_list[0].kwargs
        rgb_kwargs = mock_imshow.call_args_list[1].kwargs
        assert gs_kwargs.get("cmap") == "gray"
        assert "cmap" not in rgb_kwargs

    def test_combi(self):
        # Test the to_cspace and set_channel_dim function
        out = (
            Img(fundus1_plt)
            .to_tensor()
            .to_cspace(from_cspace="rgb", to_cspace="gray")
            .set_channel_dim(-1)
            .img
        )
        assert out.shape[-1] == 1


if __name__ == "__main__":
    unittest.main()
