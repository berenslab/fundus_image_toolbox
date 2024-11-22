import unittest
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.transforms.functional import to_tensor
from fundus_image_toolbox.utilities import ImageTorchUtils as Img

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
