from argparse import ArgumentParser
import sys
import pickle
import os
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
from PIL import Image
from .default import MODELS_DIR
from fundus_image_toolbox.utilities import ImageTorchUtils as Img
from fundus_image_toolbox.utilities import seed_everything

# Clone the repository if not present
try:
    from .segmentation.utils.notebook_utils import clahe_equalized, get_ensemble
    from .segmentation.utils.model_definition import FR_UNet
except ImportError:
    from .clone import clone_repo

    this_dir = os.path.dirname(os.path.abspath(__file__))
    clone_repo(target_dir=os.path.join(this_dir, "segmentation").__str__())
    from .segmentation.utils.notebook_utils import clahe_equalized, get_ensemble
    from .segmentation.utils.model_definition import FR_UNet


class Parser(ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def plot_masks(images, masks):
    """Plot images and masks side by side.

    Args:
        images (list, str): Path(s) to image(s) or image(s) themselves (np.ndarray, PIL.Image, torch.Tensor)
        masks (list, np.ndarray): Mask(s)

    Returns:
        None
    """
    images = Img(images).to_batch().to_numpy().img
    masks = [masks] if len(np.array(masks).shape) == 2 else masks

    if len(images) != len(masks):
        raise ValueError("Number of images and masks must be the same.")

    fig, ax = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))
    if len(images) == 1:
        ax = np.expand_dims(ax, axis=0)
    for i in range(len(images)):
        ax[i, 0].imshow(images[i])
        ax[i, 1].imshow(masks[i], cmap="gray")
    plt.show()


def save_masks(x_paths, masks, path):
    """Save masks to a pickle file.

    Args:
        x_paths (list, str): Path(s) to image(s)
        masks (list, np.ndarray): Mask(s)
        path (str): Directorry to save the masks in

    Returns:
        None
    """
    if isinstance(x_paths, str):
        x_paths = [x_paths]
        masks = [masks]
    masks_flattened = [mask.flatten() for mask in masks]
    basenames = [os.path.basename(x) for x in x_paths]
    df = pd.DataFrame(
        {
            "image": basenames,
            "mask": masks_flattened,
            "shape": [mask.shape for mask in masks],
        }
    )
    p = os.path.join(path, "masks.pkl")
    os.makedirs(os.path.dirname(p), exist_ok=True)

    if os.path.exists(p):
        df_ = pickle.load(open(p, "rb"))
        df_ = df_[~df_["image"].isin(df["image"])]
        df = pd.concat([df_, df], axis=0)

    pickle.dump(df, open(p, "wb"))


def load_ensemble(device: str = "cuda:0"):
    models_paths = [
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".pth")
    ]
    ensemble_models = get_ensemble(models_paths, dropout=False, device=device)
    return ensemble_models


def predict(
    ensemble_models: list,
    images: Union[
        List[str],
        str,
        List[Image.Image],
        Image.Image,
        List[np.ndarray],
        np.ndarray,
        List[torch.Tensor],
        torch.Tensor,
    ],
    device: str = None,
    size: Tuple[int, int] = (512, 512),
    save_to: str = None,
    plot=False,
    threshold=0.5,
):
    """Predict vessel mask for a single image or a batch of images.

    Args:
        ensemble_models (list): List of models to ensemble.
        images (list, str, Image.Image, np.ndarray, torch.Tensor): List of paths to images, or a
            single path or a batch of image objects or a single image.
        device (str): Device to run inference on. If None, the device the model is on is used.
        size (tuple): Size of the output masks.
        save_to (str): Path to save the masks, optional.
        plot (bool): Whether to plot the masks.
        threshold (float): Threshold for binarization.

    Returns:
        np.ndarray: Predicted mask(s)
    """
    if device is None:
        device = str(next(ensemble_models[0].parameters()).device)

    seed_everything(23, silent=True)

    images = Img(images).to_batch().img
    images = [
        Img(image).to_tensor(from_cspace="rgb", to_cspace="bgr").to_numpy().img
        for image in images
    ]

    pred_masks = [[] for _ in range(len(ensemble_models))]

    for j, model in enumerate(ensemble_models):
        model = model.to(device)

        processed_images = []
        original_sizes = []
        for i, x in enumerate(images):
            height, width = x.shape[:2]
            original_sizes.append((width, height))

            if size[0] is None:
                size = (int(size[1] * width / height), size[1])
            elif size[1] is None:
                size = (size[0], int(size[0] * height / width))

            x = clahe_equalized(x)  # BGR > RGB
            x = cv2.resize(x, size)
            x = Img(x).to_tensor().img.to(device)

            processed_images.append(x)

        processed_images = torch.stack(processed_images)  # (n, c, h, w)

        with torch.no_grad():
            pred_logit = model(processed_images)
            pred = torch.sigmoid(pred_logit)  # (n, 1, h, w)
            pred_mask = np.array(pred.cpu() > threshold, dtype=int)
            pred_mask = pred_mask.squeeze()  # (n, h, w) or (h, w)
            if len(pred_mask.shape) == 2:
                pred_mask = np.expand_dims(pred_mask, axis=0)  # (n=1, h, w)

            pred_masks[j].append(pred_mask)

    masks = np.mean(pred_masks, axis=0)  # (1, n, h, w)
    masks = masks.squeeze()  # (n, h, w)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)  # (n=1, h, w)

    # Strech mask to input image size
    resized_masks = []
    for i, m in enumerate(masks):
        m = cv2.resize(m, original_sizes[i])
        resized_masks.append(m)
    masks = np.array(resized_masks)

    if plot:
        plot_masks(images, masks)

    if save_to is not None:
        save_masks(images, masks, save_to)

    if len(masks) == 1:
        return masks[0]

    return masks


def load_masks_from_filenames(filenames, masks_dir: str = None):
    """Load masks from a pickle file.

    Args:
        filenames (list, str): Path(s) to image(s)
        masks_dir (str): Directory to load the masks from

    Returns:
        list or ndarray: Mask(s)
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    p = os.path.join(masks_dir, "masks.pkl")
    masks = pickle.load(open(p, "rb"))
    out = []

    for filename in filenames:
        row = masks.loc[masks["image"] == os.path.basename(filename)]
        mask = row["mask"].values[0]
        mask = np.array(mask)
        mask = mask.reshape(row["shape"].values[0])
        out.append(mask)

    if len(out) == 1:
        return out[0]
    return out


if __name__ == "__main__":
    parser = Parser()
    parser.add_argument(
        "--paths",
        nargs="+",
        type=str,
        default=None,
        help="Paths to images to be segmented.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run inference on."
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Size of the output masks."
    )
    parser.add_argument(
        "--save_to", type=str, default=None, help="Path to save the masks, optional."
    )
    parser.add_argument(
        "--plot", type=bool, default=False, help="Whether to plot the masks."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for binarization."
    )
    args = parser.parse_args()

    ensemble_models = load_ensemble(args.device)
    predict(
        ensemble_models,
        args.paths,
        args.device,
        args.size,
        args.save_to,
        args.plot,
        args.threshold,
    )
