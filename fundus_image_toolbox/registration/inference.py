# %%
from typing import List, Union
import os
import requests
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import yaml
import pandas as pd
import pickle
from fundus_image_toolbox.utilities import ImageTorchUtils as Img

from .SuperRetina import (
    pre_processing,
    simple_nms,
    remove_borders,
    sample_keypoint_desc,
    SuperRetina,
)

WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "SuperRetina/save/SuperRetina.pth"
)
IMG_SIZE = 512
DEFAULT_CONFIG = {
    "device": "cuda:0",
    "use_matching_trick": True,
    "nms_size": 1,
    "nms_thresh": 0.0005,
    "knn_thresh": 0.85,
}


def wget(link, target):
    # platform independent wget alternative
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
    }
    r = requests.get(link, headers=headers, stream=True)
    downloaded_file = open(target, "wb")

    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            downloaded_file.write(chunk)


def download_weights(url="https://zenodo.org/records/11241985/files/SuperRetina.pth"):
    os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
    print("Downloading weights...")
    wget(url, WEIGHT_PATH)
    print("Done")


def get_mask(filename, masks: pd.DataFrame = None, show=True, binarize=True):
    directory = os.path.dirname(filename)
    if masks is None:
        p = os.path.join(directory, "masks.pkl")
        masks = pickle.load(open(p, "rb"))

    row = masks.loc[masks["image"] == os.path.basename(filename)]
    mask = row["mask"].values[0]
    size = row["size"].values[0]

    if binarize:
        thresh = 0.2
        mask = [1 if x > thresh else 0 for x in mask]

    mask = np.array(mask).reshape(size, size)

    if show:
        plt.imshow(mask)
        plt.show()
        plt.close()

    assert np.sum(mask) > 0

    return mask


# %%
def any_to_image_array(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # To BGR
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # To BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def any_to_greyscale(image):
    image = any_to_image_array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Cast to Uint8
    image = image.astype(np.uint8)
    return image


# %%
def enhance(image):
    image = any_to_greyscale(image)
    black_mask = image < 10

    # Replace black pixels with mean of the image
    image[black_mask] = image[~black_mask].mean()

    # Normalize
    image = (image - image.min()) / (image.max() - image.min())

    # Enhance images using clahe
    image = pre_processing(image)
    image = (image * 255).astype(np.uint8)

    # Add in the mask again
    image[black_mask == 1] = 0

    return image


# %%
def transform(image: Image, model_image_height, model_image_width):
    compose = transforms.Compose(
        [
            # Resize the image to the model input size at smaller edge length
            transforms.Resize(model_image_height),
            # Center crop the image
            transforms.CenterCrop((model_image_height, model_image_width)),
            transforms.ToTensor(),
        ]
    )

    return compose(image)


# %%
def get_config(config=Union[dict, str, None]):
    """Get the configuration for inference. If config is None, use the default config. Adds the
    path to the model weights, if needed, as well as the image size for the model
    (Default: 512x512). The latter is independent of the image output size, so you can leave it at
    the default.

    Args:
        config (dict or str, optional): The configuration. Defaults to None. Options:
            - None
            - str: path to yaml file
            - dict: configuration dictionary

    Returns:
        dict: The configuration dictionary
    """
    if config is None:
        config = DEFAULT_CONFIG
    elif isinstance(config, str):
        if os.path.exists(config):
            with open(config) as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError("Config file not found.")
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError("Config must be either a path to a yaml file or a dictionary")

    if "PREDICT" in config:
        config = config["PREDICT"]

    if not hasattr(config, "model_save_path"):
        config["model_save_path"] = WEIGHT_PATH
    if not hasattr(config, "model_image_height"):
        config["model_image_height"] = IMG_SIZE
    if not hasattr(config, "model_image_width"):
        config["model_image_width"] = IMG_SIZE

    return config


# %%
def get_device(config: dict):
    device = (
        torch.device(config["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return device


# %%
def load_model(config: dict = None):
    """Load the SuperRetina model and the knn matcher for matching keypoints. Returns the model and the matcher.

    Args:
        config (dict): The configuration dictionary
    """
    if config is None:
        config = DEFAULT_CONFIG

    config = get_config(config)

    device = get_device(config)

    if not os.path.isfile(config["model_save_path"]):
        download_weights()

    model = SuperRetina().to(device)
    model.load_state_dict(
        torch.load(config["model_save_path"], map_location=device)["net"]
    )
    model.eval()

    knn_matcher = cv2.BFMatcher(cv2.NORM_L2)

    return model, knn_matcher


# %%
def predict_points(fixed_tensor, moving_tensor, config, model):
    inputs = torch.cat((moving_tensor.unsqueeze(0), fixed_tensor.unsqueeze(0)))
    inputs = inputs.to(get_device(config))

    with torch.no_grad():
        detector_pred, descriptor_pred = model(inputs)

    scores = simple_nms(detector_pred, config["nms_size"])

    b, _, h, w = detector_pred.shape
    scores = scores.reshape(-1, h, w)

    keypoints = [torch.nonzero(s > config["nms_thresh"]) for s in scores]

    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(
        zip(*[remove_borders(k, s, 4, h, w) for k, s in zip(keypoints, scores)])
    )

    keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

    descriptors = [
        sample_keypoint_desc(k[None], d[None], 8)[0].cpu().data
        for k, d in zip(keypoints, descriptor_pred)
    ]
    keypoints = [k.cpu() for k in keypoints]

    return keypoints, descriptors


# %%
def get_match_visualization(imageA, imageB, kpsA, kpsB, matches, status):
    # Initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    if len(imageA.shape) == 2:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2RGB)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2RGB)

    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # Loop over the matches
    for i, ((match, _), s) in enumerate(zip(matches, status)):
        trainIdx, queryIdx = match.trainIdx, match.queryIdx
        # Only process the match if the keypoint was successfully matched
        if s == 1:
            # Draw the match
            ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
            ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
            # Each line gets a color
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
            cv2.line(vis, ptA, ptB, color, 2)

    return vis


# %%
def match(matcher, fixed_desc, moving_desc, config):
    """Match the keypoints using the knn matcher"""
    goodMatch = []
    status = []
    try:
        matches = matcher.knnMatch(moving_desc, fixed_desc, k=2)
        for m, n in matches:
            if m.distance < config["knn_thresh"] * n.distance:
                goodMatch.append(m)
                status.append(True)
            else:
                status.append(False)
    except Exception:
        pass

    return goodMatch, status, matches


def check_collapse(
    goodMatch, cv_kpts_fixed, cv_kpts_moving, round_to: int = 1, threshold: float = 0.65
):
    """Filter out matches where keypoints in source are projected to one point in target (and vice versa)"""
    # Get keypoints as list of tuples
    moving_keypoints = [cv_kpts_moving[m.queryIdx].pt for m in goodMatch]
    fixed_keypoints = [cv_kpts_fixed[m.trainIdx].pt for m in goodMatch]
    moving_keypoints = [
        (round(kp[0], round_to), round(kp[1], 2)) for kp in moving_keypoints
    ]
    fixed_keypoints = [
        (round(kp[0], round_to), round(kp[1], 2)) for kp in fixed_keypoints
    ]

    # Get unique keypoints
    unique_moving = len(list(set(moving_keypoints)))
    unique_fixed = len(list(set(fixed_keypoints)))
    ratio_moving = unique_moving / len(moving_keypoints)
    ratio_fixed = unique_fixed / len(fixed_keypoints)
    if ratio_moving < threshold or ratio_fixed < threshold:
        raise Exception(
            "Failed to align the two images: Too many points are projected to one!"
        )


# %%
def find_homography_and_align(
    fixed_image,
    moving_image,
    original_moving_image,
    cv_kpts_fixed,
    cv_kpts_moving,
    goodMatch,
    matches,
    status,
    show=False,
    show_mapping=False,
    save_to=None,
):
    # Check if too many points are projected to one
    check_collapse(goodMatch, cv_kpts_fixed, cv_kpts_moving)

    H_m = None
    inliers_num_rate = 0
    good = goodMatch.copy()
    if len(goodMatch) >= 4:
        src_pts = [cv_kpts_moving[m.queryIdx].pt for m in good]
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = [cv_kpts_fixed[m.trainIdx].pt for m in good]
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

        H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        good = np.array(good)[mask.ravel() == 1]
        status = np.array(status)
        temp = status[status == True]
        temp[mask.ravel() == 0] = False
        status[status == True] = temp
        inliers_num_rate = mask.sum() / len(mask.ravel())

    if show_mapping:
        query_np = np.array([kp.pt for kp in cv_kpts_moving])
        refer_np = np.array([kp.pt for kp in cv_kpts_fixed])
        refer_np[:, 0] += moving_image.shape[1]
        matched_image = get_match_visualization(
            moving_image, fixed_image, cv_kpts_moving, cv_kpts_fixed, matches, status
        )
        plt.figure(dpi=300)
        plt.scatter(query_np[:, 0], query_np[:, 1], s=1, c="r")
        plt.scatter(refer_np[:, 0], refer_np[:, 1], s=1, c="r")
        plt.axis("off")
        plt.title("Match Result, #goodMatch: {}".format(len(good)))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.close()

        print("The rate of inliers: {:.3f}%".format(inliers_num_rate * 100))

    # Get dims of fixed image
    image_height, image_width = fixed_image.shape[:2]

    if H_m is not None:
        h, w = image_height, image_width
        original_moving_align = cv2.warpPerspective(
            original_moving_image,
            H_m,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        moving_align = original_moving_align.copy()

        merged = np.zeros((h, w, 3), dtype=np.uint8)

        if len(moving_align.shape) == 3:
            moving_align = enhance(moving_align)
            original_moving_align = cv2.cvtColor(
                original_moving_align, cv2.COLOR_BGR2RGB
            )

        if len(fixed_image.shape) == 3:
            fixed_gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
        else:
            fixed_gray = fixed_image
        merged[:, :, 0] = moving_align
        merged[:, :, 1] = fixed_gray

    else:
        raise Exception("Failed to align the two images!")

    def fig_grid():
        # Create 1x4 grid and return the axes
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for a in ax:
            a.axis("off")
        ax[0].imshow(fixed_gray, "gray")
        ax[0].set_title("fixed")
        ax[1].imshow(moving_image, "gray")
        ax[1].set_title("moving")
        ax[2].imshow(merged)
        ax[2].set_title("merged")
        ax[3].imshow(original_moving_align)
        ax[3].set_title("out")

        return fig, ax

    if H_m is not None and show:
        fig, ax = fig_grid()
        plt.show()

    if save_to is not None and H_m is not None:
        fig, ax = fig_grid()
        plt.savefig(save_to)
        plt.close()

    return original_moving_align


# %%
def register_one(
    fixed_image: Union[str, np.ndarray, Image.Image],
    moving_image: Union[str, np.ndarray, Image.Image],
    show: bool = True,
    show_mapping: bool = False,
    save_to: str = None,
    config: dict = DEFAULT_CONFIG,
    model=None,
    matcher=None,
):
    """Register two images using SuperRetina and homography transformation.

    Args:
        fixed_image (Union[str, np.array, Image.Image]): The fixed image.
        moving_image (Union[str, np.array, Image.Image]): The moving image to register to the fixed.
        show (bool, optional): Show the images. Defaults to True.
        show_mapping (bool, optional): Show the mapping of SuperPoints. Defaults to False.
        save_to (str, optional): Save the images to this path. Defaults to None.
        config (dict, optional): The configuration dict for inference. Defaults to DEFAULT_CONFIG.
        model (optional): The SuperRetina model. Defaults to None.
        matcher (optional): The matcher for matching keypoints. Defaults to None.

    Returns:
        [np.array]: The aligned moving image.
    """

    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        if isinstance(fixed_image, str) and isinstance(moving_image, str):
            save_to = os.path.join(
                save_to,
                os.path.basename(fixed_image).__str__()
                + "_"
                + os.path.basename(moving_image).__str__(),
            )
        else:
            file_count = len(os.listdir(save_to))
            save_to = os.path.join(save_to, "aligned_" + str(file_count))

    original_moving_image = any_to_image_array(moving_image)

    fixed_image = enhance(fixed_image)
    moving_image = enhance(moving_image)

    config = get_config(config)

    fixed_tensor = transform(
        Image.fromarray(fixed_image),
        config["model_image_height"],
        config["model_image_width"],
    )
    moving_tensor = transform(
        Image.fromarray(moving_image),
        config["model_image_height"],
        config["model_image_width"],
    )

    image_height, image_width = fixed_image.shape

    if model is None or matcher is None:
        model, matcher = load_model(config)

    keypoints, descriptors = predict_points(fixed_tensor, moving_tensor, config, model)

    fixed_keypoints, moving_keypoints = keypoints[1], keypoints[0]
    fixed_desc, moving_desc = (
        descriptors[1].permute(1, 0).numpy(),
        descriptors[0].permute(1, 0).numpy(),
    )

    # Mapping keypoints to scaled keypoints
    cv_kpts_moving = [
        cv2.KeyPoint(
            int(i[0] / config["model_image_width"] * image_width),
            int(i[1] / config["model_image_height"] * image_height),
            30,
        )  # 30 is keypoints size, which can be ignored
        for i in moving_keypoints
    ]
    cv_kpts_fixed = [
        cv2.KeyPoint(
            int(i[0] / config["model_image_width"] * image_width),
            int(i[1] / config["model_image_height"] * image_height),
            30,
        )
        for i in fixed_keypoints
    ]

    goodMatch, status, matches = match(matcher, fixed_desc, moving_desc, config)

    moving_aligned = find_homography_and_align(
        fixed_image,
        moving_image,
        original_moving_image,
        cv_kpts_fixed,
        cv_kpts_moving,
        goodMatch,
        matches,
        status,
        show=show,
        show_mapping=show_mapping,
        save_to=save_to,
    )

    return moving_aligned


def register(
    fixed_image: Union[
        str,
        List[str],
        List[np.ndarray],
        List[torch.Tensor],
        List[Image.Image],
        np.ndarray,
        torch.Tensor,
        Image.Image,
    ],
    moving_image: Union[
        str,
        List[str],
        List[np.ndarray],
        List[torch.Tensor],
        List[Image.Image],
        np.ndarray,
        torch.Tensor,
        Image.Image,
    ],
    show: bool = True,
    show_mapping: bool = False,
    save_to: str = None,
    config: dict = DEFAULT_CONFIG,
    model=None,
    matcher=None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Register a pair or pairs of images from batches using SuperRetina and homography transformation.

    Args:
        fixed_image (Union[torch.Tensor, list, str, np.array, Image.Image]): The fixed image(s).
        moving_image (Union[torch.Tensor, list, str, np.array, Image.Image]): The moving image(s) to register to the fixed.
        show (bool, optional): Show the images. Defaults to True.
        show_mapping (bool, optional): Show the mapping of SuperPoints. Defaults to False.
        save_to (str, optional): Save the images to this path. Defaults to None.
        config (dict, optional): The configuration dict for inference. Defaults to DEFAULT_CONFIG.
        model (optional): The SuperRetina model. Defaults to None.
        matcher (optional): The matcher for matching keypoints. Defaults to None.

    Returns:
        list of np.ndarrays or np.ndarray: The aligned moving image(s).
    """

    # To batch of numpy images
    fixed_image = Img(fixed_image).to_tensor().squeeze().to_batch().to_numpy().img
    moving_image = Img(moving_image).to_tensor().squeeze().to_batch().to_numpy().img

    if len(fixed_image) != len(moving_image):
        raise ValueError("The number of fixed and moving images must be the same.")

    moving_aligned = []
    for fixed, moving in zip(fixed_image, moving_image):
        moving_aligned.append(
            register_one(
                fixed, moving, show, show_mapping, save_to, config, model, matcher
            )
        )

    if len(moving_aligned) == 1:
        return moving_aligned[0]
    return moving_aligned


if __name__ == "__main__":
    fixed_image = "./example1.jpg"
    moving_image = "./example2.jpg"

    moving_aligned = register(fixed_image, moving_image, show=True)
    plt.imshow(moving_aligned, "gray")
