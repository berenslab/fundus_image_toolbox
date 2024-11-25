import numpy as np
import torch
import os
import cv2
from typing import Union
from fundus_image_toolbox.utilities import ImageTorchUtils as Img
from PIL import Image

# Adapted from https://github.com/HzFu/EyeQ/blob/master/EyeQ_preprocess/fundus_prep.py


def imread(file_path, c=None):
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)

    if im is None:
        raise "Can not read image"

    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imwrite(file_path, image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)


def fold_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_mask_BZ(img):
    """Creates mask of fundus. The function does not explicitly create a circular mask. However,
    it does perform a series of operations that could result in a roughly circular mask if the
    region of interest in the input image is approximately circular and centrally located.

    How:
    1. The function first converts the input image to grayscale if it's not already, and then applies a binary threshold to create an initial mask. This separates the image into regions of interest (white) and background (black) based on pixel intensity.
    2. The function then creates a new mask that is the inverse of the original mask and performs flood fill operations from the corners. This helps to remove small artifacts or noise near the borders of the image.
    3. The original mask and the flood-filled mask are combined to create a new mask that preserves the regions of interest while removing noise.
    4. The function then applies morphological operations (erosion and dilation) using a rectangular structuring element. These operations help to smooth the mask and remove small regions or noise.

    The erosion operation shrinks the white regions (regions of interest), which can help to round off any irregularities. The dilation operation then expands the white regions back to their original size, but the expansion is uniform in all directions, which can help to maintain the rounded shape.

    So, while the function does not explicitly create a circular mask, the combination of thresholding, flood fill, and morphological operations can result in a roughly circular mask if the region of interest in the input image is approximately circular and centrally located.
    """
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threshold = np.mean(gray_img) / 3 - 5
    _, mask = cv2.threshold(gray_img, max(0, threshold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    new_mask = (1 - mask).astype(np.uint8)
    _, new_mask, _, _ = cv2.floodFill(
        new_mask, nn_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY
    )
    _, new_mask, _, _ = cv2.floodFill(
        new_mask,
        nn_mask,
        (new_mask.shape[1] - 1, new_mask.shape[0] - 1),
        (0),
        cv2.FLOODFILL_MASK_ONLY,
    )
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    return mask


def _get_center_by_edge(mask):
    """Calculates the center coordinates of a circular region based on the edges of the mask. It uses the row and column sums of the mask to find the approximate center coordinates."""
    center = [0, 0]
    x = mask.sum(axis=1)
    center[0] = np.where(x > x.max() * 0.95)[0].mean()
    x = mask.sum(axis=0)
    center[1] = np.where(x > x.max() * 0.95)[0].mean()
    return center


def _get_radius_by_mask_center(mask, center):
    """Calculates the radius based on a given mask and center coordinates.
    In summary, this function takes a mask and center coordinates, performs morphological operations on the mask, calculates the distances of non-zero elements from the center, and determines the radius based on the distribution of distances.
    """
    mask = mask.astype(np.uint8)
    ksize = max(mask.shape[1] // 400 * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    index = np.where(mask > 0)
    d_int = np.sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
    b_count = np.bincount(np.ceil(d_int).astype(int))
    radius = np.where(b_count > b_count.max() * 0.995)[0].max()
    return radius


def _get_circle_by_center_bbox(shape, center, bbox, radius):
    "Gets mask from center and radius"
    center_mask = np.zeros(shape=shape).astype("uint8")
    center_tmp = (int(center[0]), int(center[1]))
    # Draw circle
    center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)
    return center_mask


def get_mask(img):
    if img.ndim == 3:
        # raise 'image dim is not 3'
        g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img = img.copy()
    else:
        raise "image dim is not 1 or 3"
    h, w = g_img.shape
    shape = g_img.shape[0:2]
    g_img = cv2.resize(g_img, (0, 0), fx=0.5, fy=0.5)
    tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask = get_mask_BZ(tg_img)
    center = _get_center_by_edge(tmp_mask)
    radius = _get_radius_by_mask_center(tmp_mask, center)

    # Resize back
    center = [center[0] * 2, center[1] * 2]
    radius = int(radius * 2)
    s_h = max(0, int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h - s_h, 2 * radius), min(w - s_w, 2 * radius))
    tmp_mask = _get_circle_by_center_bbox(shape, center, bbox, radius)
    return tmp_mask, bbox, center, radius


def get_center_radius(mask):
    center = _get_center_by_edge(mask)
    radius = _get_radius_by_mask_center(mask, center)

    return center, radius


def center_image(img, mask, center=None, radius=None):
    """Centers the circle in the image and resize st. circle edges touch image edges"""
    if center is None or radius is None:
        center, radius = get_center_radius(mask)
    h, w = img.shape[:2]

    # Scale image
    scale = w / (2 * radius)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)

    # Get center of image
    center_img = (h / 2, w / 2)

    # Get new center of circle
    center_circle = (center[0] * scale, center[1] * scale)

    # Get transform
    transform = np.array(center_img) - np.array(center_circle)

    # Apply transform
    M = np.float32([[1, 0, transform[1]], [0, 1, transform[0]]])
    img = cv2.warpAffine(img, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))

    return img, mask


def mask_image(img, mask):
    img[mask <= 0, ...] = 0
    return img


def remove_back_area(img, bbox=None, border=None):
    image = img
    if border is None:
        border = np.array(
            (
                bbox[0],
                bbox[0] + bbox[2],
                bbox[1],
                bbox[1] + bbox[3],
                img.shape[0],
                img.shape[1],
            ),
            dtype=int,
        )
    image = image[border[0] : border[1], border[2] : border[3], ...]
    return image, border


def supplemental_black_area(img, border=None):
    image = img
    if border is None:
        h, v = img.shape[0:2]
        max_l = max(h, v)
        if image.ndim > 2:
            image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
        else:
            image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
        border = (
            int(max_l / 2 - h / 2),
            int(max_l / 2 - h / 2) + h,
            int(max_l / 2 - v / 2),
            int(max_l / 2 - v / 2) + v,
            max_l,
        )
    else:
        max_l = border[4]
        if image.ndim > 2:
            image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
        else:
            image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
    image[border[0] : border[1], border[2] : border[3], ...] = img
    return image, border


def process_without_gb(img):
    borders = []
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img, bbox=bbox)
    mask, _ = remove_back_area(mask, border=r_border)
    borders.append(r_border)
    r_img, sup_border = supplemental_black_area(r_img)
    mask, _ = supplemental_black_area(mask, border=sup_border)
    borders.append(sup_border)
    return r_img, borders, (mask * 255).astype(np.uint8), center, radius


def process_path(image_path, save_path=None, size=None, suffix=".png"):
    """load img from path and circle crop"""
    dst_image = os.path.splitext(image_path.split("/")[-1])[0] + suffix
    dst_path = os.path.join(save_path, dst_image)

    img = imread(image_path)

    img, mask, center, radius = process_img(img, size=size)

    if save_path:
        imwrite(dst_path, img)

    return img, mask, center, radius


def process_img(img, size=None):
    try:
        img, borders, mask, center, radius = process_without_gb(img)

        if size:
            img = cv2.resize(img, size)
            mask = cv2.resize(mask, size)

    except:
        print("failed")

    img, mask = center_image(img, mask)
    center, radius = get_center_radius(mask)

    return img, mask, center, radius


def crop(
    img: Union[str, Image.Image, np.ndarray, torch.Tensor, list],
    size=(512, 512),
    return_all=False,
    to_numpy=True,
):
    """Fit a circle to the image (or images in a batch), center it and crop off the background.

    Args:
        img: Image path(s), RGB numpy array(s) or RGB torch tensor(s).
        size: Size of the output images. Default is (512, 512).
        return_all: If True, return the cropped images, masks, centers, and radii. Default is False.
        to_numpy: If True, return numpy arrays, else torch tensors. Default is True.

    Returns:
        Cropped images as numpy arrays or torch tensors or all the outputs if return_all is True.
    """
    input_is_a_batch = Img(img).is_batch_like()

    # Process images, batches of images, and paths to numpy batch
    img_batch = (
        Img(img).to_tensor(silent=True).set_channel_dim(-1).to_batch().to_numpy().img
    )

    # Process
    imgs, masks, centers, radii = [], [], [], []
    for img in img_batch:
        img, mask, center, radius = process_img(img, size=size)
        img = Img(img).to_numpy().img
        imgs.append(img)
        masks.append(mask)
        centers.append(center)
        radii.append(radius)

    imgs, masks, centers, radii = (
        np.array(imgs),
        np.array(masks),
        np.array(centers),
        np.array(radii),
    )

    # Convert to torch tensors
    if not to_numpy:
        imgs = Img(imgs).to_batch().img
        masks = Img(masks).to_batch().img
        centers = torch.tensor(centers)
        radii = torch.tensor(radii)

    # Return single element if input was single element
    if not input_is_a_batch:
        imgs, masks, centers, radii = imgs[0], masks[0], centers[0], radii[0]

    if return_all:
        return imgs, masks, centers, radii

    return imgs
