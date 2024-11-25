import os
import time
import yaml
import datetime
import numpy as np
from typing import List, Union
import torch
from torch.utils.data import DataLoader
import PIL
from torchvision.transforms.functional import to_tensor

from .model import FundusQualityModel, download_weights
from .default import MODELS_DIR, ENSEMBLE_MODELS
from pathlib import Path


def is_datetime_format(s):
    try:
        datetime.datetime.strptime(s, "%Y-%m-%d %H-%M-%S")
        return True
    except ValueError:
        return False


def any_to_tensor(image):
    """Converts an image to a torch tensor of shape (C, H, W)

    Args:
        image (str, np.ndarray, torch.Tensor): Image path, numpy array or torch tensor.

    Returns:
        torch.Tensor: Image as a torch tensor.
    """
    if isinstance(image, str):
        image = to_tensor(
            PIL.Image.open(image)
        )  # PIL Image with shape (H, W, C) -> (C, H, W)
    elif isinstance(image, np.ndarray):
        image = to_tensor(
            PIL.Image.fromarray(image)
        )  # numpy array with shape (H, W, C) -> (C, H, W)
    elif isinstance(image, torch.Tensor):
        image = image
    else:
        raise ValueError(
            "Image should be a string, numpy array or torch tensor but is of type",
            type(image),
        )
    return image


def get_ensemble(models_dir: str = MODELS_DIR, device: str = "cpu"):
    """Load the 10-model ensemble from the specified models or project directory.

    Args:
        device (str, optional): Device to use. Defaults to "cpu".
        models_dir (str, optional): Directory containing the "models" folder or its parent.
            Defaults to "./quality_prediction/models".

    Returns:
        ensemble: List of FundusQualityModel objects
    """
    models_dir = Path(models_dir)
    if "models" not in models_dir.parts:
        models_dir = models_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for e in ENSEMBLE_MODELS:
        if e not in [p.name for p in models_dir.iterdir() if p.is_dir()]:
            print("At least one model was not found.")
            download_weights()

    model_dirs = [
        p for p in models_dir.iterdir() if p.is_dir() and is_datetime_format(p.name)
    ]
    assert len(model_dirs) == len(
        ENSEMBLE_MODELS
    ), f"Expected 10 models, got {len(model_dirs)}. Did you download the models and place them into {models_dir}?"

    # Get configs and load models
    configs = []
    for p in model_dirs:
        with open(p / "config.yaml") as c:
            configs.append(yaml.safe_load(c))
            configs[-1]["device"] = device

    ensemble = []
    for ckpt, conf in zip(model_dirs, configs):
        model = FundusQualityModel(conf)
        model.load_checkpoint(str(ckpt))
        ensemble.append(model)

    return ensemble


def ensemble_predict(
    ensemble: List[FundusQualityModel],
    image: Union[list, str, torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    print_result: bool = False,
):
    """Predicts the quality of an image or image batch (0: ungradable, 1: gradable)
        using an ensemble of models.

    Args:
        ensemble (list): List of FundusQualityModel objects.
        image (list, str, np.ndarray, torch.tensor): Image path(s) as a List[str] or an image as
            tensor or np.ndarray or a batch of images as a tensor.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        ensemble_pred: Ensemble prediction(s) (confidence score between 0 and 1)
        binary_ensemble_pred: Ensemble predicted class(es), where 1 is good quality"""
    preds = []

    if isinstance(image, list) and isinstance(image[0], str):
        image = [any_to_tensor(i) for i in image]

    if (
        isinstance(image, (torch.Tensor, np.ndarray, list))
        and isinstance(image[0], (torch.Tensor, np.ndarray, PIL.Image.Image))
        and len(image[0].shape) == 3
    ):
        if not isinstance(image[0], torch.Tensor):
            image = [any_to_tensor(i) for i in image]
        # Do batch prediction
        n = len(image)
        preds = {i: [] for i in range(n)}
        for model in ensemble:
            pred = model.predict_from_batch(image, threshold=None, load_best=False)
            for i in range(n):
                preds[i].append(pred[i])
        for i in range(n):
            preds[i] = np.array(preds[i])
        ensemble_pred = np.array([preds[i].mean() for i in range(n)])
        binary_ensemble_pred = np.where(ensemble_pred >= threshold, 1, 0)

    elif isinstance(image, (list, str)) or isinstance(
        image, (torch.Tensor, np.ndarray)
    ):
        # Do single image prediction
        preds = []
        for model in ensemble:
            preds.append(
                model.predict_from_image(image, threshold=None, load_best=False)
            )
        preds = np.array(preds)

        ensemble_pred = preds.mean()
        binary_ensemble_pred = 1 if ensemble_pred >= threshold else 0

        if print_result:
            print(
                f"Ensemble confidence score: {ensemble_pred:.4f} \nEnsemble predicted class: {binary_ensemble_pred}, where 1 is good quality)"
            )

    else:
        raise ValueError(
            "Image(s) must be a list of paths, a path, or a tensor of the image or of a batch of images."
        )

    # Squeeze
    if isinstance(ensemble_pred, (list, np.ndarray)) and len(ensemble_pred) == 1:
        ensemble_pred = ensemble_pred[0]
    if (
        isinstance(binary_ensemble_pred, (list, np.ndarray))
        and len(binary_ensemble_pred) == 1
    ):
        binary_ensemble_pred = binary_ensemble_pred[0]

    return ensemble_pred, binary_ensemble_pred


def ensemble_predict_from_dataloader(
    ensemble: List[FundusQualityModel],
    dataloader: DataLoader,
    threshold: float = 0.5,
    print_result: bool = True,
):
    """Predicts the quality of images in a dataloader (0: ungradable, 1: gradable) using an ensemble of models.

    Args:
        ensemble (list): List of FundusQualityModel objects.
        dataloader (DataLoader): DataLoader containing the images.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        print_result (bool, optional): Whether to print the result. Defaults to True.

    Returns:
        ensemble_preds: Ensemble prediction logits
        binary_ensemble_preds: Ensemble predicted classes, where 1 is good quality
        labels: True labels
    """
    labels = []
    preds = []
    for model in ensemble:
        pred, labels = model.predict_from_dataloader(dataloader, load_best=False)
        preds.append(pred)
    preds = np.array(preds)
    labels = np.array(labels)

    ensemble_preds = preds.mean(axis=0)
    binary_ensemble_preds = np.where(ensemble_preds >= threshold, 1, 0)

    assert (
        len(binary_ensemble_preds) == len(labels) == len(ensemble_preds)
    ), "Prediction and label lengths do not match."

    if print_result:
        print(
            f"Ensemble predicted logits: {ensemble_preds} \nEnsemble predicted classes: {binary_ensemble_preds}, where 1 is good quality)"
        )

    return ensemble_preds, binary_ensemble_preds, labels
