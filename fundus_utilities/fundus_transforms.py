# Adapted from Huang et al., 2023
# https://github.com/YijinHuang/pytorch-classification/blob/master/data/transforms.py

from typing import Any, Dict
from torchvision import transforms

# ImageNet pixel mean and std for each channel
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 512, split: str = "train", normalize=True, mean=MEAN, std=STD):
    """Data transformations for fundus images, inspired by Huang et al. (2023).

    Args:
        img_size (int): image size (width -or- height)
        split (str): train, val, validation, or test

    Returns:
        callable transforms.Compose() object
    """

    if split != "train":
        transformations = [
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
        ]

        if not normalize:
            return transforms.Compose(transformations)
        return transforms.Compose(transformations+get_normalization(mean,std))

    aug_args: Dict[str, Any] = {
        "random_crop": {"scale": [0.85, 1.15], "ratio": [0.75, 1.25], "p": 1.0},
        "flip": {"p": 0.5},
        "color_jitter": {
            "p": 0.5,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0,
            "hue": 0,
        },
        "rotation": {"p": 1.0, "degrees": [-180, 180]},
    }

    operations = {
        "random_crop": random_apply(
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=aug_args["random_crop"]["scale"],
                ratio=aug_args["random_crop"]["ratio"],
                antialias=True,
            ),
            p=aug_args["random_crop"]["p"],
        ),
        "horizontal_flip": transforms.RandomHorizontalFlip(p=aug_args["flip"]["p"]),
        "vertical_flip": transforms.RandomVerticalFlip(p=aug_args["flip"]["p"]),
        "color_jitter": random_apply(
            transforms.ColorJitter(
                brightness=aug_args["color_jitter"]["brightness"],
                contrast=aug_args["color_jitter"]["contrast"],
                saturation=aug_args["color_jitter"]["saturation"],
                hue=aug_args["color_jitter"]["hue"],
            ),
            p=aug_args["color_jitter"]["p"],
        ),
        "rotation": random_apply(
            transforms.RandomRotation(degrees=aug_args["rotation"]["degrees"], fill=0),
            p=aug_args["rotation"]["p"],
        ),
    }

    augmentations = [transforms.ToTensor()]
    for operation in operations.items():
        augmentations.append(operation[1])

    if not normalize:
        return transforms.Compose(augmentations)
    return transforms.Compose(augmentations+get_normalization(mean,std))


def random_apply(op, p):
    """Randomly apply a transformation"""
    return transforms.RandomApply([op], p=p)


def get_normalization(mean=MEAN, std=STD):
    """Add normalization transformation using dataset wide pixel mean and std.
    
    Args:
        mean: list of mean values for each channel. Defaults to ImageNet mean.
        std: list of std values for each channel. Defaults to ImageNet std.
    
    Returns:
        list: transforms.Normalize wrapped by a list"""

    normalization = transforms.Normalize(
        mean, std
    )
    return [normalization]

def get_unnormalization(mean=MEAN, std=STD):
    """Add unnormalization transformation using original dataset wide pixel mean and std.
    
    Args:
        mean: list of mean values for each channel. Defaults to ImageNet mean.
        std: list of std values for each channel. Defaults to ImageNet std.
    
    Returns:
        transforms.Normalize: unnormalization transformation
    """
    new_mean = [-m/s for m,s in zip(mean,std)]
    new_std = [1/s for s in std]

    return get_normalization(new_mean, new_std)[0]
