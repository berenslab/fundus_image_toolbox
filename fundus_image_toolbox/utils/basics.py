import os
from typing import List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exist(paths: Union[str, list]):
    """Check if a path or multiple paths exist."""
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            return False
    return True

def exists (paths: Union[str, list]):
    """Check if a path or multiple paths exist."""
    return exist(paths)

def flatten_one(l):
    """Flatten one level of nesting"""
    return [item for sublist in l for item in sublist]

def parse_list(arg: Union[list, str]):
    """Parse a list-like structures as a list."""
    if isinstance(arg, list):
        return arg
    elif (
        isinstance(arg, np.ndarray)
        or isinstance(arg, tuple)
        or isinstance(arg, set)
        or isinstance(arg, pd.Series)
    ):
        return list(arg)
    else:
        try:
            arg = str(arg)
            arg = arg.replace(" ", "").replace("[", "").replace("]", "")
            if "," in arg:
                return arg.split(",")
            else:
                # not a list
                return arg
        except Exception as e:
            raise f"Could not parse {arg} of type {type(arg)} as a list: " + str(e)
        
def on_slurm_job():
    """Check if the code is running on a SLURM job."""
    if "SLURM_JOB_ID" in os.environ:
        return True
    else:
        return False
    
def _show_single(img: np.ndarray):
    img2d = np.squeeze(img)
    if img2d.ndim == 2:
        plt.imshow(img2d, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")

def show(img: Union[np.ndarray, List[np.ndarray]]):
    plt.figure()
    if isinstance(img, list):
        valid_imgs = [
            f for f in img
            if isinstance(f, np.ndarray) and f.size > 0
        ]
        if not valid_imgs:
            raise ValueError("show() received no valid images to display.")
        for i, f in enumerate(valid_imgs):
            plt.subplot(1, len(valid_imgs), i + 1)
            _show_single(f)
    else:
        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("show() received an invalid or empty image.")
        _show_single(img)
    plt.show()

def print_type(img: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(img, list):
        for i, f in enumerate(img):
            print_type(f)
    else:
        print(f"Type: {type(img)}")
        if isinstance(img, np.ndarray):
            print(f"Shape: {img.shape}")
            print(f"Type: {img.dtype}")
            print()