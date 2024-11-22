import os


def set_seed(seed: int = 12345, silent=False) -> None:
    """Set seed for reproducibility."""
    try:
        import numpy as np

        np.random.seed(seed)
        if not silent:
            print(f"Numpy random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for numpy: Numpy not found.")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not silent:
            print(f"Torch random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for torch: Torch not found.")

    try:
        import random

        random.seed(seed)
        if not silent:
            print(f"Random seed was set as {seed}")
    except ImportError:
        if not silent:
            print("Could not set seed for random: Random not found.")
        else:
            print()

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not silent:
        print(f"PYTHONHASHSEED was set as {seed}")
        print()
