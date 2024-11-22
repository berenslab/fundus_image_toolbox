from typing import List
from tqdm import tqdm
import torch

def get_pixel_mean_and_sd(loaders:List[torch.utils.data.DataLoader], device:str = "cuda:0"):
    """Compute the mean and standard deviation of a dataset.
    
    Args:
        loaders (list): list of torch data loaders; Use no augmentations here!
        device (str): torch device to use
    
    Returns:
        list: mean of the dataset for each channel
        list: standard deviation of the dataset for each channel
    """
    
    cnt = 0
    fst_moment = torch.empty(3).to(device)
    snd_moment = torch.empty(3).to(device)
    for loader in loaders:
        for (data, label) in tqdm(loader):
            b, c, h, w = data.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2,
                                    dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean, std