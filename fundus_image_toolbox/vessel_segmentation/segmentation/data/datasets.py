from PIL import Image
import torch
import numpy as np
from glob import glob
import cv2
from ..utils.notebook_utils import clahe_equalized
from torch.utils.data import Dataset # or non-generic Dataset class?

class DRIVEDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        
        self.images_path = sorted(glob(data_path + "/test_data/*"))
        self.masks_path = sorted(glob(data_path + "/test_label/*"))
        self.n_samples = len(self.images_path)

    def __getitem__(self, index):

        data = Image.open(self.images_path[index]).resize((512,512), resample=Image.Resampling.NEAREST)
        label = Image.open(self.masks_path[index]).resize((512,512), resample=Image.Resampling.NEAREST)

        data = np.array(data)
        label = np.array(label)

        if data.shape[-1]==3:
            data = torch.from_numpy(np.array(data).transpose(2, 0, 1)).float() / 255
            label = torch.from_numpy(np.array(label)).float().unsqueeze(0) / 255
        else:
            data = torch.from_numpy(data).unsqueeze(0).float() / 255
            label = torch.from_numpy(label).float().unsqueeze(0) / 255

        return data, label
    
    def __len__(self):
        return self.n_samples

drive_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/DRIVE'
drive_dataset = DRIVEDataset(drive_path)

drive_dataset.__len__()


class CHASEDBDataset(Dataset):

    def __init__(self, data_path):
        self.images_path = sorted(glob(data_path + "/test_data/*"))
        self.masks_path = sorted(glob(data_path + "/test_label/*"))

        self.transforms = None
        self.size = 512
        self.n_samples = len(self.images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = clahe_equalized(image)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image = image / 255.0  # (512, 512, 3) Normalizing to range (0,1)
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        try:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
            # print(self.masks_path[index])
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(str(e))
            print(self.masks_path[index])
        
        mask = mask / 255.0  # (512, 512)
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
    

chase_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/CHASE_DB1'
chase_dataset = CHASEDBDataset(chase_path)

chase_dataset.__len__()