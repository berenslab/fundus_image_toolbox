from types import SimpleNamespace
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

from .transforms_multi import (
    RandomZoom,
    ResizedCenterCropAndPad,
    RandomSqueeze,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomTranslation,
    RandomRotation,
    ImageAdjustment,
    ToTensor,
)
from .default import DEFAULT_CSV_PATH

SPLIT_SEED = 12345


class ODFoveaDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, split="train", device="cuda:0", data_root=".."
    ):
        self.df = df
        self.transformation = self._get_transformation(split)
        self.device = device
        self.data_root = data_root

    def __getitem__(self, index):
        image_path = self.df.iloc[index]["image_path"]
        # image = Image.open(os.path.join(self.data_root,image_path))
        image = Image.open(self.data_root / image_path)
        fovea_label = self.df.iloc[index][["fovea_x", "fovea_y"]].values.astype(float)
        od_label = self.df.iloc[index][["od_x", "od_y"]].values.astype(float)
        label = (fovea_label, od_label)
        image, label = self.transformation((image, label))
        fovea_label, od_label = label
        return (
            image.to(self.device),
            fovea_label.to(self.device),
            od_label.to(self.device),
        )

    def __len__(self):
        return len(self.df)

    def _get_transformation(self, split):
        if split == "train":
            return Compose(
                [
                    RandomSqueeze(),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomTranslation(),
                    RandomRotation(),
                    ResizedCenterCropAndPad(),
                    RandomZoom(),
                    ImageAdjustment(),
                    ToTensor(),
                ]
            )
        elif split in ["val", "test"]:
            return Compose([ResizedCenterCropAndPad(), ToTensor()])
        else:
            raise ValueError("Split should be either 'train' or 'test'")


class ODFoveaLoader:
    def __init__(self, config: SimpleNamespace):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        if config.csv_path is None:
            config.csv_path = DEFAULT_CSV_PATH
        if config.data_root is None:
            config.data_root = "../"

        self.df = self.get_df(config.csv_path)

        # Split into 20 percent validation, 20 percent test
        dev_df, self.test_df = train_test_split(
            self.df, test_size=0.2, shuffle=True, random_state=SPLIT_SEED
        )
        self.train_df, self.val_df = train_test_split(
            dev_df, test_size=0.25, shuffle=True, random_state=SPLIT_SEED
        )

        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        self.train_dataset = ODFoveaDataset(
            self.train_df,
            split="train",
            device=config.device,
            data_root=config.data_root,
        )
        self.val_dataset = ODFoveaDataset(
            self.val_df, split="val", device=config.device, data_root=config.data_root
        )
        self.test_dataset = ODFoveaDataset(
            self.test_df, split="test", device=config.device, data_root=config.data_root
        )

        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Val dataset: {len(self.val_dataset)}")
        print(f"Test dataset: {len(self.test_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=False
        )

    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_df(self, path):
        df = pd.read_csv(path).reset_index(drop=True)
        return df
