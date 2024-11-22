import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from .dataset import FundusQualityDataset
from fundus_image_toolbox.utilities import ImbalancedDatasetSampler
from .default import DATA_ROOT


class FundusQualityLoader:
    def __init__(
        self,
        config,
        root_dir=DATA_ROOT,
        augment=True,
        normalize=True,
        verbose=True,
        drimdb_dir=None,
        deepdrid_dir=None,
    ):
        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        self.root_dir = root_dir

        self.train_dataset = FundusQualityDataset(
            config,
            root_dir,
            split="train",
            augment=augment,
            normalize=normalize,
            verbose=verbose,
            drimdb_dir=drimdb_dir,
            deepdrid_dir=deepdrid_dir,
        )
        self.val_dataset = FundusQualityDataset(
            config,
            root_dir,
            split="val",
            augment=augment,
            normalize=normalize,
            verbose=verbose,
            drimdb_dir=drimdb_dir,
            deepdrid_dir=deepdrid_dir,
        )
        self.test_dataset = FundusQualityDataset(
            config,
            root_dir,
            split="test",
            augment=augment,
            normalize=normalize,
            verbose=verbose,
            drimdb_dir=drimdb_dir,
            deepdrid_dir=deepdrid_dir,
        )

        if config.balance_datasets:
            # Balance by original dataset origin: areds, registration_outs, drimdb, deepdrid
            shuffle = False
            sampler = ImbalancedDatasetSampler(
                self.train_dataset,
                method="balanced",
                labels=self.train_dataset.data["dataset"],
            )
        else:
            shuffle = False
            sampler = None

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=False
        )

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
