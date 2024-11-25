# Adopted from https://github.com/ufoym/imbalanced-dataset-sampler/
"""
MIT License

Copyright (c) 2018 Ming

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset.
    An imbalanced dataset sampler for oversampling low frequent classes and undersampling high
    frequent ones. Does not alter length of dataset.


    Args:
        dataset: torch dataset
        method: "balanced". Not yet implemented: "oversampling" or "undersampling"
        labels: a list of labels to balance by
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index

    Returns:
        Iterable: indices to sample from

    Example:
        ```
        if config.balance_datasets:
            # Balance by original dataset origin (DRIMDB, DeepDRID-ISBI2020)
            sampler = ImbalancedDatasetSampler(self.train_dataset, method="balanced", labels = self.train_dataset.data["dataset"])
            shuffle = False
        else:
            sampler = None
            shuffle = True
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler)
        ```
    """

    def __init__(
        self,
        dataset,
        method: str = "balanced",
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        self.method = method

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df["indices"] = self.indices
        self.df = df.sort_index()

        self.label_count = df["label"].value_counts()
        label_count_sorted = self.label_count.sort_values()

        self.minority_class = label_count_sorted.keys()[0]
        self.majority_class = label_count_sorted.keys()[-1]

        self.df_minority = df[df["label"] == self.minority_class]
        self.df_majority = df[df["label"] == self.majority_class]

        if method == "balanced":
            self.num_samples = len(self.indices) if num_samples is None else num_samples
        else:
            raise NotImplementedError

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        if self.method == "balanced":
            weights = 1.0 / self.label_count[self.df["label"]]
            weights = torch.DoubleTensor(weights.to_list())
            return (
                self.indices[i]
                for i in torch.multinomial(weights, self.num_samples, replacement=True)
            )

        else:
            raise NotImplementedError

    def __len__(self):
        return self.num_samples
