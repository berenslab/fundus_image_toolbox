import os
from pathlib import Path
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from fundus_image_toolbox.utilities import (
    multilevel_3way_split as split3,
    seed_everything,
)

from .transforms import get_transforms

# Data root. In there, folders "bad" and "good" with subfolders "drimdb" and "deepdrid-isbi2020"
# must exist.
from .default import DATA_ROOT

SPLIT_SEED = 12345


class FundusQualityDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        config,
        DATA_ROOT=DATA_ROOT,
        split="train",
        augment=True,
        normalize=True,
        verbose=True,
        drimdb_dir=None,
        deepdrid_dir=None,
    ):
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config

        self.data_root = DATA_ROOT
        self.split = split
        self.augment = augment
        seed_everything(self.config.seed, silent=True)
        self.verbose = verbose
        self.normalize = normalize
        self.data_dirs = {
            "drimdb": Path(drimdb_dir),
            "deepdrid-isbi2020": Path(deepdrid_dir),
        }

        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_root / self.data.iloc[idx]["image"]
        image = Image.open(img_name)
        label = self.data.iloc[idx]["label"]
        label = torch.tensor(label)
        origin = self.data.iloc[idx]["dataset"]

        if "drimdb" in origin:
            image = self._preprocess_drimdb(image)

        image = self._transform(image)

        return image.to(self.config.device), label.to(self.config.device)

    def _load_data(self) -> torch.utils.data.Dataset:
        datasets = (
            ["drimdb", "deepdrid-isbi2020"]
            if self.config.use_datasets == "all"
            else self.config.use_datasets
        )

        self._transfer_data()

        data = []
        for quality in ["bad", "good"]:
            for dataset in datasets:
                image_dir = self.data_root / quality / dataset
                for img_name in image_dir.iterdir():
                    if dataset == "deepdrid-isbi2020":
                        patient = img_name.split("_")[0]
                    else:
                        patient = len(data) - 10000
                    data.append(
                        [Path(quality) / dataset / img_name, quality, dataset, patient]
                    )
        df = pd.DataFrame(data, columns=["image", "quality", "dataset", "patient"])
        df["label"] = [0.0 if x == "bad" else 1.0 for x in df["quality"]]

        # Split the data individually by origin of dataset to yield resproducible splits independent of use_datasets
        train_df, val_df, test_df = (
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
        )
        for dataset in datasets:
            _train_df, _val_df, _test_df = split3(
                df[df["dataset"] == dataset].reset_index(),
                [0.6, 0.2, 0.2],
                SPLIT_SEED,
                stratify_by="label",
                split_by="patient",
            )
            train_df = pd.concat([train_df, _train_df])
            val_df = pd.concat([val_df, _val_df])
            test_df = pd.concat([test_df, _test_df])

        if self.verbose:
            # Print label proportions, origins
            df = eval(self.split + "_df").copy(deep=True)
            print(f"{self.split} split:")
            print(df["label"].value_counts())
            print(df["dataset"].value_counts())
            print()
            del df

        if self.split == "train":
            return train_df
        elif self.split == "val":
            return val_df
        elif self.split == "test":
            return test_df

    def _transfer_data(self):
        """If DATA_ROOT does not yet contain the data, transfer it from the passed locations."""
        # Verify that desired folder structure exists
        for dataset in self.data_dirs:
            if not (self.data_root / "good" / dataset).exists():
                (self.data_root / "good" / dataset).mkdir(parents=True)
            if not (self.data_root / "bad" / dataset).exists():
                (self.data_root / "bad" / dataset).mkdir(parents=True)

        # Check if data is already present
        n_files = sum(
            [
                len(files)
                for r, d, files in os.walk(self.data_root)
                if any(f.endswith(".jpg") for f in files) and "iwebalbumfiles" not in r
            ]
        )
        if np.isclose(n_files, 2216, atol=50):
            return

        print("Copying data...")

        # Transfer DrimDB data
        drimdb_dir = self.data_dirs["drimdb"]
        for d in ["Bad", "Good", "Outlier"]:
            assert (
                Path(drimdb_dir) / d
            ).exists(), f"Directory {(Path(drimdb_dir) / d)} not found. Check your path to DrimDB data."
        for d in ["Bad", "Outlier"]:
            for f in (Path(drimdb_dir) / d).iterdir():
                os.system(
                    f"cp '{Path(drimdb_dir) / d / f}' '{self.data_root / 'bad' / 'drimdb' / f.name}'"
                )
        for f in (Path(drimdb_dir) / "Good").iterdir():
            os.system(
                f"cp '{Path(drimdb_dir) / 'Good' / f}' '{self.data_root / 'good' / 'drimdb' / f.name}'"
            )

        # Transfer DeepDrid data
        deepdrid_dir = self.data_dirs["deepdrid-isbi2020"]
        assert (
            Path(deepdrid_dir) / "regular_fundus_images"
        ).exists(), f"Directory {(Path(deepdrid_dir) / 'regular_fundus_images')} not found. Check your path to DeepDrid data."
        self._get_deepdrid()

        print("Done.")

    def _get_deepdrid(self, resize=True):
        def _resize_and_save(df, target_dirs, size=512):
            for i, row in df.iterrows():
                img = Image.open(row["image_path"])
                img = img.resize((size, size))
                label = row["Overall quality"]  # 1: good, 0: bad
                target_dir = target_dirs[label]
                suffix = ".jpg"
                img.save(target_dir / f"{row['image_id']}{suffix}")

        def _copy(df, target_dirs):
            for i, row in df.iterrows():
                img_path = row["image_path"]
                label = row["Overall quality"]
                # target_path = os.path.join(target_dirs[label], row["image_id"]+".jpg")
                target_path = target_dirs[label] / f"{row['image_id']}.jpg"
                os.system(f"cp '{img_path}' '{target_path}'")

        fundus_dir = Path(self.data_dirs["deepdrid-isbi2020"]) / "regular_fundus_images"
        train_data = fundus_dir / "regular-fundus-training"
        valid_data = fundus_dir / "regular-fundus-validation"
        test_data = fundus_dir / "Online-Challenge1&2-Evaluation"

        train_df = pd.read_csv(train_data / "regular-fundus-training.csv")
        val_df = pd.read_csv(valid_data / "regular-fundus-validation.csv")
        test_df = pd.read_excel(test_data / "Challenge2_labels.xlsx")

        train_df["image_path"] = train_df["image_id"].apply(
            lambda x: train_data / "Images" / x.split("_")[0] / f"{x}.jpg"
        )
        val_df["image_path"] = val_df["image_id"].apply(
            lambda x: valid_data / "Images" / x.split("_")[0] / f"{x}.jpg"
        )
        test_df["image_path"] = test_df["image_id"].apply(
            lambda x: test_data / "Images" / x.split("_")[0] / f"{x}.jpg"
        )

        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        target_dir_good = self.data_root / "good" / "deepdrid-isbi2020"
        target_dir_bad = self.data_root / "bad" / "deepdrid-isbi2020"

        target_dirs = {1: target_dir_good, 0: target_dir_bad}

        if resize:
            _resize_and_save(df, target_dirs)
        else:
            _copy(df, target_dirs)

    def _transform(self, image):
        split = self.split if self.augment else "test"
        return get_transforms(
            img_size=self.config.img_size, split=split, normalize=self.normalize
        )(image)

    def _preprocess_drimdb(self, image: Image.Image):
        """Preprocess DRIMDB images before applying transforms"""

        # 1. : Cut off the top 5% of height to remove the text
        cut = int(image.height * 0.05)
        image = image.crop((0, cut, image.width, image.height))

        # 2. : Resize to square by adding black padding at top and bottom
        size = max(image.size)
        new_image = Image.new("RGB", (size, size), (0, 0, 0))
        new_image.paste(image, (0, (size - image.size[1]) // 2))
        out = new_image

        return out
