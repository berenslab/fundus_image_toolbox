from types import SimpleNamespace
from typing import Union
import yaml
from pathlib import Path
import requests

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torchvision import models
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import os
from PIL import ImageDraw, Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    auc,
    roc_curve,
    precision_recall_curve,
)
from fundus_image_toolbox.utilities import ImageTorchUtils as Img
from fundus_image_toolbox.utilities import seed_everything

from .transforms import get_unnormalization, get_transforms
from .default import ENSEMBLE_MODELS, MODELS_DIR


def wget(link, target):
    # platform independent wget alternative
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
    }
    r = requests.get(link, headers=headers, stream=True)
    downloaded_file = open(target, "wb")

    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            downloaded_file.write(chunk)


def download_weights(url="https://zenodo.org/records/11174749/files/weights.tar.gz"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading weights...")
    target = (MODELS_DIR / "weights.tar.gz").__str__()
    wget(url, target)
    print("Extracting weights...")
    os.system(f'tar -xzf {MODELS_DIR / "weights.tar.gz"} -C {MODELS_DIR}')
    print("Removing tar file...")
    os.remove(target)
    print("Done")


def plot_quality(fundus, conf: float, label: int, threshold: float = 0.5):
    """Plots the fundus image with a colorbar indicating the confidence score.

    Args:
        fundus (str, np.ndarray): Fundus image path or image as numpy array
        conf (float): Confidence score
        label (int): Label (0: ungradable, 1: gradable)
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        None
    """
    # Image to array with (H, W, C) shape
    fundus = Img(fundus).to_tensor().squeeze().to_numpy().set_channel_dim(-1).img

    if Img(fundus).is_batch_like():
        raise ValueError("Pass a single image, not a batch.")

    fig = plt.figure(figsize=(4.3, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[25, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Plot the fundus image
    ax1.imshow(fundus)
    ax1.axis("off")
    ax1.text(
        0,
        20,
        f'{"Ungradeable" if label == 0 else "Gradeable"}',
        fontsize=12,
        color="white",
        backgroundcolor="black",
    )

    # Create a colorbar
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation="vertical")

    # Draw a horizontal line at the threshold with legend
    ax2.axhline(y=threshold, color="k", linestyle="-")

    # Draw an X at the confidence score
    ax2.scatter(0.5, conf, color="black", marker="x", s=50)

    plt.show()


class FundusQualityModel:
    """Fundus Quality Model with one output logit for binary classification"""

    def __init__(self, config):
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config
        self.model = self._get_model(self.config.model_type)
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.model_selection_metric = (
            self.config.model_selection_metric
            if hasattr(self.config, "model_selection_metric")
            else "accuracy"
        )

        self.was_trained = False
        seed_everything(self.config.seed, silent=True)

        metrics = ["acc", "auroc", "auprc"]
        self.performance = {"train": {}, "val": {}, "test": {}}
        for metric in metrics:
            self.performance["train"][metric] = np.nan
            self.performance["val"][metric] = np.nan
            self.performance["test"][metric] = np.nan

        self.model.to(self.config.device)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.checkpoint_path = (
            MODELS_DIR / f"{self.timestamp}/{self.config.model_type}_best.pt"
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def load_checkpoint(self, dir: str):
        dir = Path(dir)
        if dir.suffix == ".pt":
            dir = dir.parent
        if dir.name.endswith("/"):
            dir = dir.parent
        if str(dir) in ENSEMBLE_MODELS and not dir.is_dir():
            download_weights()
        self.timestamp = dir.name
        self.checkpoint_path = dir / f"{self.config.model_type}_best.pt"

        self.model.load_state_dict(
            torch.load(
                self.checkpoint_path, map_location=self.config.device, weights_only=True
            )
        )
        self.model.to(self.config.device)

        print(f"Model loaded from {self.checkpoint_path.parent.name}")

    def train(self, train_dataloader, val_dataloader):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.loss_tracker = {"train": [], "val": []}
        self.acc_tracker = {"train": [], "val": []}

        for epoch in range(self.config.epochs):
            pbar = tqdm(total=len(train_dataloader))
            pbar.set_description(f"Epoch: {epoch+1} Training...")

            train_loss, train_acc = self._train_val_step(
                train_dataloader, self.loss_fn, self.optimizer, pbar
            )
            self.loss_tracker["train"].append(train_loss)
            self.acc_tracker["train"].append(train_acc)

            pbar = tqdm(total=len(val_dataloader))
            pbar.set_description(f"Epoch: {epoch+1} Validating...")

            val_loss, val_acc = self._train_val_step(
                val_dataloader, self.loss_fn, pbar=pbar
            )
            self.loss_tracker["val"].append(val_loss)
            self.acc_tracker["val"].append(val_acc)

            pbar.set_description(
                f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                if "loss" not in self.model_selection_metric:
                    torch.save(self.model.state_dict(), self.checkpoint_path)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                if "loss" in self.model_selection_metric:
                    torch.save(self.model.state_dict(), self.checkpoint_path)

        pbar.close()

        self.was_trained = True

        print(
            f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss}, Best acc: {self.best_acc}\n"
        )
        # Save summary file and loss plot
        self._plot_loss(save=True)

    def _train_val_step(self, dataloader, loss_fn, optimizer=None, pbar=None):
        if optimizer is not None:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_acc = 0.0

        for image_batch, label_batch in dataloader:
            pbar.update(1)

            image_batch, label_batch = image_batch.to(
                self.config.device
            ), label_batch.to(self.config.device)
            label_batch = label_batch.unsqueeze(1).float()

            with torch.set_grad_enabled(optimizer is not None):
                output = self.model(image_batch)
                loss = loss_fn(output, label_batch)
                running_loss += loss.item()

            with torch.no_grad():
                preds = torch.sigmoid(output)
                preds = torch.where(preds > 0.5, 1, 0)
                running_acc += torch.mean((preds == label_batch).float()).item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

        return epoch_loss, epoch_acc

    def predict_from_dataloader(
        self, dataloader, threshold=None, device=None, load_best=True, numpy_cpu=True
    ):
        if device is not None:
            self.config.device = device

        if load_best:
            self.load_checkpoint(self.checkpoint_path)

        self.model.eval()

        all_preds = []
        all_labels = []

        for image_batch, label_batch in dataloader:
            image_batch, label_batch = image_batch.to(
                self.config.device
            ), label_batch.to(self.config.device)

            with torch.no_grad():
                output = self.model(image_batch)
                preds = torch.sigmoid(output)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_batch.cpu().numpy())

            all_preds = torch.tensor(np.array(all_preds))
            all_labels = torch.tensor(all_labels).squeeze()
            assert all_preds.shape[0] == all_labels.shape[0]

            if threshold:
                all_preds = torch.where(all_preds >= threshold, 1, 0).float().squeeze()
            else:
                all_preds = all_preds.squeeze()

            if numpy_cpu:
                all_preds = all_preds.cpu().numpy()
                all_labels = all_labels.cpu().numpy()

            assert all_preds.shape[0] == all_labels.shape[0]

        return all_preds, all_labels

    def predict_from_image(
        self,
        image: Union[str, torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        device: str = None,
        load_best: bool = True,
        numpy_cpu: bool = True,
    ):
        """Predicts from a single image and returns the prediction (logit or class) as a numpy array.

        Args:
            image (str, torch.Tensor or numpy.ndarray): Image to predict from
            threshold (float, optional): Threshold to convert logit to class. Defaults to 0.5.
                classes: 0 if logit < threshold "ungradable", 1 if logit >= threshold "gradable"
            device (str, optional): Device to use. Defaults to None.
            load_best (bool, optional): Load best checkpoint. Defaults to True.
            numpy_cpu (bool, optional): Return numpy array in cpu. Defaults to True.

        Returns:
            numpy.ndarray: Prediction
        """
        image = Img(image).to_tensor().img
        image = get_transforms(split="test")(to_pil_image(image)).unsqueeze(0)

        return self.predict_from_batch(
            image, threshold, device, load_best, numpy_cpu, transform=False
        )

    def predict_from_batch(
        self,
        image_batch: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        device: str = None,
        load_best: bool = True,
        numpy_cpu: bool = True,
        transform=True,
    ):
        """Predicts from a batch of images and returns the predictions (logits or classes) as a numpy array.

        Args:
            image_batch (torch.Tensor or numpy.ndarray): Batch of images to predict from
            threshold (float, optional): Threshold to convert logits to classes. Defaults to 0.5.
                classes: 0 if logit < threshold "ungradable", 1 if logit >= threshold "gradable"
            device (str, optional): Device to use. If None, model's device is used. Defaults to None.
            load_best (bool, optional): Load best checkpoint. Defaults to True.
            numpy_cpu (bool, optional): Return numpy array in cpu. Defaults to True.

        Returns:
            numpy.ndarray: Predictions
        """
        image_batch = Img(image_batch).to_batch().img
        if transform:
            image_batch = [
                get_transforms(split="test")(to_pil_image(image))
                for image in image_batch
            ]
            image_batch = torch.stack(image_batch)

        if device is not None:
            self.config.device = device

        if load_best:
            self.load_checkpoint(self.checkpoint_path)

        self.model.eval()

        image_batch = image_batch.to(self.config.device)

        with torch.no_grad():
            output = self.model(image_batch)
            preds = torch.sigmoid(output)

            if threshold:
                preds = torch.where(preds >= threshold, 1, 0).float()  # .squeeze()
            else:
                preds = preds  # .squeeze()

            if numpy_cpu:
                preds = preds.cpu().numpy()

        return preds

    def evaluate(
        self,
        test_dataloader=None,
        val_dataloader=None,
        train_dataloader=None,
        threshold=0.5,
        best=True,
        plot_auc=False,
    ):
        if (
            sum(
                1
                for _ in filter(
                    None.__ne__, [test_dataloader, val_dataloader, train_dataloader]
                )
            )
            != 1
        ):
            raise ValueError("You should pass exactly one dataloader.")

        if val_dataloader:
            loader = "val"
            dataloader = val_dataloader
        elif test_dataloader:
            loader = "test"
            dataloader = test_dataloader
        elif train_dataloader:
            loader = "train"
            dataloader = train_dataloader

        all_preds, all_labels = self.predict_from_dataloader(
            dataloader, threshold=threshold, load_best=best, numpy_cpu=False
        )

        binary_preds = torch.where(all_preds > threshold, 1, 0).float().squeeze()

        acc = torch.mean((binary_preds == all_labels).float()).item()
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        auroc = auc(fpr, tpr)
        precision_, recall_, _ = precision_recall_curve(all_labels, all_preds)
        auprc = auc(recall_, precision_)

        if plot_auc:
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"Ensemble (AUROC = {auroc:.2f})")
            ax.plot(recall_, precision_, label=f"Ensemble (AUPRC = {auprc:.2f})")
            ax.legend()
            plt.show()

        self.performance[loader]["acc"] = acc
        self.performance[loader]["auroc"] = auroc
        self.performance[loader]["auprc"] = auprc

        return self.performance[loader]

    def save_summary(self, log=None):
        summary_path = (
            MODELS_DIR / f"{self.timestamp}/{self.config.model_type}_summary.txt"
        )
        config_path = MODELS_DIR / f"{self.timestamp}/config.yaml"

        if not config_path.is_file():
            with open(config_path, "w") as c:
                yaml.dump(vars(self.config), c)

        with open(summary_path, "a") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "\n")
            if log is not None:
                if isinstance(log, str):
                    f.write(log + "\n")
                elif isinstance(log, dict):
                    for key, value in log.items():
                        f.write(f"{key}: {value}\n")
                else:
                    print("Cannot log what was passed of type", type(log))

            if self.was_trained:
                for key, value in vars(self.config).items():
                    f.write(f"{key}: {value}\n")

                f.write(f"\nBest epoch: {self.best_epoch}\n")
                f.write(f"Best val loss: {self.best_loss}\n")
                f.write(f"Best val acc: {self.best_acc}\n")

            for loader in ["train", "val", "test"]:
                if not np.isnan(self.performance[loader]["acc"]):
                    f.write(f"{loader} performance:\n")
                    for metric in ["acc", "auroc", "auprc"]:
                        f.write(f"{metric}: {self.performance[loader][metric]:.4f}\n")
                    f.write("\n")

    def _plot_loss(self, save=False):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for key in self.loss_tracker.keys():
            ax[0].plot(self.loss_tracker[key], label=key)
            ax[1].plot(self.acc_tracker[key], label=key)
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].set_title("Accuracy")
        ax[1].legend()

        if save:
            plt.savefig(
                MODELS_DIR / f"{self.timestamp}/{self.config.model_type}_loss_plot.png"
            )

        plt.show()

    def plot_grid(self, dataloader, size=10, cols=6):
        self.model.eval()
        samples = np.random.choice(len(dataloader.dataset), size, replace=False)
        rows = np.ceil(len(samples) / cols).astype(int)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axs = axs.flatten()

        size = len(axs) if size < len(axs) else size

        for i, idx in enumerate(samples):
            if i < size:
                image, label = dataloader.dataset[idx]
                image = image.unsqueeze(0).to(self.config.device)
                output = torch.sigmoid(self.model(image)).item()
                output = 0 if output < 0.5 else 1
                correct = output == label

                # Unnormalize
                image = get_unnormalization()(image.squeeze().cpu())
                image = image.permute(1, 2, 0)

                # Clip
                image = np.clip(image, 0, 1).numpy()

                # Rectangle
                col = "green" if label else "red"
                image = np.array(image * 255, dtype=np.uint8)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                img_size = image.size
                draw.rectangle([0, 0, img_size[0], img_size[1]], outline=col, width=10)
                image = np.array(image)

                axs[i].imshow(image)

                if not correct:
                    axs[i].set_title(f"wrong", color="red")

                axs[i].axis("off")

            else:
                axs[i].axis("off")
        plt.show()

    def _get_model(self, type, n_outs=1):
        if type == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = torch.nn.Linear(512, n_outs)
        elif type == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1")
            model.fc = torch.nn.Linear(512, n_outs)
        elif type == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, n_outs)
        elif type == "resnet101":
            model = models.resnet101(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, n_outs)
        elif type == "resnet152":
            model = models.resnet152(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, n_outs)
        elif type == "efficientnet-b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1280, n_outs)
        elif type == "efficientnet-b1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1280, n_outs)
        elif type == "efficientnet-b2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1408, n_outs)
        elif type == "efficientnet-b3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1536, n_outs)
        elif type == "efficientnet-b4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1792, n_outs)
        elif type == "efficientnet-b5":
            model = models.efficientnet_b5(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2048, n_outs)
        elif type == "efficientnet-b6":
            model = models.efficientnet_b6(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2304, n_outs)
        elif type == "efficientnet-b7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2560, n_outs)

        else:
            raise ValueError("Model type not supported")

        return model
