from datetime import datetime
import os
from types import SimpleNamespace
from typing import List, Union
from pathlib import Path
import requests

import yaml
import numpy as np
from tqdm import tqdm
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import box_iou
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.transforms.functional import to_pil_image
from fundus_image_toolbox.utilities import ImageTorchUtils as Img

from .transforms_multi import ToPILImage
from .default import DEFAULT_MODEL, MODELS_DIR


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


class ODFoveaModel:
    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.device = config.device
        self.model = self._get_model(config.model_type).to(self.device)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H_%M.%S")
        self.checkpoint_path = (
            MODELS_DIR / self.timestamp / f"multi_{self.config.model_type}_best.pt"
        )

        print(f"Initializing {self.config.model_type} on {self.device}")

        self.loss_func = nn.SmoothL1Loss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train(self, train_dataloader, val_dataloader):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not (self.checkpoint_path.parent / "config.yaml").exists():
            yaml.safe_dump(
                vars(self.config),
                open(self.checkpoint_path.parent / "config.yaml", "w"),
            )

        best_epoch = 0
        self.loss_tracking = {"train": [], "val": []}
        self.iou_tracking = {"train": [], "val": []}
        self.dist_tracking = {"train": [], "val": []}
        self.best_loss = float("inf")

        for epoch in range(self.config.epochs):

            pbar = tqdm(total=len(train_dataloader))
            pbar.set_description(f"Epoch: {epoch+1} Training...")

            training_loss, training_iou, training_dist = self._train_val_step(
                train_dataloader, self.model, self.loss_func, self.optimizer, pbar
            )
            self.loss_tracking["train"].append(training_loss)
            self.iou_tracking["train"].append(training_iou)
            self.dist_tracking["train"].append(training_dist)

            pbar.set_description(f"Epoch: {epoch+1} Evaluating...")
            with torch.inference_mode():
                val_loss, val_iou, val_dist = self._train_val_step(
                    val_dataloader, self.model, self.loss_func, None, pbar
                )
                self.loss_tracking["val"].append(val_loss)
                self.iou_tracking["val"].append(val_iou)
                self.dist_tracking["val"].append(val_dist)

                if val_loss < self.best_loss:
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.best_loss = val_loss

            text = f"{epoch+1}/{self.config.epochs} - Train loss: {training_loss:.4}, IoU: {training_iou:.2}, Dist: {training_dist:.4} - Val loss: {val_loss:.4}, IoU: {val_iou:.2}, Dist: {val_dist:.4}"
            pbar.set_description(text)

        pbar.close()

        print(
            f'Best model at epoch {best_epoch} with loss {self.best_loss:.4} and IoU {self.iou_tracking["val"][best_epoch]:.2} and distance {self.dist_tracking["val"][best_epoch]:.4}'
        )
        print(f"Model saved at {self.checkpoint_path}")

        # Save summary to file
        p = self.checkpoint_path.parent
        with open(p / "summary.txt", "w") as f:
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best loss: {self.best_loss:.4}\n")
            f.write(f'Best IoU: {self.iou_tracking["val"][best_epoch]:.2}\n')
            f.write(f'Best distance: {self.dist_tracking["val"][best_epoch]:.4}\n')

        # Save loss, IoU and distance tracking to files
        with open(p / f"multi_{self.config.model_type}_train_loss.txt", "w") as f:
            f.write("\n".join(map(str, self.loss_tracking["train"])))
        with open(p / f"multi_{self.config.model_type}_val_loss.txt", "w") as f:
            f.write("\n".join(map(str, self.loss_tracking["val"])))
        with open(p / f"multi_{self.config.model_type}_train_iou.txt", "w") as f:
            f.write("\n".join(map(str, self.iou_tracking["train"])))
        with open(p / f"multi_{self.config.model_type}_val_iou.txt", "w") as f:
            f.write("\n".join(map(str, self.iou_tracking["val"])))
        with open(p / f"multi_{self.config.model_type}_train_dist.txt", "w") as f:
            f.write("\n".join(map(str, self.dist_tracking["train"])))
        with open(p / f"multi_{self.config.model_type}_val_dist.txt", "w") as f:
            f.write("\n".join(map(str, self.dist_tracking["val"])))

        pbar.close()

    def evaluate(
        self,
        test_dataloader,
        checkpoint_path=None,
        load_checkpoint=True,
        save_summary=True,
    ):
        """Evaluate the model on the test set

        Args:
            test_dataloader (torch.utils.data.DataLoader): Test set dataloader
            checkpoint_path (str, optional): Path to the model checkpoint. If "latest", the latest
                model in "models/" will be used. Defaults to None."
            load_checkpoint (bool, optional): Whether to load the checkpoint. Defaults to True.
            save_summary (bool, optional): Whether to save the summary to a file. Defaults to True.
        """
        if checkpoint_path is not None:
            if checkpoint_path == "latest":
                checkpoint_path = sorted(MODELS_DIR.iterdir())[-1]
                checkpoint_path = (
                    checkpoint_path / f"multi_{self.config.model_type}_best.pt"
                )
            # if not os.path.exists(checkpoint_path):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
            else:
                self.checkpoint_path = checkpoint_path

        if load_checkpoint:
            self.load_checkpoint()

        with torch.inference_mode():
            pbar = tqdm(total=len(test_dataloader))
            test_loss, test_iou, test_dist = self._train_val_step(
                test_dataloader, self.model, self.loss_func, None, pbar
            )
            standardized_test_dist = test_dist * self.config.img_size

            print(
                f"Test loss: {test_loss:.4}, IoU: {test_iou:.2}, Dist: {test_dist:.4}, Standardized dist: {standardized_test_dist:.4}"
            )

        if save_summary:
            with open(f"{self.checkpoint_path.parent}/summary.txt", "a") as f:
                f.write(f"Test loss: {test_loss:.4}\n")
                f.write(f"Test IoU: {test_iou:.2}\n")
                f.write(f"Normalized test distance: {test_dist:.4}\n")
                f.write(
                    f"Pixel distance in {self.config.img_size}px images: {standardized_test_dist:.4}\n"
                )

        return test_loss, test_iou, test_dist, standardized_test_dist

    def predict(
        self,
        images: Union[
            str,
            List[str],
            List[np.ndarray],
            List[torch.Tensor],
            List[Image.Image],
            np.ndarray,
            torch.Tensor,
            Image.Image,
        ],
        cpu_numpy=True,
    ):
        """Predict the fovea and optic disc locations in an image or image batch.

        Args:
            image: Torch image batch of same sized images -or- image paths to same sized images
                -or- image path -or- image (array, tensor or PIL image)

        Returns:
            (List of) Coordinates of the fovea and optic disc in the format (f_x, f_y, od_x, od_y)
        """
        self.model.eval()

        # To Tensor batch of images
        images = Img(images).to_batch().img

        orig_shapes = [(img.shape[1], img.shape[2]) for img in images]

        # Apply test-time augmentation
        transforms = Compose(
            [
                ToTensor(),
                Resize(self.config.img_size, antialias=True),
                CenterCrop(self.config.img_size),
            ]
        )
        images = torch.stack([transforms(to_pil_image(img)) for img in images]).to(
            self.device
        )

        with torch.inference_mode():
            outs = self.model(images)

            # Adjust labels to original image size
            for i in range(len(outs)):
                h, w = orig_shapes[i]
                f_x, f_y = outs[i][0], outs[i][1]
                od_x, od_y = outs[i][2], outs[i][3]
                outs[i] = torch.stack([f_x * w, f_y * h, od_x * w, od_y * h])

            if cpu_numpy:
                outs = [out.cpu().numpy() for out in outs]

            if len(outs) == 1:
                return outs[0]

        return outs

    def load_checkpoint(self):
        print(f"Loading model from {self.checkpoint_path}")
        if not self.checkpoint_path.exists():
            if DEFAULT_MODEL in self.checkpoint_path.__str__():
                print(f"Default model {DEFAULT_MODEL} not found, downloading...")
                if not (MODELS_DIR / DEFAULT_MODEL).exists():
                    self._download_weights()
            else:
                raise FileNotFoundError(f"Checkpoint {self.checkpoint_path} not found")

        self.model.load_state_dict(
            torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
        )

    def _get_model(self, type):
        if type == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(512, 4)
        elif type == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(512, 4)
        elif type == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, 4)
        elif type == "resnet101":
            model = models.resnet101(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, 4)
        elif type == "resnet152":
            model = models.resnet152(weights="IMAGENET1K_V2")
            model.fc = torch.nn.Linear(2048, 4)
        elif type == "efficientnet-b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1280, 4)
        elif type == "efficientnet-b1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1280, 4)
        elif type == "efficientnet-b2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1408, 4)
        elif type == "efficientnet-b3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1536, 4)
        elif type == "efficientnet-b4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(1792, 4)
        elif type == "efficientnet-b5":
            model = models.efficientnet_b5(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2048, 4)
        elif type == "efficientnet-b6":
            model = models.efficientnet_b6(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2304, 4)
        elif type == "efficientnet-b7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(2560, 4)

        else:
            raise ValueError("Model type not supported")

        return model

    def _download_weights(
        self, url="https://zenodo.org/records/11174642/files/weights.tar.gz"
    ):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = (MODELS_DIR / "weights.tar.gz").__str__()
        wget(url, weights_path)
        print("Extracting weights...")
        os.system(f"tar -xzf {weights_path} -C {MODELS_DIR}")

        # Get Windows compatible folder names
        extracted_folders = sorted(MODELS_DIR.iterdir())
        for folder in extracted_folders:
            if folder.is_dir():
                new_folder_name = folder.name.replace(":", "_")
                folder.rename(folder.parent / new_folder_name)

        print("Removing tar file...")
        os.remove(weights_path)
        print("Done")

    def _train_val_step(self, dataloader, model, loss_func, optimizer=None, pbar=None):
        if optimizer is not None:
            model.train()
        else:
            model.eval()

        running_loss = 0
        running_iou = 0
        running_dist = 0

        for image_batch, f_label_batch, od_label_batch in dataloader:
            pbar.update(1)

            output_labels = model(image_batch)
            output_labels = (output_labels[:, :2], output_labels[:, 2:])
            label_batch = (f_label_batch, od_label_batch)

            loss_value, iou_metric_value = self._batch_loss(
                loss_func, output_labels, label_batch, optimizer
            )
            fovea_dist = self._distance_batch(output_labels[0], label_batch[0])
            od_dist = self._distance_batch(output_labels[1], label_batch[1])
            running_dist += (fovea_dist + od_dist) / 2

            running_loss += loss_value
            running_iou += iou_metric_value

        n = len(dataloader.dataset)
        return running_loss / n, running_iou / n, running_dist / n

    def _centroid_to_bbox(self, centroids, w=0.15, h=0.15):
        x0_y0 = centroids - torch.tensor([w / 2, h / 2]).to(self.device)
        x1_y1 = centroids + torch.tensor([w / 2, h / 2]).to(self.device)
        return torch.cat([x0_y0, x1_y1], dim=1)

    def _iou_batch(self, output_labels, target_labels):
        output_bbox = self._centroid_to_bbox(output_labels)
        target_bbox = self._centroid_to_bbox(target_labels)
        return torch.trace(box_iou(output_bbox, target_bbox)).item()

    def _distance_batch(self, output_labels, target_labels):
        # Sum of distances in batch
        dist = torch.sum(
            torch.sqrt(torch.sum((output_labels - target_labels) ** 2, dim=1))
        ).item()
        return dist

    def _batch_loss(self, loss_func, outputs, targets, optimizer=None):
        # Sums of losses and intersection over union/Jaccard index in batch
        loss = loss_func(outputs[0], targets[0])
        loss += loss_func(outputs[1], targets[1])

        with torch.no_grad():
            iou_metric = self._iou_batch(outputs[0], targets[0])
            iou_metric += self._iou_batch(outputs[1], targets[1])
            iou_metric /= 2
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), iou_metric

    def plot_loss(self):
        plt.plot(self.loss_tracking["train"], label="Train loss")
        plt.plot(self.loss_tracking["val"], label="Val loss")
        plt.legend()
        p = self.checkpoint_path.parent / f"multi_{self.config.model_type}_loss.png"
        plt.savefig(p)
        plt.show()
        plt.close()

    def plot_iou(self):
        plt.plot(self.iou_tracking["train"], label="Train IoU")
        plt.plot(self.iou_tracking["val"], label="Val IoU")
        plt.legend()
        p = self.checkpoint_path.parent / f"multi_{self.config.model_type}_iou.png"
        plt.savefig(p)
        plt.show()
        plt.close()

    def plot_dist(self):
        plt.plot(self.dist_tracking["train"], label="Train distance")
        plt.plot(self.dist_tracking["val"], label="Val distance")
        plt.legend()
        p = self.checkpoint_path.parent / f"multi_{self.config.model_type}_dist.png"
        plt.savefig(p)
        plt.show()
        plt.close()

    def _show_image_with_4_bounding_box(
        self, image, labels, target_labels, ax, w_h_bbox=(50, 50), thickness=2
    ):
        f_x, f_y, od_x, od_y = labels
        f_x_target, f_y_target, od_x_target, od_y_target = target_labels

        w, h = w_h_bbox

        image = image.copy()
        ImageDraw.Draw(image).rectangle(
            (
                (f_x_target - w // 2, f_y_target - h // 2),
                (f_x_target + w // 2, f_y_target + h // 2),
            ),
            outline="blue",
            width=thickness,
        )
        ImageDraw.Draw(image).rectangle(
            (
                (od_x_target - w // 2, od_y_target - h // 2),
                (od_x_target + w // 2, od_y_target + h // 2),
            ),
            outline="blue",
            width=thickness,
        )

        ImageDraw.Draw(image).rectangle(
            ((f_x - w // 2, f_y - h // 2), (f_x + w // 2, f_y + h // 2)),
            outline="green",
            width=thickness,
        )
        ImageDraw.Draw(image).rectangle(
            ((od_x - w // 2, od_y - h // 2), (od_x + w // 2, od_y + h // 2)),
            outline="green",
            width=thickness,
        )

        ax.imshow(image)

    def plot_grid(self, dataset, seed=12345):
        print("random choice with seed:", seed)
        print("pred: green")
        print("target: blue")

        self.model.eval()
        rng = np.random.default_rng(seed)  # create Generator object
        n_rows = 4  # number of rows in the image subplot
        n_cols = 8  # # number of cols in the image subplot
        indexes = rng.choice(range(len(dataset)), n_rows * n_cols, replace=False)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        for i, ax in enumerate(axs.flatten()):
            image, f_labels, od_labels = dataset[indexes[i]]
            output_labels = self.model(image.unsqueeze(0))
            output_labels = (output_labels[:, :2], output_labels[:, 2:])

            labels = (f_labels.unsqueeze(0), od_labels.unsqueeze(0))

            iou1 = self._iou_batch(output_labels[0], labels[0])
            iou2 = self._iou_batch(output_labels[1], labels[1])

            iou = (iou1 + iou2) / 2
            _, labels = ToPILImage()((image, (f_labels, od_labels)))

            output_labels = (output_labels[0].squeeze(0), output_labels[1].squeeze(0))
            image, outputs = ToPILImage()((image, output_labels))

            self._show_image_with_4_bounding_box(image, outputs, labels, ax)
            ax.set_title(f"{iou:.2f}")
            ax.axis("off")
        p = self.checkpoint_path.parent / f"multi_{self.config.model_type}_grid.png"
        plt.savefig(p)
        plt.show()
        plt.close()


def plot_input(id, dataset):
    image, f_label, od_label = dataset[id]
    image, (f_label, od_label) = ToPILImage()((image, (f_label, od_label)))

    f_x, f_y = f_label
    od_x, od_y = od_label

    thickness = 2
    w, h = 50, 50

    image = image.copy()
    ImageDraw.Draw(image).rectangle(
        ((f_x - w // 2, f_y - h // 2), (f_x + w // 2, f_y + h // 2)),
        outline="blue",
        width=thickness,
    )
    ImageDraw.Draw(image).rectangle(
        ((od_x - w // 2, od_y - h // 2), (od_x + w // 2, od_y + h // 2)),
        outline="blue",
        width=thickness,
    )
    plt.imshow(image)


def plot_coordinates(
    fundus: Union[np.ndarray, List[np.ndarray]],
    coordinates: Union[np.ndarray, List[np.ndarray]],
    axs=None,
    return_axs=None,
):
    """Plot the fundus image with the predicted fovea and optic disc coordinates

    Args:
        fundus (Union[np.ndarray, List[np.ndarray]]): Fundus image or list of fundus images
        coordinates (Union[np.ndarray, List[np.ndarray]]): Predicted coordinates of the fovea and optic disc
        axs ([type], optional): Matplotlib axis. Defaults to None.
        return_axs ([type], optional): Whether to return the axis. Defaults to None.

    Returns:
        axs: (List of) Matplotlib axis, if return_axs is True
    """

    if not isinstance(fundus, list):
        fundus = [fundus]
        coordinates = [coordinates]

    max_ncols = 10
    num_images = len(fundus)
    num_cols = min(num_images, max_ncols)
    num_rows = int(np.ceil(num_images / num_cols))

    if axs is None:
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4)
        )

    if num_images == 1 and not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    for i, f in enumerate(fundus):
        fx, fy, ox, oy = coordinates[i]
        axs[i].imshow(f)
        axs[i].scatter(fx, fy, c="g", label="Predicted Fovea Center")
        axs[i].scatter(ox, oy, c="b", label="Predicted OD Center")
        axs[i].legend()
        axs[i].axis("off")

    if return_axs:
        if num_images == 1:
            axs = axs[0]
        return axs

    else:
        plt.show()
        plt.close()
