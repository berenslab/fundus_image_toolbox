import numpy as np
import os
import yaml
from types import SimpleNamespace
from typing import List
from argparse import ArgumentParser
from pathlib import Path

from .dataset_multi import ODFoveaLoader
from .model_multi import ODFoveaModel
from .default import MODELS_DIR, DEFAULT_CSV_PATH, DEFAULT_MODEL, DEFAULT_CONFIG

import sys


class Parser(ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def get_ensemble(
    models_dir: str = MODELS_DIR,
    device: str = "cuda:0",
    return_test_dataloader: bool = False,
):
    ens = ["2024-05-07 11_13.05", "2024-05-07 11_13.13"]
    models = []
    checkpoints = []
    # for model_dir in [os.path.join(models_dir, x) for x in ens]:
    for model_dir in [Path(models_dir) / x for x in ens]:
        model, ckpt, test_dataloader = load_model(
            model_dir, device, return_test_dataloader=True
        )
        models.append(model)
        checkpoints.append(ckpt)

    if return_test_dataloader:
        return models, checkpoints, test_dataloader
    return models, checkpoints


def evaluate_ensemble(
    models: List[ODFoveaModel], checkpoints: List[str], test_dataloader: ODFoveaLoader
):
    test_loss, test_iou, test_dist, standardized_test_dist = [], [], [], []
    for model, ckpt in zip(models, checkpoints):
        loss, iou, dist, std_dist = model.evaluate(
            test_dataloader,
            checkpoint_path=ckpt,
            load_checkpoint=True,
            save_summary=False,
        )
        test_loss.append(loss)
        test_iou.append(iou)
        test_dist.append(dist)
        standardized_test_dist.append(std_dist)

    test_loss, test_iou, test_dist, standardized_test_dist = map(
        lambda x: sum(x) / len(x),
        [test_loss, test_iou, test_dist, standardized_test_dist],
    )

    return test_loss, test_iou, test_dist, standardized_test_dist


def predict_ensemble(models: List[ODFoveaModel], image_path: str):
    fovea_preds, od_preds = [], []
    for model in models:
        fovea_pred, od_pred = model.predict(image_path)
        fovea_preds.append(fovea_pred)
        od_preds.append(od_pred)

    # Mean of coordinates (x,y) of fovea and OD
    fovea_preds = np.mean(fovea_preds, axis=0)
    od_preds = np.mean(od_preds, axis=0)

    return fovea_preds, od_preds


def get_test_dataloader(config: SimpleNamespace):
    _, _, test_dataloader = ODFoveaLoader(config).get_dataloaders()
    return test_dataloader


def load_model(
    checkpoint_dir: str = "default",
    device: str = None,
    return_test_dataloader: bool = False,
):
    if checkpoint_dir == "default":
        checkpoint_dir = DEFAULT_MODEL
        config = DEFAULT_CONFIG
        checkpoint_dir = Path(checkpoint_dir)
    else:
        checkpoint_dir = Path(checkpoint_dir)
        # config = yaml.safe_load(open(os.path.join(checkpoint_dir, "config.yaml"), "r"))
        config = yaml.safe_load(open(checkpoint_dir / "config.yaml", "r"))
        config = SimpleNamespace(**config)

    if not MODELS_DIR in checkpoint_dir.parts:
        checkpoint_dir = Path(MODELS_DIR) / checkpoint_dir

    if config.csv_path is None:
        config.csv_path = DEFAULT_CSV_PATH
    if config.data_root is None:
        # config.data_root = "../../"
        config.data_root = Path(__file__).parent.parent

    config.device = device

    m_type = config.model_type
    model = ODFoveaModel(config)
    # model.checkpoint_path = os.path.join(checkpoint_dir, f"multi_{m_type}_best.pt")
    model.checkpoint_path = checkpoint_dir / f"multi_{m_type}_best.pt"
    model.load_checkpoint()

    # Load metric tracking data
    model.loss_tracking = {"train": [], "val": []}
    model.iou_tracking = {"train": [], "val": []}
    model.dist_tracking = {"train": [], "val": []}

    with open(checkpoint_dir / f"multi_{m_type}_train_loss.txt", "r") as f:
        for line in f:
            model.loss_tracking["train"].append(float(line.strip()))
    with open(checkpoint_dir / f"multi_{m_type}_val_loss.txt", "r") as f:
        for line in f:
            model.loss_tracking["val"].append(float(line.strip()))
    with open(checkpoint_dir / f"multi_{m_type}_train_iou.txt", "r") as f:
        for line in f:
            model.iou_tracking["train"].append(float(line.strip()))
    with open(checkpoint_dir / f"multi_{m_type}_val_iou.txt", "r") as f:
        for line in f:
            model.iou_tracking["val"].append(float(line.strip()))
    with open(checkpoint_dir / f"multi_{m_type}_train_dist.txt", "r") as f:
        for line in f:
            model.dist_tracking["train"].append(float(line.strip()))
    with open(checkpoint_dir / f"multi_{m_type}_val_dist.txt", "r") as f:
        for line in f:
            model.dist_tracking["val"].append(float(line.strip()))

    if return_test_dataloader:
        return model, model.checkpoint_path, get_test_dataloader(config)
    return model, model.checkpoint_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="If passed, other arguments are ignored and the model is trained using the config file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="efficientnet-b3",
        help="Type of model to train",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ],
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to train on"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, default=350, help="Size of the input image"
    )
    parser.add_argument(
        "--testset_eval",
        type=bool,
        default=True,
        help="Evaluate on test set after training",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../ADAM+IDRID+REFUGE_df.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../",
        help="Root folder that the csv entries refer to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for reproducibility. A different fixed seed is used for train-val-test split.",
    )
    config = parser.parse_args()

    if config.config is not None:
        config = yaml.safe_load(open(config.config, "r"))
        config = SimpleNamespace(**config)
    else:
        config = SimpleNamespace(**vars(config))

    print(f"Training {config.model_type} on {config.device}")
    if config.testset_eval:
        print("Evaluating on test set after training")

    train_dataloader, val_dataloader, test_dataloader = ODFoveaLoader(
        config
    ).get_dataloaders()
    model = ODFoveaModel(config)
    model.train(train_dataloader, val_dataloader)

    if config.testset_eval:
        model.evaluate(test_dataloader)

    model.plot_dist()
    model.plot_loss()
    model.plot_iou()
